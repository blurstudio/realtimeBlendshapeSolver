// fbs::BlendshapeModel: runs face_blendshapes.tflite without TensorFlow Lite.
//
// The model is a small MLP-Mixer. Every CONV_2D in it is 1x1, so it's a dense
// layer. In row-major [rows, cols] terms, the graph is:
//
//   p   = (xy - mean(xy)) / mean(|xy - mean(xy)|) * inputScale     [146, 2]
//   e   = W0 p + b0                    mix the 146 points into 96  [96, 2]
//   x   = [cls; e W1^T + b1]           2 -> 64 channels, prepend   [97, 64]
//   4 x { x += W3 relu(W2 ln(x) + b2) + b3     token mixing (97 -> 384 -> 97)
//         x += relu(ln(x) W4^T + b4) W5^T + b5 channel mixing (64 -> 256 -> 64) }
//   out = sigmoid(x[0] W18^T + b18)                                [52]
//
// ln is a layer norm over the 64 channels, with a gamma and no beta.
//
// load() reads those weights out of the .tflite flatbuffer by walking the graph
// in order. It checks every shape, so a different model fails to load rather
// than giving wrong results. This is not a general TFLite runtime.

#include "faceBlendshapes.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <vector>

namespace fbs {

namespace {

using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Bounds-checked reads from a flatbuffer. The model file is user input, so
// every read throws on an offset outside the file instead of trusting it.
class Flatbuf {
  public:
    explicit Flatbuf(std::vector<uint8_t> bytes) : m_bytes(std::move(bytes)) {}

    template <typename T> T get(size_t at) const
    {
        if (at > m_bytes.size() || m_bytes.size() - at < sizeof(T))
            throw std::runtime_error("truncated or corrupt model file");
        T v;
        std::memcpy(&v, m_bytes.data() + at, sizeof(T));
        return v;
    }

    const uint8_t *bytes(size_t at, size_t n) const
    {
        if (at > m_bytes.size() || m_bytes.size() - at < n)
            throw std::runtime_error("truncated or corrupt model file");
        return m_bytes.data() + at;
    }

    bool hasIdentifier(const char *id) const
    {
        return m_bytes.size() >= 8 && std::memcmp(m_bytes.data() + 4, id, 4) == 0;
    }

    size_t root() const { return get<uint32_t>(0); }

    // Position of field `i` of the table at `t`, or 0 if it's absent.
    size_t field(size_t t, int i) const
    {
        const int64_t vt = int64_t(t) - get<int32_t>(t);
        if (vt < 0) throw std::runtime_error("corrupt model file");
        const size_t slot = 4 + 2 * size_t(i);
        if (slot + 2 > get<uint16_t>(size_t(vt))) return 0;
        const uint16_t off = get<uint16_t>(size_t(vt) + slot);
        return off ? t + off : 0;
    }

    template <typename T> T scalar(size_t t, int i, T dflt = 0) const
    {
        const size_t f = field(t, i);
        return f ? get<T>(f) : dflt;
    }

    // Table, vector or string field: follow its offset. 0 if absent.
    size_t ref(size_t t, int i) const
    {
        const size_t f = field(t, i);
        return f ? f + get<uint32_t>(f) : 0;
    }

    struct Vec {
        size_t data = 0;
        uint32_t size = 0;
    };

    Vec vec(size_t t, int i) const
    {
        const size_t v = ref(t, i);
        return v ? Vec{v + 4, get<uint32_t>(v)} : Vec{};
    }

    // Element k of a vector of tables.
    size_t tableAt(Vec v, uint32_t k) const
    {
        if (k >= v.size) throw std::runtime_error("corrupt model file");
        const size_t at = v.data + 4 * size_t(k);
        return at + get<uint32_t>(at);
    }

    std::vector<int32_t> ints(size_t t, int i) const
    {
        const Vec v = vec(t, i);
        std::vector<int32_t> out(v.size);
        for (uint32_t k = 0; k < v.size; ++k) out[k] = get<int32_t>(v.data + 4 * size_t(k));
        return out;
    }

  private:
    std::vector<uint8_t> m_bytes;
};

// Field indices from tensorflow/compiler/mlir/lite/schema/schema.fbs. A union
// takes two slots (type, then value), so Operator.builtin_options is 4, not 3.
namespace schema {
enum Model { kOperatorCodes = 1, kSubgraphs = 2, kBuffers = 4 };
enum OperatorCode { kDeprecatedBuiltinCode = 0, kBuiltinCode = 3 };
enum SubGraph { kTensors = 0, kInputs = 1, kOutputs = 2, kOperators = 3 };
enum Tensor { kShape = 0, kType = 1, kBuffer = 2 };
enum Operator { kOpcodeIndex = 0, kOpInputs = 1, kOpOutputs = 2, kBuiltinOptions = 4 };
enum Buffer { kData = 0 };
enum Conv2DOptions { kStrideW = 1, kStrideH = 2, kActivation = 3 };
enum TensorType { kFloat32 = 0, kFloat16 = 1 };
enum BuiltinOp { kAdd = 0, kConcatenation = 2, kConv2D = 3, kDequantize = 6, kMul = 18 };
enum Activation { kNone = 0, kRelu = 1 };
} // namespace schema

struct Dense {
    Mat w; // [out, in]
    Eigen::VectorXf b;
    bool relu = false;
};

// act(x W^T + b), applied to each row of x.
Mat denseRows(const Dense &d, const Mat &x)
{
    Mat y = (x * d.w.transpose()).rowwise() + d.b.transpose();
    return d.relu ? Mat(y.cwiseMax(0.0f)) : y;
}

// act(W x + b), applied to each column of x.
Mat denseCols(const Dense &d, const Mat &x)
{
    Mat y = (d.w * x).colwise() + d.b;
    return d.relu ? Mat(y.cwiseMax(0.0f)) : y;
}

Mat layerNorm(const Mat &x, const Eigen::RowVectorXf &gamma, float eps)
{
    const Mat c = x.colwise() - x.rowwise().mean();
    const Eigen::ArrayXf inv = (c.array().square().rowwise().mean() + eps).rsqrt();
    return (c.array().colwise() * inv).rowwise() * gamma.array();
}

constexpr int kNumBlocks = 4;
constexpr int kTokens = 97, kChannels = 64, kTokenHidden = 384, kChannelHidden = 256;

// Expected [out, in] of each 1x1 conv, in graph order.
std::vector<std::pair<int, int>> expectedDenseShapes()
{
    std::vector<std::pair<int, int>> s = {{96, kNumSubset}, {kChannels, 2}};
    for (int i = 0; i < kNumBlocks; ++i) {
        s.push_back({kTokenHidden, kTokens});
        s.push_back({kTokens, kTokenHidden});
        s.push_back({kChannelHidden, kChannels});
        s.push_back({kChannels, kChannelHidden});
    }
    s.push_back({kNumBlendshapes, kChannels});
    return s;
}

} // namespace

struct BlendshapeModel::Weights {
    float inputScale = 0, eps = 0;
    Eigen::RowVectorXf cls;
    std::vector<Dense> dense;              // expectedDenseShapes() order
    std::vector<Eigen::RowVectorXf> gamma; // token, channel for each block
};

namespace {

using Weights = BlendshapeModel::Weights;

Weights parseWeights(const Flatbuf &fb)
{
    using namespace schema;
    const auto fail = [](const std::string &why) {
        throw std::runtime_error(why + "; is this face_blendshapes.tflite?");
    };
    if (!fb.hasIdentifier("TFL3")) fail("not a TFLite model");

    const size_t model = fb.root();
    const Flatbuf::Vec opcodes = fb.vec(model, kOperatorCodes);
    const Flatbuf::Vec subgraphs = fb.vec(model, kSubgraphs);
    const Flatbuf::Vec buffers = fb.vec(model, kBuffers);
    if (subgraphs.size != 1) fail("expected one subgraph");
    const size_t sg = fb.tableAt(subgraphs, 0);
    const Flatbuf::Vec tensors = fb.vec(sg, kTensors);
    const Flatbuf::Vec ops = fb.vec(sg, kOperators);

    const auto tensor = [&](int32_t i) { return fb.tableAt(tensors, uint32_t(i)); };
    const auto shape = [&](int32_t i) { return fb.ints(tensor(i), kShape); };
    const auto opcode = [&](size_t op) {
        const size_t oc = fb.tableAt(opcodes, fb.scalar<uint32_t>(op, kOpcodeIndex));
        // Older files only set the int8 deprecated field; newer ones set both.
        return std::max<int32_t>(
            fb.scalar<int32_t>(oc, kBuiltinCode), fb.scalar<int8_t>(oc, kDeprecatedBuiltinCode)
        );
    };

    // The float16 weights reach the graph through DEQUANTIZE ops.
    std::map<int32_t, int32_t> dequantized; // output tensor -> its float16 input
    for (uint32_t k = 0; k < ops.size; ++k) {
        const size_t op = fb.tableAt(ops, k);
        if (opcode(op) == kDequantize)
            dequantized[fb.ints(op, kOpOutputs).at(0)] = fb.ints(op, kOpInputs).at(0);
    }

    // A constant tensor's values, or empty if it's computed at run time.
    const auto values = [&](int32_t i) {
        if (dequantized.count(i)) i = dequantized[i];
        const size_t t = tensor(i);
        const Flatbuf::Vec data = fb.vec(fb.tableAt(buffers, fb.scalar<uint32_t>(t, kBuffer)), kData);
        std::vector<float> out;
        const int8_t type = fb.scalar<int8_t>(t, kType);
        if (type == kFloat32) {
            out.resize(data.size / 4);
            if (!out.empty()) std::memcpy(out.data(), fb.bytes(data.data, data.size), out.size() * 4);
        } else if (type == kFloat16) {
            const uint8_t *p = fb.bytes(data.data, data.size);
            out.resize(data.size / 2);
            for (size_t j = 0; j < out.size(); ++j) {
                uint16_t h;
                std::memcpy(&h, p + 2 * j, 2);
                out[j] = float(Eigen::numext::bit_cast<Eigen::half>(h));
            }
        }
        return out;
    };

    // The graph's single input and output.
    const std::vector<int32_t> in = fb.ints(sg, kInputs), out = fb.ints(sg, kOutputs);
    if (in.size() != 1 || out.size() != 1) fail("unexpected tensor count");
    if (shape(in[0]) != std::vector<int32_t>{1, kNumSubset, 2} ||
        shape(out[0]) != std::vector<int32_t>{kNumBlendshapes} ||
        fb.scalar<int8_t>(tensor(in[0]), kType) != kFloat32 ||
        fb.scalar<int8_t>(tensor(out[0]), kType) != kFloat32)
        fail("unexpected tensor shapes; expected float32 [1,146,2] -> [52]");

    Weights w;
    bool haveScale = false, haveEps = false;
    for (uint32_t k = 0; k < ops.size; ++k) {
        const size_t op = fb.tableAt(ops, k);
        const int32_t code = opcode(op);
        const std::vector<int32_t> ins = fb.ints(op, kOpInputs);
        if (code == kConv2D) {
            if (ins.size() != 3) fail("CONV_2D without a bias");
            const std::vector<int32_t> s = shape(ins[1]);
            const size_t opts = fb.ref(op, kBuiltinOptions);
            const int8_t act = opts ? fb.scalar<int8_t>(opts, kActivation) : 0;
            if (s.size() != 4 || s[1] != 1 || s[2] != 1 || !opts ||
                fb.scalar<int32_t>(opts, kStrideW) != 1 || fb.scalar<int32_t>(opts, kStrideH) != 1 ||
                (act != kNone && act != kRelu))
                fail("CONV_2D that isn't a 1x1 dense layer");
            Dense d;
            const std::vector<float> wv = values(ins[1]), bv = values(ins[2]);
            if (wv.size() != size_t(s[0]) * s[3] || bv.size() != size_t(s[0]))
                fail("CONV_2D weights are not constant");
            d.w = Eigen::Map<const Mat>(wv.data(), s[0], s[3]);
            d.b = Eigen::Map<const Eigen::VectorXf>(bv.data(), s[0]);
            d.relu = act == kRelu;
            w.dense.push_back(std::move(d));
            continue;
        }
        if (code != kMul && code != kAdd && code != kConcatenation) continue;
        for (int32_t i : ins) {
            const std::vector<float> v = values(i);
            if (v.empty()) continue;
            const Eigen::Map<const Eigen::RowVectorXf> row(v.data(), Eigen::Index(v.size()));
            if (code == kConcatenation) w.cls = row;
            else if (code == kMul && v.size() == size_t(kChannels)) w.gamma.push_back(row);
            else if (code == kMul && v.size() == 1 && !haveScale) {
                w.inputScale = v[0];
                haveScale = true;
            } else if (code == kAdd && v.size() == 1) {
                w.eps = v[0];
                haveEps = true;
            } else fail("unexpected constant operand");
        }
    }

    const auto shapes = expectedDenseShapes();
    if (w.dense.size() != shapes.size()) fail("unexpected number of layers");
    for (size_t i = 0; i < shapes.size(); ++i)
        if (w.dense[i].w.rows() != shapes[i].first || w.dense[i].w.cols() != shapes[i].second)
            fail("unexpected layer shape");
    if (w.gamma.size() != 2 * kNumBlocks || w.cls.size() != kChannels || !haveScale || !haveEps)
        fail("unexpected model structure");
    return w;
}

} // namespace

BlendshapeModel::BlendshapeModel() = default;
BlendshapeModel::~BlendshapeModel() = default;

void BlendshapeModel::unload()
{
    m_weights.reset();
    m_path.clear();
}

bool BlendshapeModel::load(const std::string &modelPath, std::string &error)
{
    unload();
    std::ifstream f(modelPath, std::ios::binary);
    if (!f) {
        error = "Could not open model file: " + modelPath;
        return false;
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    try {
        m_weights = std::make_unique<const Weights>(parseWeights(Flatbuf(std::move(bytes))));
    } catch (const std::exception &e) {
        error = "Could not load model " + modelPath + ": " + e.what();
        return false;
    }
    m_path = modelPath;
    return true;
}

bool BlendshapeModel::run(const float *xy, float *scores, std::string &error) const
{
    if (!m_weights) {
        error = "Model is not loaded";
        return false;
    }
    const Weights &w = *m_weights;
    const std::vector<Dense> &d = w.dense;

    const Eigen::Map<const Mat> points(xy, kNumSubset, 2);
    Mat p = points.rowwise() - points.colwise().mean();
    p *= w.inputScale / p.rowwise().norm().mean();

    Mat x(kTokens, kChannels);
    x.row(0) = w.cls;
    x.bottomRows(kTokens - 1) = denseRows(d[1], denseCols(d[0], p));
    for (int b = 0; b < kNumBlocks; ++b) {
        const Dense *layer = &d[2 + 4 * b];
        x += denseCols(layer[1], denseCols(layer[0], layerNorm(x, w.gamma[2 * b], w.eps)));
        x += denseRows(layer[3], denseRows(layer[2], layerNorm(x, w.gamma[2 * b + 1], w.eps)));
    }
    const Mat logits = denseRows(d.back(), x.topRows(1));
    Eigen::Map<Eigen::RowVectorXf>(scores, kNumBlendshapes) =
        (1.0f + (-logits.array()).exp()).inverse().matrix();
    return true;
}

} // namespace fbs
