# Builds the TensorFlow Lite C API shared library (tensorflowlite_c) by driving
# TFLite's own CMake project. TFLite pulls about ten third-party dependencies
# (abseil, eigen, flatbuffers, ruy, ...) with CMake FetchContent, so this is
# much more reliable than translating it to meson.
#
# The CMake configure step downloads those dependencies, so the first build
# needs network access and takes a while. Later builds reuse the CMake build dir.
#
# Exposes:
#   tflite_c_lib  custom_target whose output is the shared library
#   tflite_c_inc  include_directories for the C API headers (tensorflow/lite/c/c_api.h)
project('tensorflow-lite', version : '2.21.0', meson_version : '>=1.1')

py = import('python').find_installation()
cmake = find_program('cmake', version : '>=3.16')

system = host_machine.system()
if system == 'windows'
  lib_name = 'tensorflowlite_c.dll'
elif system == 'darwin'
  lib_name = 'libtensorflowlite_c.dylib'
else
  lib_name = 'libtensorflowlite_c.so'
endif

cmake_build_dir = get_option('cmake_build_dir')
if cmake_build_dir == ''
  cmake_build_dir = meson.current_build_dir() / 'cmake'
endif

cmake_args = [
  # TFLite is always built optimized. It only exposes a C API, so it doesn't
  # need to match the plugin's debug/release CRT.
  '-DCMAKE_BUILD_TYPE=Release',
  # Without this, tensorflow/lite/CMakeLists.txt git-clones a hard-coded (and,
  # in 2.21.0, stale: v2.19.0) tensorflow and mixes its headers with ours.
  '-DTENSORFLOW_SOURCE_DIR=' + meson.current_source_dir(),
  '-DTFLITE_ENABLE_XNNPACK=' + (get_option('xnnpack') ? 'ON' : 'OFF'),
  '-DTFLITE_ENABLE_GPU=OFF',
  '-DTFLITE_ENABLE_NNAPI=OFF',
  '-DTFLITE_ENABLE_EXTERNAL_DELEGATE=OFF',
] + get_option('cmake_args')

if system == 'windows'
  cmake_args += '-DTFLITE_ENABLE_MMAP=OFF'
endif

build_cmd = [
  py, files('build_tflite.py'),
  '--cmake', cmake,
  '--source', meson.current_source_dir() / 'tensorflow' / 'lite' / 'c',
  '--build-dir', cmake_build_dir,
  '--lib-name', lib_name,
  '--output', '@OUTPUT0@',
]
if get_option('cmake_generator') != ''
  build_cmd += ['--generator', get_option('cmake_generator')]
endif
build_cmd += ['--'] + cmake_args

install_dir = get_option('lib_install_dir')
tflite_c_lib = custom_target(
  'tensorflowlite_c',
  output : lib_name,
  command : build_cmd,
  console : true,
  build_by_default : true,
  install : install_dir != '',
  install_dir : install_dir != '' ? install_dir : get_option('libdir'),
)

tflite_c_inc = include_directories('.')
