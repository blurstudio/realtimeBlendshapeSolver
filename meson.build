project(
  'MayaTfLiteArkitSolver', 'cpp',
  # b_vscrt=md: Maya uses the release CRT, so the plugin does too, even in a
  # debug build.
  default_options : ['cpp_std=c++17', 'b_vscrt=md'],
  meson_version : '>=1.1',
)

maya_dep = dependency('maya')
maya_gl_dep = dependency('maya-opengl')
maya_name_suffix = maya_dep.get_variable('name_suffix')
maya_version = maya_dep.get_variable('maya_version')

# Header-only. Uses a system Eigen if there is one, else subprojects/eigen.wrap.
eigen_dep = dependency('eigen3', version : '>=3.4')

solver_files = files(
  'src/blendshapeModel.cpp',
  'src/faceBlendshapes.cpp',
)

source_files = solver_files + files(
  'src/tfliteArkitSolverNode.cpp',
  'src/pluginMain.cpp',
)

# If a user-built version file exists, then just use that
# Otherwise grab the latest tag from git
fs = import('fs')
if fs.is_file('src/version.h')
  message('Using existing version.h')
else
  git = find_program('git', native: true, required: true)
  version_h = vcs_tag(
    command: [git, 'describe', '--tags', '--match', 'v[0-9]*', '--dirty=+'],
    fallback: 'v0.0.1',
    input: 'src/version.h.in',
    output: 'version.h',
  )
  source_files = source_files + version_h
endif

install_dir = meson.global_source_root() / 'output_Maya' + maya_version

outlib = shared_library(
  meson.project_name(),
  source_files,
  install: true,
  install_dir : install_dir,
  include_directories : include_directories(['src']),
  dependencies : [maya_dep, maya_gl_dep, eigen_dep],
  name_prefix : '',
  name_suffix : maya_name_suffix,
)

# Runtime files are installed next to the plugin, which is where it looks for
# them by default.
model_file = 'thirdparty/model/face_blendshapes.tflite'
obj_file = 'thirdparty/model/canonical_face_model.obj'

if fs.is_file(model_file)
  install_data(model_file, install_dir : install_dir)
else
  warning('Model not found. Run: python -I scripts/fetch_dependencies.py')
endif

# Standalone (no Maya) check of the model implementation and input conventions.
# Run with: meson test -C <builddir>
solver_test = executable(
  'faceBlendshapesTest',
  solver_files + files('tests/faceBlendshapesTest.cpp'),
  include_directories : include_directories(['src']),
  dependencies : eigen_dep,
  build_by_default : false,
)
if fs.is_file(model_file) and fs.is_file(obj_file)
  test('faceBlendshapes', solver_test, args : [
    meson.global_source_root() / model_file,
    meson.global_source_root() / obj_file,
  ])
endif
