project(
  'MayaTfLiteArkitSolver', 'cpp',
  default_options : ['cpp_std=c++17'],
  meson_version : '>=1.1',
)

maya_dep = dependency('maya')
maya_gl_dep = dependency('maya-opengl')
maya_name_suffix = maya_dep.get_variable('name_suffix')
maya_version = maya_dep.get_variable('maya_version')

# The TFLite C library is loaded at runtime (see src/tfliteRuntime.h), so there
# is nothing to link against except the platform's dynamic loader.
dl_dep = dependency('dl', required : host_machine.system() != 'windows')

solver_files = files(
  'src/faceBlendshapes.cpp',
  'src/tfliteRuntime.cpp',
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
  dependencies : [maya_dep, maya_gl_dep, dl_dep],
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

# The TFLite C library. `tflite_lib` is a path or a build target, for the test.
tflite_lib = ''
have_tflite_lib = false
if get_option('tflite') == 'source'
  # Downloads tensorflow (subprojects/tensorflow-lite.wrap) and builds
  # tensorflowlite_c with CMake. The first build takes a while.
  tflite_proj = subproject(
    'tensorflow-lite',
    default_options : ['lib_install_dir=' + install_dir],
  )
  tflite_lib = tflite_proj.get_variable('tflite_c_lib')
  have_tflite_lib = true
else
  foreach name : ['libLiteRt.dll', 'libLiteRt.so', 'libLiteRt.dylib']
    if tflite_lib == '' and fs.is_file('thirdparty/litert' / name)
      tflite_lib = meson.global_source_root() / 'thirdparty/litert' / name
    endif
  endforeach
  have_tflite_lib = tflite_lib != ''
  if have_tflite_lib
    install_data(tflite_lib, install_dir : install_dir)
  else
    warning('LiteRT library not found. Run: python -I scripts/fetch_dependencies.py')
  endif
endif

# Standalone (no Maya) check of the TFLite binding and model conventions.
# Run with: meson test -C <builddir>
solver_test = executable(
  'faceBlendshapesTest',
  solver_files + files('tests/faceBlendshapesTest.cpp'),
  include_directories : include_directories(['src']),
  dependencies : [dl_dep],
  build_by_default : false,
)
if have_tflite_lib and fs.is_file(model_file) and fs.is_file(obj_file)
  test('faceBlendshapes', solver_test, args : [
    tflite_lib,
    meson.global_source_root() / model_file,
    meson.global_source_root() / obj_file,
  ])
endif
