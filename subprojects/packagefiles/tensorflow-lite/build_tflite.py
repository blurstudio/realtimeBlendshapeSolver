"""Configure and build TFLite's C library with CMake, then copy it to meson's output.

Called by this subproject's meson.build. Configure only reruns when the CMake
arguments change; otherwise it's an incremental `cmake --build`.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def find_built_lib(build_dir, lib_name):
    # Multi-config generators (Visual Studio) put it in Release/; Ninja and
    # Makefiles put it at the top. Skip the FetchContent trees.
    hits = [
        p for p in glob.glob(os.path.join(build_dir, "**", lib_name), recursive=True)
        if "_deps" not in os.path.relpath(p, build_dir).split(os.sep)
    ]
    if not hits:
        sys.exit("build finished but %s was not found under %s" % (lib_name, build_dir))
    return max(hits, key=os.path.getmtime)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmake", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--build-dir", required=True)
    p.add_argument("--lib-name", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--generator", default="")
    p.add_argument("cmake_args", nargs="*")
    args = p.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    os.makedirs(build_dir, exist_ok=True)

    configure = [args.cmake, "-S", args.source, "-B", build_dir]
    if args.generator:
        configure += ["-G", args.generator]
    configure += args.cmake_args

    stamp = os.path.join(build_dir, "meson_configure_args.txt")
    want = "\n".join(configure)
    have = open(stamp).read() if os.path.isfile(stamp) else None
    if have != want or not os.path.isfile(os.path.join(build_dir, "CMakeCache.txt")):
        run(configure)
        with open(stamp, "w") as f:
            f.write(want)

    run([
        args.cmake, "--build", build_dir, "--config", "Release",
        "--target", "tensorflowlite_c", "--parallel", str(os.cpu_count() or 4),
    ])

    built = find_built_lib(build_dir, args.lib_name)
    shutil.copy2(built, args.output)
    print("copied", built, "->", args.output)


if __name__ == "__main__":
    main()
