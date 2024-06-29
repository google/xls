#!/usr/bin/env python3
#
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=broad-exception-raised
"""Script for project-specific release of the XLS toolchain.

Usage:
   make_release.py TARGET_DIR

Must be run from a top-level build directory. This creates a directory
TARGET_DIR containing XLS binaries built in this client
"""

import os
import shutil
import subprocess
import sys

XLS_ROOT = 'xls'

# All paths are relative to XLS_ROOT
BINARIES = (
    'dslx/interpreter_main',
    'dslx/ir_convert/ir_converter_main',
    'tools/opt_main',
    'tools/codegen_main',
    'visualization/ir_viz/app',
)
PACKAGE_UTIL = 'tools/package_bazel_build'

# BUILD file to write into the release directory.
BUILD_FILE = """package(
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"], # Apache 2.0
)

exports_files(glob(["*"]))
"""

# WORKSPACE file to write into the release directory
WORKSPACE_FILE = """workspace(name = "com_google_xls")
"""


def build_and_copy_binaries(wc_dir, target_dir):
  """Builds and copies the necessary binaries to the target directory."""
  # TODO(meheff): We should verify that the client is clean.
  targets = [os.path.join(XLS_ROOT, b) for b in BINARIES]
  package_target = os.path.join(XLS_ROOT, PACKAGE_UTIL)

  bazel_args = ['bazel', 'build', '-c', 'opt'] + targets + [package_target]
  print('Running:\n  ' + ' '.join(bazel_args))
  subprocess.check_call(
      ['bazel', 'build', '-c', 'opt'] + targets + [package_target], cwd=wc_dir
  )

  print(f'Copying binaries to {target_dir}')

  # Build package command up
  package_args = ['./bazel-bin/' + package_target]
  package_args.extend(['--output_dir', target_dir])
  for t in targets:
    package_args.extend(['--inc_target', t])

  subprocess.check_call(package_args, cwd=wc_dir)


def write_build_file(target_dir):
  """Writes a BUILD file to the target directory."""
  build_path = os.path.join(target_dir, 'BUILD')
  print(f'Writing BUILD file {build_path}')
  with open(build_path, 'w') as f:
    f.write(BUILD_FILE)


def write_workspace_file(target_dir):
  """Writes a WORKSPACE file to the target directory."""
  build_path = os.path.join(target_dir, 'WORKSPACE')
  print(f'Writing WORKSPACE file {build_path}')
  with open(build_path, 'w') as f:
    f.write(WORKSPACE_FILE)


def copy_xls_info(target_dir):
  """Writes a XLS license and info  to the target directory."""
  shutil.copy('LICENSE', target_dir)
  shutil.copy('README.md', target_dir)
  shutil.copy('CONTRIBUTING.md', target_dir)


def main(args):
  if len(args) != 1:
    raise Exception('Usage: make_release.py TARGET_DIR')

  target_dir = args[0]
  if os.path.exists(target_dir):
    raise Exception(f'{target_dir} already exists')

  print(f'Creating release directory {target_dir}')
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  build_and_copy_binaries(os.getcwd(), target_dir)

  write_build_file(target_dir)
  write_workspace_file(target_dir)
  copy_xls_info(target_dir)


if __name__ == '__main__':
  main(sys.argv[1:])
