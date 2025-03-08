#!/usr/bin/env python3

# Copyright 2025 The XLS Authors
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

"""Checks that C++ files only include absolute paths.

This prevents accidental inclusion via relpath even if it happens to work.
"""

import sys
import re

ALLOWED_INCLUDE_STARTS = {
  'xls/',
  'absl/',
  'gmock/',
  'gtest/',
  'benchmark/',
  'llvm/',
  'fuzztest/',
  'verible/',
  're2/',
  'z3/',
  'google/protobuf/',
  'grpcpp/',
  'grpc/',
  'openssl/',
  'nlohmann/',
  'cppitertools/',
  'ortools/',
  'external/zstd/',
  'riegeli/',
  'tools/cpp/runfiles',
  'linenoise.h',
  'libs/json11/',
  '%s',  # For format strings embedded in files.
}

def check_file(filename):
  with open(filename, encoding='utf-8') as f:
    content = f.read()

  # Look for quoted (non-system) includes
  includes = re.findall(r'#include\s*"([^"]*)"', content)
  bad_includes = [
    inc for inc in includes
    if not any(inc.startswith(start) for start in ALLOWED_INCLUDE_STARTS)]

  if bad_includes:
    print(f'{filename}: Found non-absolute includes:')
    for include in bad_includes:
      print(f'  {include}')
    return 1
  return 0

def main():
  exit_code = 0
  for filename in sys.argv[1:]:
    exit_code |= check_file(filename)
  sys.exit(exit_code)

if __name__ == '__main__':
  main()
