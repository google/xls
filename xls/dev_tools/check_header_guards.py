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

# pylint: disable=missing-function-docstring

"""Script to run from XLS repo root, checks if header guards are compliant."""

import os
import re
import sys


def get_expected_guard(filepath, repo_root):
  # Convert the file path relative to repo root to an uppercase header guard
  # format.
  relative_path = os.path.relpath(filepath, repo_root)
  guard = relative_path.upper().replace(os.sep, '_').replace('.', '_') + '_'
  return guard


def check_header_guard(filepath, expected_guard):
  with open(filepath, encoding='utf-8') as file:
    lines = file.readlines()

  # Check for the presence of the expected header guard.
  guard_pattern = re.compile(r'#ifndef\s+(\S+)')
  for line in lines:
    match = guard_pattern.match(line.strip())
    if match:
      actual_guard = match.group(1)
      return actual_guard == expected_guard, actual_guard

  return False, None


def find_h_files(repo_root):
  # Find all `.h` files within the repo root, excluding `xls/contrib` and
  # `third_' + `party`.
  h_files = []
  for root, _, files in os.walk(repo_root):
    if 'xls/contrib' in root or 'third_' + 'party' in root:
      continue
    for file in files:
      if file.endswith('.h'):
        h_files.append(os.path.join(root, file))
  return h_files


def main():
  repo_root = os.path.join(os.getcwd())
  h_files = find_h_files(repo_root)

  non_compliant_files = []

  for h_file in h_files:
    expected_guard = get_expected_guard(h_file, repo_root)
    compliant, actual_guard = check_header_guard(h_file, expected_guard)
    if not compliant:
      non_compliant_files.append((h_file, expected_guard, actual_guard))

  if non_compliant_files:
    print('Non-style-compliant header files:')
    for file, expected, actual in non_compliant_files:
      print(file)
      print(f'  want: {expected}')
      actual_str = actual if actual else 'None'
      print(f'  got: {actual_str}')
    sys.exit(-1)
  else:
    print('All header files are style compliant.')


if __name__ == '__main__':
  main()
