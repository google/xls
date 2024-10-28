# Copyright 2024 The XLS Authors
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

"""Tests for public artifacts exposing extern "C" symbols."""

import os
import subprocess
import unittest


class TestCApiSymbols(unittest.TestCase):
  """Tests for public artifacts exposing extern "C" symbols."""

  def test_symbols_match(self):
    """Tests c_api_symbols.txt matches extern C symbols in .a files."""
    # Get the runfiles directory
    runfiles_dir = os.environ['TEST_SRCDIR']

    # Construct paths
    workspace_name = os.environ['TEST_WORKSPACE']
    c_api_symbols_path = os.path.join(
        runfiles_dir, workspace_name, 'xls/public/c_api_symbols.txt'
    )
    static_libs_dir = os.path.join(
        runfiles_dir, workspace_name, 'xls/public'
    )

    # Read symbols from c_api_symbols.txt
    with open(c_api_symbols_path, 'r') as f:
      expected_symbols = set(line.strip() for line in f if line.strip())

    # Find all .a files in the static_libs_dir
    static_libs = []
    for root, _, files in os.walk(static_libs_dir):
      for file in files:
        if file.endswith('.a'):
          static_libs.append(os.path.join(root, file))

    # Extract symbols from the .a files
    actual_symbols = set()
    for lib in static_libs:
      # Run nm on each .a file
      result = subprocess.run(
          ['nm', lib],
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          check=True,
      )
      if result.returncode != 0:
        self.fail(f'nm failed on {lib} with error: {result.stderr}')
      for line in result.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[1] in {'T', 'R', 'D'}:
          symbol = parts[2]
          if symbol.startswith('xls_'):
            actual_symbols.add(symbol)
          elif symbol.startswith('_xls_'):
            actual_symbols.add(symbol[1:])

    # Compare the symbols
    self.assertEqual(
        expected_symbols,
        actual_symbols,
        'Mismatch between c_api_symbols.txt and symbols extracted from .a'
        ' files.',
    )


if __name__ == '__main__':
  unittest.main()
