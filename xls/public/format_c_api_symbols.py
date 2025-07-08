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

"""Helper tool for formatting c_api_symbols.txt."""

import os
import sys


def main() -> None:
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  c_api_symbols_path = os.path.join(cur_dir, 'c_api_symbols.txt')
  if not os.path.exists(c_api_symbols_path):
    raise FileNotFoundError(f'c_api_symbols.txt not found at {c_api_symbols_path}')


  with open(c_api_symbols_path, 'r') as f:
    lines = f.readlines()

  lines = sorted(set(lines))
  with open(c_api_symbols_path, 'w') as f:
    for line in lines:
      line = line.strip()
      f.write(line + '\n')

  print(f'Formatted {c_api_symbols_path} with {len(lines)} unique symbols.')

if __name__ == '__main__':
  sys.exit(main())
