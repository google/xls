# Lint as: python3
#
# Copyright 2020 Google LLC
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

"""Utility for turning an XLS text file into a pretty printed token sequence."""

import pprint
from absl import app

from xls.dslx.scanner import Scanner


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')

  path = argv[1]

  with open(path, 'r') as f:
    text = f.read()

  pprint.pprint(Scanner(path, text).pop_all())


if __name__ == '__main__':
  app.run(main)
