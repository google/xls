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

"""Produces the ANSI-syntax-colored version of the input file on stdout.

Example:

  highlight_main <path>
"""

import sys
from typing import Text

from absl import app
from absl import flags
import termcolor

from xls.dslx.python import cpp_scanner as scanner

FLAGS = flags.FLAGS


class AnsiHighlightHandler(scanner.HighlightHandler):
  """Wraps token text with ANSI-coloring escape sequences."""

  def handle_keyword(self, s: Text) -> Text:
    return termcolor.colored(s, color='yellow')

  def handle_number(self, s: Text) -> Text:
    return termcolor.colored(s, color='red')

  def handle_comment(self, s: Text) -> Text:
    return termcolor.colored(s, color='blue')

  def handle_builtin(self, s: Text) -> Text:
    return termcolor.colored(s, color='cyan')

  def handle_type(self, s: Text) -> Text:
    return termcolor.colored(s, color='green')

  def handle_other(self, s: Text) -> Text:
    return s


def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')

  path = argv[1]
  with open(path) as f:
    contents = f.read()

  handler = AnsiHighlightHandler()
  s = scanner.Scanner(path, contents, include_whitespace_and_comments=True)
  while not s.at_eof():
    t = s.pop()
    sys.stdout.write(t.to_highlight_str(handler))


if __name__ == '__main__':
  app.run(main)
