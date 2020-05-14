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

"""Helper functions for use with the parser.Parser."""

import builtins
import functools
import io
import os
import sys
from typing import Text, Optional, Callable

import termcolor

from xls.dslx import ast
from xls.dslx import parser
from xls.dslx import scanner
from xls.dslx import span as span_mod


def pprint_positional_error(error: span_mod.PositionalError,
                            *,
                            output: io.IOBase = sys.stderr,
                            color: Optional[bool] = None,
                            error_context_line_count: int = 5,
                            fs_open: Optional[Callable] = None) -> None:
  """Prints pretty message to output for a error with a position.

  ANSI color escapes are used when output appears to be a tty.

  Args:
    error: The parse error that was raised in our attempt to parse.
    output: File to which we'll print the pretty error text.
    color: If provided, forces color on (True) or forces color off (False).
      Otherwise attempts to detect whether the output environment supports
      color.
    error_context_line_count: Number of context lines to print around the error
      position.
    fs_open: If not None, will be used in lieu of builtins.open; useful for e.g.
      faking out filesystems in test environments.

  Raises:
    ValueError: if the error_context_line_count is not odd (only odd values can
      be symmetrical around the erroneous line).
  """
  assert isinstance(error, span_mod.PositionalError), error
  if error.printed:
    return
  error.printed = True

  if error_context_line_count % 2 != 1:
    raise ValueError('Expected odd error_context_line_count; got {}'.format(
        error_context_line_count))
  span = error.span
  # If nobody is hooking "open", use the built-in one.
  fs_open = fs_open or builtins.open
  with fs_open(error.filename) as f:
    text = f.read()
  lines = text.splitlines()
  if span.limit.lineno >= len(lines):
    raise ValueError('Position of error is outside of the range of text lines; '
                     'error lineno: {}; lines: {}; message was: {}'.format(
                         span.limit.lineno, len(lines), error.message))
  line_count_each_side = error_context_line_count // 2
  low_lineno = max(span.start.lineno - line_count_each_side, 0)
  lines_before = lines[low_lineno:span.start.lineno]
  target_line = lines[span.start.lineno]
  # Note: since this is a limit there's a trailing +1.
  high_lineno = span.limit.lineno + line_count_each_side + 1
  lines_after = lines[span.start.lineno + 1:high_lineno]

  # Note: "color" is a tristate, None means no fixed request.
  use_color = color is True  # pylint: disable=g-bool-id-comparison
  use_color = use_color or (color is None and output.isatty())

  if use_color:
    fmt = termcolor.colored('{} {:04d}:', color='yellow') + ' {}'
    print_red = functools.partial(termcolor.cprint, color='red', file=output)
  else:
    fmt = '{} {:04d}: {}'
    print_red = functools.partial(print, file=output)

  def emit_line(lineno: int, line: Text, is_culprit: bool = False):
    # Note: humans generally think line i=0 is "line 1".
    print(fmt.format('*' if is_culprit else ' ', lineno + 1, line), file=output)

  leader = f'{error.filename}:{low_lineno+1}-{high_lineno}'
  print(
      termcolor.colored(leader, 'yellow') if use_color else leader, file=output)
  for i, line in enumerate(lines_before):
    emit_line(low_lineno + i, line)

  # Emit the culprit line.
  emit_line(span.start.lineno, target_line, is_culprit=True)
  # Emit error indicator, leading spaces correspond to leading line number.
  print_red('{:8s}'.format('') + '~' * span.start.colno + '^' + '-' *
            (max(0, span.limit.colno - span.start.colno - 2)) + '^ ' +
            str(error))

  # Emit the lines that come after.
  for i, line in enumerate(lines_after):
    emit_line(span.start.lineno + 1 + i, line)


def parse_text(text: Text,
               name: Text,
               print_on_error: bool,
               filename: Text,
               *,
               fs_open=None) -> ast.Module:
  """Returns a parsed module from DSL program text.

  Pretty-prints error information to stderr if an error is encountered, before
  re-raising.

  Args:
    text: The text to parse.
    name: Name that should be given to the resulting module.
    print_on_error: Whether to print to stderr when an error occurs -- if false,
      the error is simply raised and nothing is printed.
    filename: Filename that "text" orginates from.
    fs_open: Lets the user substitute their own filesystem open; e.g. if the
      program is not materialized on the real filesystem.

  Raises:
    ParseError: When a parsing error occurs.
    ScanError: When a scanning error occurs.
  """
  try:
    return parser.Parser(scanner.Scanner(filename, text)).parse_module(name)
  except (parser.ParseError, scanner.ScanError) as e:
    if print_on_error:
      pprint_positional_error(e, fs_open=fs_open)
    raise


def parse_path(path: Text, print_on_error: bool) -> ast.Module:
  """Returns a parsed module from a syntax file path.

  See "parse_text".

  Args:
    path: Filesystem path at which the syntax file resides.
    print_on_error: Whether to print to stderr when an error occurs -- if false,
      the error is simply raised and nothing is printed.

  Raises:
    ParseError: When a parsing error occurs.
    ScanError: When a scanning error occurs.
  """
  with open(path) as f:
    text = f.read()

  name = os.path.basename(path)
  name, _ = os.path.splitext(name)

  return parse_text(text, name, print_on_error=print_on_error, filename=path)
