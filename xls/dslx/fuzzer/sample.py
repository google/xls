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

# Lint as: python3
"""Library describing code samples generated and run by the fuzzer."""

import json
import re

from typing import Text, Optional, NamedTuple, Sequence

from xls.dslx.interpreter import value_parser
from xls.dslx.interpreter.value import Value

Args = Sequence[Value]
ArgsBatch = Sequence[Args]
ArgsBatchIn = Sequence[Sequence[Value]]


def parse_args(args_text: Text) -> Args:
  """Parses a semicolon-delimited list of values.

  Example input:
    bits[32]:6; (bits[8]:2, bits[16]:4)

  Returns bits values are always unsigned.

  Args:
    args_text: Text to parse.

  Returns:
    List of parsed Values,
  """
  return tuple(
      value_parser.value_from_string(a)
      for a in args_text.split(';')
      if a.strip())


def parse_args_batch(args_text: Text) -> ArgsBatch:
  """Parses a batch of arguments, one argument set per line."""
  return tuple(parse_args(l) for l in args_text.splitlines() if l.strip())


def args_batch_to_text(args_batch: ArgsBatch) -> Text:
  """Returns a string representation of argument batch."""
  lines = []
  for args in args_batch:
    lines.append('; '.join(str(a) for a in args))
  return '\n'.join(lines)


class SampleOptions(NamedTuple):
  """Options describing how to run a code sample.

  Attributes:
    input_is_dslx: Whether the code sample is DSLX. Otherwise assumed to be XLS
      IR.
    convert_to_ir: Convert the input code sample to XLS IR. Only meaningful if
      input_is_dslx is True.
    optimize_ir: Optimize the XLS IR.
    use_jit: Use LLVM jit when evaluating the XLS IR.
    codegen: Generate Verilog from the optimized IR. Requires optimize_ir to be
      True.
    codegen_args: List of arguments to pass to codegen_main. Requires codegen to
      be True.
    simulate: Run the Verilog simulator on the generated Verilog. Requires
      codegen to be True.
    simulator: The Verilog simulator to use, e.g. "iverilog".
  """
  input_is_dslx: bool = True
  convert_to_ir: bool = True
  optimize_ir: bool = True
  use_jit: bool = True
  codegen: bool = False
  codegen_args: Optional[Sequence[Text]] = None
  simulate: bool = False
  simulator: Optional[Text] = None

  def to_json(self):
    """Returns a JSON-encoded string describing this object."""
    return json.dumps(self._asdict())

  @classmethod
  def from_json(cls, json_text: Text):
    """Parses a JSON-encoded string and returns a SampleOptions object."""
    options = SampleOptions(**json.loads(json_text))
    # JSON parsing produces lists rather than tuples for JSON arrays. Convert
    # these elements into tuples for immutability
    for k, v in options._asdict().items():
      if isinstance(v, list):
        options = options._replace(**{k: tuple(v)})
    return options


class Sample(NamedTuple):
  """Abstraction describing a fuzzer code sample and how to run it.

  Attributes:
    input_text: The code sample as text.
    options: The SampleOptions object describing how to run the sample.
    args_batch: The (optional) argument values to use for interpretation and
      simulation.
  """
  input_text: Text
  options: SampleOptions
  args_batch: Optional[ArgsBatch] = None

  def to_crasher(self) -> Text:
    """Returns a "crasher" text encapsulating the sample.

    A crasher is a text serialialization of the sample which enables easy
    reproduction from a single text file. Crashers may be checked in as tests in
    third_party/xls/dslx/fuzzer/crashers.

    A crasher has the following format:
      // options: <JSON-serialized SampleOptions>
      // args: <argument set 0>
      // ...
      // args: <argument set 1>
      <code sample>
    """
    lines = ['// options: ' + self.options.to_json()]
    if self.args_batch is not None:
      for args in self.args_batch:  # pylint: disable=not-an-iterable
        lines.append('// args: ' + '; '.join(str(a) for a in args))
    return '\n'.join(lines) + '\n' + self.input_text + '\n'

  @classmethod
  def from_crasher(cls, s: Text) -> 'Sample':
    """Parses a crasher and returns a Sample object."""
    options = None
    args_batch = []
    input_lines = []
    for line in s.splitlines():
      m = re.match(r'\s*//\s*options:(.*)', line)
      if m:
        assert options is None
        options = SampleOptions.from_json(m.group(1))
        continue
      m = re.match(r'\s*//\s*args:(.*)', line)
      if m:
        args_batch.append(parse_args(m.group(1)))
        continue
      input_lines.append(line)
      input_text = '\n'.join(input_lines)

    assert options is not None
    return Sample(input_text, options, tuple(args_batch))
