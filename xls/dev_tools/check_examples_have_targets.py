# Copyright 2024 The XLS Authors.
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

"""Checks there are test targets for all .x files in the examples directory."""

import collections
from collections.abc import Sequence
import glob
import os
import pprint
import subprocess
import sys
from typing import Dict, Set
import xml.etree.ElementTree as etree


def get_x_files_for_rule(
    rule: etree.Element, library_target_to_x_files: Dict[str, Set[str]]
) -> Set[str]:
  """Returns the set of .x files that seem to be used by a given rule."""
  x_files = set()

  srcs = rule.find('.//list[@name="srcs"]')
  if srcs is not None:
    label_list = srcs.find('label')
    assert label_list is not None
    for label in label_list:
      assert hasattr(label, 'value'), label
      assert isinstance(label.value, str), label.value
      if label.value.endswith('.x'):
        x_files.add(label.value)
      elif label.value in library_target_to_x_files:
        x_files |= library_target_to_x_files[label.value]

  library = rule.find('.//label[@name="library"]')
  if library is not None:
    x_files |= library_target_to_x_files[library.attrib['value']]

  for rule_input in rule.findall('rule-input'):
    name = rule_input.attrib['name']
    assert isinstance(name, str), name
    if name.endswith('.x'):
      x_files.add(name)
    elif name in library_target_to_x_files:
      x_files |= library_target_to_x_files[name]

  return x_files


def collect_x_files_for_library(
    rule: etree.Element, library_target_to_x_files: Dict[str, Set[str]]
) -> None:
  """Collects .x files used in a given rule into library_target_to_x_files."""
  rule_name = rule.attrib['name']
  x_files: Set[str] = get_x_files_for_rule(rule, library_target_to_x_files)
  library_target_to_x_files[rule_name] = x_files


def get_x_to_test_targets():
  """Returns a mapping from x files to test targets using it."""
  xml = subprocess.check_output(
      ['bazel', 'query', '--output=xml', '//xls/examples:all']
  )
  e = etree.fromstring(xml)

  x_to_targets = collections.defaultdict(list)

  library_target_to_x_files = {}

  # Find all the rules that contain .x files in the sources list.
  for rule in e.findall('rule'):
    if rule.attrib['class'] == 'xls_dslx_library':
      collect_x_files_for_library(rule, library_target_to_x_files)
    elif rule.attrib['class'] == 'xls_dslx_test':
      rule_name = rule.attrib['name']
      for x_file in get_x_files_for_rule(rule, library_target_to_x_files):
        x_to_targets[x_file].append(rule_name)

  return x_to_targets


def _to_target(relpath: str) -> str:
  dirname, filename = os.path.split(relpath)
  return f'//{dirname}:{filename}'


def _does_file_seem_to_contain_tests(path: str) -> bool:
  with open(path) as f:
    contents = f.read()
  return '#[test' in contents or '#[test_proc' in contents


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    print('Too many command-line arguments.', file=sys.stderr)
    sys.exit(-1)

  x_to_targets = get_x_to_test_targets()
  pprint.pprint(x_to_targets)

  x_files = glob.glob('xls/examples/*.x')
  pprint.pprint(x_files)

  missing_targets = False
  for x_file in x_files:
    if not _does_file_seem_to_contain_tests(x_file):
      continue

    x_file_target = _to_target(x_file)
    if x_file_target not in x_to_targets:
      print(
          f'ERROR: could not find target for .x file: {x_file}', file=sys.stderr
      )
      missing_targets = True

  if not missing_targets:
    print(
        'OK: no .x files with tests seem to be missing associated test targets'
    )

  sys.exit(1 if missing_targets else 0)


# -- pytest


def test_get_direct_x_files_for_test_rule():
  contents = """
  <rule class="xls_dslx_test" location="/path/to/xls/examples/BUILD:67:14" name="//xls/examples:adler32_dslx_test">
      <string name="name" value="adler32_dslx_test"/>
      <list name="srcs">
          <label value="//xls/examples:adler32.x"/>
      </list>
      <rule-input name="//xls/examples:adler32.x"/>
  </rule>
  """
  e = etree.fromstring(contents)
  assert get_x_files_for_rule(e, {}) == {'//xls/examples:adler32.x'}


def test_collect_x_files_for_library():
  """Tests the above routine with some sample XML data from Bazel query."""
  contents = """
   <rule class="xls_dslx_library" location="/path/to/xls/examples/BUILD:521:17" name="//xls/examples:find_index_dslx">
        <string name="name" value="find_index_dslx"/>
        <list name="srcs">
            <label value="//xls/examples:find_index.x"/>
        </list>
        <rule-input name="//:license"/>
        <rule-input name="//xls/dslx:dslx_fmt"/>
        <rule-input name="//xls/dslx:interpreter_main"/>
        <rule-input name="//xls/dslx/ir_convert:ir_converter_main"/>
        <rule-input name="//xls/examples:find_index.x"/>
        <rule-input name="//xls/jit:aot_compiler"/>
        <rule-input name="//xls/jit:jit_wrapper_generator_main"/>
        <rule-input name="//xls/tools:benchmark_codegen_main"/>
        <rule-input name="//xls/tools:benchmark_main"/>
        <rule-input name="//xls/tools:check_ir_equivalence_main"/>
        <rule-input name="//xls/tools:codegen_main"/>
        <rule-input name="//xls/tools:eval_ir_main"/>
        <rule-input name="//xls/tools:opt_main"/>
    </rule>
  """
  e = etree.fromstring(contents)
  library_target_to_x_files = {}
  collect_x_files_for_library(e, library_target_to_x_files)
  assert library_target_to_x_files == {
      '//xls/examples:find_index_dslx': {'//xls/examples:find_index.x'}
  }


if __name__ == '__main__':
  main(sys.argv)
