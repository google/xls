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

# Lint as: python3
"""Performs an import, as in the 'import' keyword."""

import functools
import os
from typing import Dict

from absl import logging
from xls.common import runfiles
from xls.dslx import import_fn
from xls.dslx import parse_and_typecheck


def do_import(
    subject: import_fn.ImportTokens, cache: Dict[import_fn.ImportTokens,
                                                 import_fn.ModuleInfo]
) -> import_fn.ModuleInfo:
  """Imports the module identified (globally) by 'subject'.

  Resolves against an existing import in 'cache' if it is present.

  Args:
    subject: Tokens that globally uniquely identify the module to import; e.g.
      something built-in like ('std',) for the standard library or something
      fully qualified like ('xls', 'lib', 'math').
    cache: Cache that we resolve against so we don't waste resources
      re-importing things in the import DAG.

  Returns:
    The imported module information.
  """
  if subject in cache:
    return cache[subject]

  if subject in [('std',), ('float32',), ('bfloat16',)]:
    path = 'xls/dslx/stdlib/{}.x'.format(subject[0])
  else:
    path = os.path.join(*subject) + '.x'

  f_import = functools.partial(do_import, cache=cache)
  fully_qualified_name = '.'.join(subject)

  if os.path.exists(path):
    with open(path, mode='rb') as f:
      contents = f.read().decode('utf-8')
  else:
    contents = runfiles.get_contents_as_text(path)
    path = runfiles.get_path(path)

  logging.vlog(3, 'Parsing and typechecking %r: start', fully_qualified_name)
  m, node_to_type = parse_and_typecheck.parse_text(
      contents,
      fully_qualified_name,
      f_import=f_import,
      filename=path,
      print_on_error=True,
      do_typecheck=True)
  logging.vlog(3, 'Parsing and typechecking %r: done', fully_qualified_name)

  assert node_to_type is not None
  cache[subject] = (m, node_to_type)
  return m, node_to_type
