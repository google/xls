# Copyright 2022 The XLS Authors
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
"""Test that checks mkdocs nav and markdown files correspond."""

import os
import sys
from typing import Sequence

from absl import app
import yaml

from xls.common import runfiles


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with open(runfiles.get_path('mkdocs.yml')) as f:
    config = yaml.load(f, Loader=yaml.Loader)

  nav = config['nav']

  nav_files = []

  def extract_files(node):
    if isinstance(node, str):
      nav_files.append(node)
    elif isinstance(node, list):
      for e in node:
        extract_files(e)
    elif isinstance(node, dict):
      for _, v in node.items():
        extract_files(v)
    else:
      raise NotImplementedError(f'Unhandled YAML construct: {node!r}')

  extract_files(nav)

  existing_files = []
  top_dirpath = None
  for dirpath, _, filenames in runfiles.walk_resources('docs_src/'):
    if top_dirpath is None:
      top_dirpath = dirpath

    existing_files += [
        os.path.relpath(os.path.join(dirpath, f), top_dirpath)
        for f in filenames
        if f.endswith('.md')
    ]

  # These are handled specially in the import/export flow, so it is ok if
  # they look they do not exist before export.
  special_files = {'contributing.md', 'README.md'}

  no_nav = set(existing_files) - set(nav_files)
  exit_code = 0
  if no_nav:
    print(
        'the following files have no corresponding navigation:',
        no_nav,
        file=sys.stderr,
    )
    exit_code = 1

  # Files that have a corresponding navigation, but do not exist (DNE) in the
  # filesystem.
  nav_but_dne = set(nav_files) - set(existing_files) - special_files
  if nav_but_dne:
    print(
        'the following files have navigation but do not exist:',
        nav_but_dne,
        file=sys.stderr,
    )
    exit_code = 1

  sys.exit(exit_code)


if __name__ == '__main__':
  app.run(main)
