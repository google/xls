# Lint as: python3
#
# Copyright 2020 The XLS Authors
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
"""Library for accessing runfiles."""

import os

from rules_python.python.runfiles import runfiles

_BASE_PATH = 'com_google_xls'


def get_path(relpath: str) -> str:
  path = os.path.join(_BASE_PATH, relpath)
  r = runfiles.Create()
  runfile_path = r.Rlocation(path)
  if not os.path.exists(runfile_path):
    raise FileNotFoundError(f'Cannot find runfile {relpath}')
  return runfile_path


def get_contents_as_text(relpath: str) -> str:
  with open(get_path(relpath), 'r', encoding='utf-8') as f:
    return f.read()
