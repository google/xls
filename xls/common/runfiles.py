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
from typing import Iterable

from python.runfiles import runfiles

_BASE_PATH = 'com_google_xls'


def get_path(relpath: str, *, repository: str = _BASE_PATH) -> str:
  path = os.path.join(repository, relpath)
  r = runfiles.Create()
  runfile_path = r.Rlocation(path)
  if not os.path.exists(runfile_path):
    raise FileNotFoundError(
        f'Cannot find runfile {relpath} (repository: {repository})'
    )
  return runfile_path


def get_contents_as_text(relpath: str) -> str:
  with open(get_path(relpath), 'r', encoding='utf-8') as f:
    return f.read()


def get_contents_as_bytes(relpath: str) -> bytes:
  with open(get_path(relpath), 'rb', encoding=None) as f:
    return f.read()


def walk_resources(relpath: str) -> Iterable[str]:
  return os.walk(get_path(relpath))
