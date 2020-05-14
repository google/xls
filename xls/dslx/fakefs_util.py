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
"""Helpers for working with fake filesystems.

Helper for making a scoped fake filesystem; e.g. for use in tests or synthetic
environments like the fuzzer.
"""

import contextlib
from typing import Text

from pyfakefs import fake_filesystem_unittest as ffu


@contextlib.contextmanager
def scoped_fakefs(path: Text, contents: Text):
  with ffu.Patcher() as patcher:
    create_file = getattr(patcher.fs, 'create_file', None) or getattr(
        patcher.fs, 'CreateFile')
    create_file(path, contents=contents)
    yield
