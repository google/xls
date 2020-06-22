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

"""Testing adapter layer.

This is an adapter that helps paper over differences between Google-internal and
OSS/external testing facilities.
"""

import sys
import tempfile

from xls.common.python import init_xls
from absl.testing import absltest

TestCase = absltest.TestCase
TempFileCleanup = absltest.TempFileCleanup
skipIf = absltest.unittest.skipIf


def create_named_output_text_file(name: str) -> str:
  return tempfile.NamedTemporaryFile(suffix=name, delete=False).name


def main() -> None:
  init_xls.init_xls(sys.argv)
  absltest.main()
