# Copyright 2021 The XLS Authors
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

"""Helper tool for verifying checksums for (user) build targets."""

import os
import subprocess
from typing import Sequence

from absl import app

from xls.common import runfiles


def main(argv: Sequence[str]) -> None:
  if len(argv) != 3:
    raise app.UsageError(
        'Invalid command-line arguments; want %s <path> <expected-sha256>'
    )
  _, path, want = argv
  if not os.path.exists(path):
    path = runfiles.get_path(path)
  output = subprocess.check_output(['sha256sum', path], encoding='utf-8')
  got = output.split(' ', 1)[0]
  if got == want:
    print('OK: {!r} == {!r}'.format(got, want))
  else:
    raise ValueError(
        'Unexpected sha256 for {}; want {}; got: {}'.format(path, want, got)
    )


if __name__ == '__main__':
  app.run(main)
