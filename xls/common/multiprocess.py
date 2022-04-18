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

"""Multi-process adapter API.

Resolves differences between in-house multiprocessing infrastructure and
the standard multiprocessing module.
"""

import multiprocessing as mp
from typing import Any

from absl import app


def has_user_data_support() -> bool:
  """See comment on get_user_data()."""
  return False


# TODO(leary): 2020-06-20 It would be better to try to unify behind a single API
# instead of having two usage modes that live simultaneously. We should attempt
# to adapt multiprocessing to conform to this API style.
def get_user_data() -> Any:
  """This is not used in open source.

  Instead, values are flowed around in standard multiprocessing style rather
  than keeping a per-process state accessed in a global way.
  """
  return None


def Process(target, args) -> mp.Process:  # pylint: disable=invalid-name
  return mp.Process(target=target, args=args)


def run_main(main, user_data):  # pylint: disable=unused-argument
  app.run(main)
