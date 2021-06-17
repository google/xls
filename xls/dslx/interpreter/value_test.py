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

"""Tests for xls.dslx.interpreter.value."""

import pickle

from absl.testing import absltest
from xls.dslx.python import interp_value


class ValueTest(absltest.TestCase):

  def test_pickle(self):
    py_value = 1 << 65
    v = interp_value.interp_value_from_ir_string(f'bits[66]:{py_value:#b}')
    dumped = pickle.dumps(v, 2)
    print(dumped)
    v_prime = pickle.loads(dumped)
    self.assertEqual(v, v_prime)


if __name__ == '__main__':
  absltest.main()
