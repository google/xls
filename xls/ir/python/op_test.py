#
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
"""Tests for xls.ir.python.type."""

from xls.common import test_base
from xls.ir.python import op


class OpTest(test_base.TestCase):

  def test_op_to_string(self):
    # Verify all_ops returns something non trivial.
    self.assertGreater(len(op.all_ops()), 1)
    # Tests a few arbitrary ops.
    self.assertEqual(op.op_to_string(op.Op.EQ), 'eq')
    self.assertEqual(op.op_to_string(op.Op.ADD), 'add')
    self.assertEqual(op.op_to_string(op.Op.PRIORITY_SEL), 'priority_sel')


if __name__ == '__main__':
  test_base.main()
