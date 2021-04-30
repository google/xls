# Lint as: python3
#
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
"""Tests for xls.synthesis.timing_characterization_client_main."""

import collections
import tempfile

from absl.testing import absltest
from xls.delay_model import delay_model_pb2
from xls.synthesis import timing_characterization_client as client


class TimingCharacterizationClientMainTest(absltest.TestCase):

  def test_save_load_checkpoint(self):
    model = delay_model_pb2.DelayModel()
    # Maps an op name to the set of bit configurations we've run that op with.
    data_points = collections.defaultdict(set)

    # Set up some dummy data.
    ops = ["op_a", "op_b", "op_c", "op_d", "op_e"]
    bit_configs = ["3, 1, 1", "4, 1, 2", "5, 2, 1", "6, 2, 2"]
    for op in ops:
      data_points[op] = set()
      for bit_config in bit_configs:
        data_points[op].add(bit_config)
        result = delay_model_pb2.DataPoint()
        result.operation.op = op
        for elem in bit_config.split(",")[1:]:
          operand = delay_model_pb2.Operation.Operand()
          operand.bit_count = int(elem)
          result.operation.operands.append(operand)
        result.operation.bit_count = int(bit_config[0])
        result.delay = 5
        model.data_points.append(result)
    tf = tempfile.NamedTemporaryFile()
    client.save_checkpoint(model, tf.name)
    loaded_data_points, loaded_model = client.init_data(tf.name)

    self.assertEqual(model, loaded_model)
    # Fancy equality checking so we get clearer error messages on
    # mismatch.
    for op in ops:
      self.assertIn(op, loaded_data_points)
      loaded_op = loaded_data_points[op]
      for bit_config in bit_configs:
        self.assertIn(bit_config, loaded_op)
        self.assertIn(bit_config, data_points[op])
    self.assertEqual(data_points, loaded_data_points)


if __name__ == "__main__":
  absltest.main()
