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

import multiprocessing as mp
import tempfile

from absl.testing import absltest
from xls.estimators import estimator_model_pb2
from xls.estimators import estimator_model_utils
from xls.synthesis import timing_characterization_client as client


class TimingCharacterizationClientMainTest(absltest.TestCase):

  def test_save_load_checkpoint(self):

    results = estimator_model_pb2.DataPoints()
    # Maps an op name to the set of bit configurations we've run that op with.

    # Set up some fake data.
    ops = ["op_a", "op_b", "op_c", "op_d", "op_e"]
    bit_configs = ["3, 1, 1", "4, 1, 2", "5, 2, 1", "6, 2, 2"]
    tf = tempfile.NamedTemporaryFile()
    lock = mp.Lock()
    for op in ops:
      for bit_config in bit_configs:
        result = estimator_model_pb2.DataPoint()
        result.operation.op = op
        for elem in bit_config.split(",")[1:]:
          operand = estimator_model_pb2.Operation.Operand()
          operand.bit_count = int(elem)
          result.operation.operands.append(operand)
        result.operation.bit_count = int(bit_config[0])
        result.delay = 5
        results.data_points.append(result)
        client.save_checkpoint(result, tf.name, lock)

    saved_results_dict = estimator_model_utils.map_data_points_by_key(
        results.data_points
    )
    loaded_results = client.load_checkpoints(tf.name)
    self.assertEqual(results, loaded_results)
    loaded_results_dict = estimator_model_utils.map_data_points_by_key(
        loaded_results.data_points
    )
    # Fancy equality checking so we get clearer error messages on
    # mismatch.
    for point in results.data_points:
      request_key = estimator_model_utils.get_data_point_key(point)
      self.assertIn(request_key, loaded_results_dict)
      self.assertIn(request_key, saved_results_dict)
    self.assertEqual(len(results.data_points), len(loaded_results_dict))
    self.assertEqual(saved_results_dict, loaded_results_dict)


if __name__ == "__main__":
  absltest.main()
