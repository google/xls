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

"""Tests for xls.visualization.ir_viz.python.ir_to_json."""

import json
import sys

from xls.common.python import init_xls
from xls.visualization.ir_viz.python import ir_to_json
from absl.testing import absltest


def setUpModule():
  # This is required so that module initializers are called including those
  # which register delay models.
  init_xls.init_xls(sys.argv)


class IrToJsonTest(absltest.TestCase):

  def test_ir_to_json(self):
    json_str = ir_to_json.ir_to_json(
        """package test_package

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}""", 'unit')
    json_dict = json.loads(json_str)
    self.assertEqual(json_dict['name'], 'test_package')

    function_dict = json_dict['function_bases'][0]
    self.assertIn('edges', function_dict)
    self.assertLen(function_dict['edges'], 2)
    self.assertIn('nodes', function_dict)
    self.assertLen(function_dict['nodes'], 3)

  def test_ir_to_json_with_scheduling(self):
    json_str = ir_to_json.ir_to_json(
        """package test

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  ret neg.2: bits[32] = neg(add.1)
}""", 'unit', 2)
    json_dict = json.loads(json_str)
    function_dict = json_dict['function_bases'][0]

    self.assertIn('edges', function_dict)
    self.assertLen(function_dict['edges'], 3)
    self.assertIn('nodes', function_dict)
    self.assertLen(function_dict['nodes'], 4)
    for node in function_dict['nodes']:
      if node['id'] == 'x' or node['id'] == 'y':
        self.assertEqual(node['attributes']['cycle'], 0)
      elif node['id'] == 'add_1':
        self.assertEqual(node['attributes']['cycle'], 0)
      elif node['id'] == 'neg_2':
        self.assertEqual(node['attributes']['cycle'], 1)


if __name__ == '__main__':
  absltest.main()
