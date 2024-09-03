#
# Copyright 2024 The XLS Authors
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
"""Tests for extract_sample_points_from_ir_main."""

import subprocess

from google.protobuf import text_format
from xls.common import gfile
from xls.common import runfiles
from xls.common import test_base
from xls.estimators.delay_model import delay_model_pb2

BINARY_PATH = runfiles.get_path(
    'xls/dev_tools/extract_sample_points_from_ir_main'
)

TEST_IR = """
package foo

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=3, id=1)
  ret sub.2: bits[32] = sub(literal.1, literal.1, id=2)
}

fn g() -> bits[32] {
  literal.3: bits[32] = literal(value=4, id=3)
  sub.4: bits[32] = sub(literal.3, literal.3, id=4)
  literal.5: bits[4] = literal(value=5, id=5)
  literal.6: bits[4] = literal(value=1, id=6)
  sub.7: bits[4] = sub(literal.5, literal.6, id=7)
  ret shra.8: bits[32] = shra(sub.4, sub.7, id=8)
}
"""

TEST_OP_MODELS = """
op_models {
  op: "kSub"
  estimator { regression {} }
}
op_models {
  op: "kShll"
  estimator { regression {} }
}
op_models {
  op: "kShra"
  estimator { alias_op: "kShll" }
}
op_models {
  op: "kAnd"
  estimator { logical_effort {} }
}
op_models {
  op: "kSignExt"
  estimator { regression {} }
}
"""


class ExtractSamplePointsFromIrMainTest(test_base.TestCase):

  def test_basic(self):
    """Test tool without specifying --schedule_path."""
    ir_file = self.create_tempfile(content=TEST_IR)
    op_models_file = self.create_tempfile(content=TEST_OP_MODELS)
    out_file = self.create_tempfile()
    output = subprocess.check_output(
        [
            BINARY_PATH,
            '--ir_path={}'.format(ir_file.full_path),
            '--op_models_path={}'.format(op_models_file.full_path),
            '--out_path={}'.format(out_file.full_path),
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertEqual(
        output,
        """Operation                 Params                                      Frequency
kSub                      32, 32 -> 32                                        2
kShll                     32, 4 -> 32                                         1
kSub                      4, 4 -> 4                                           1

""",
    )
    with gfile.open(out_file.full_path, 'r') as f:
      output = text_format.ParseLines(f, delay_model_pb2.OpSamplesList())

    self.assertLen(output.op_samples, 3)
    self.assertEqual(output.op_samples[0].op, 'kIdentity')
    self.assertLen(output.op_samples[0].samples, 1)
    self.assertEqual(output.op_samples[0].samples[0].result_width, 1)
    self.assertEqual(output.op_samples[0].samples[0].operand_widths, [1])
    self.assertEqual(output.op_samples[1].op, 'kShll')
    self.assertLen(output.op_samples[1].samples, 1)
    self.assertEqual(output.op_samples[1].samples[0].result_width, 32)
    self.assertEqual(output.op_samples[1].samples[0].operand_widths, [32, 4])
    self.assertEqual(output.op_samples[2].op, 'kSub')
    self.assertLen(output.op_samples[2].samples, 2)
    self.assertEqual(output.op_samples[2].samples[0].result_width, 4)
    self.assertEqual(output.op_samples[2].samples[0].operand_widths, [4, 4])
    self.assertEqual(output.op_samples[2].samples[1].result_width, 32)
    self.assertEqual(output.op_samples[2].samples[1].operand_widths, [32, 32])

  def test_with_delay_model(self):
    """Test tool without specifying --schedule_path."""
    ir_file = self.create_tempfile(content=TEST_IR)
    op_models_file = self.create_tempfile(content=TEST_OP_MODELS)
    out_file = self.create_tempfile()
    output = subprocess.check_output(
        [
            BINARY_PATH,
            '--ir_path={}'.format(ir_file.full_path),
            '--op_models_path={}'.format(op_models_file.full_path),
            '--out_path={}'.format(out_file.full_path),
            '--delay_model=asap7',
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')

    # Make sure there are delay values, but don't lock down what they are.
    self.assertRegex(
        output,
        r"""Operation                 Params                                      Frequency
kSub                      32, 32 -> 32                                        2
kShll                     32, 4 -> 32                                         1
kSub                      4, 4 -> 4                                           1

Operation                 Params                                          Delay
kSub                      32, 32 -> 32.*\d+
kShll                     32, 4 -> 32.*\d+
kSub                      4, 4 -> 4.*\d+

""",
    )


if __name__ == '__main__':
  test_base.main()
