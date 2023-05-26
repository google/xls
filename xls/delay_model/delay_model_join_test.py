#
# Copyright 2023 The XLS Authors
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
"""Tests for xls.tools.extract_samples_from_delay_model."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

DELAY_MODEL_JOIN_PATH = (
    runfiles.get_path('xls/delay_model/delay_model_join')
)

TEST_OPMODEL_ADD = """
op_models {
  op: "kAdd"
  estimator {
    regression {
      expressions {
        factor {
          source: OPERAND_BIT_COUNT
          operand_number: 0
        }
      }
    }
  }
}
"""

TEST_DATAPOINT_ADD = """
data_points {
  operation {
    op: "kAdd"
    bit_count: 32
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
  }
  delay: 1196
}
"""

TEST_DATAPOINT_UMUL = """
data_points {
  operation {
    op: "kUMul"
    bit_count: 32
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
  }
  delay: 3910
}
"""

TEST_DELAY_MODEL_ADD = """# proto-file: xls/delay_model/delay_model.proto
# proto-message: xls.delay_model.DelayModel
op_models {
  op: "kAdd"
  estimator {
    regression {
      expressions {
        factor {
          source: OPERAND_BIT_COUNT
        }
      }
    }
  }
}
data_points {
  operation {
    op: "kAdd"
    bit_count: 32
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
  }
  delay: 1196
}
"""


class DelayExtractSamplesFromDelayModelTest(test_base.TestCase):

  def test_basic(self):
    """Test tool with correct input no warning."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)

    stdout = subprocess.check_output(
        [DELAY_MODEL_JOIN_PATH,
         '--op_models', opm_add_file.full_path,
         '--data_points', dp_add_file.full_path,
         ]).decode('utf-8')
    self.assertEqual(
        stdout, TEST_DELAY_MODEL_ADD)
    self.assertNotIn(
        'ISSUES FOUND', stdout)

  def test_mismatch(self):
    """Test tool with mismatched op_models and data_points."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_umul_file = self.create_tempfile(content=TEST_DATAPOINT_UMUL)

    try:
      subprocess.run(
          [DELAY_MODEL_JOIN_PATH,
           '--op_models', opm_add_file.full_path,
           '--data_points', dp_umul_file.full_path,
           ],
          check=True,
          capture_output=True,
      )
    except subprocess.CalledProcessError as e:
      return_code = e.returncode
      stderr = e.stderr.decode('utf-8')

    self.assertNotEqual(return_code, 0)
    self.assertIn('ISSUES FOUND', stderr)
    self.assertIn(
        'has a regression estimator but no data points', stderr)
    self.assertIn(
        'has data points but doesn\'t have a regression estimator', stderr)


if __name__ == '__main__':
  test_base.main()
