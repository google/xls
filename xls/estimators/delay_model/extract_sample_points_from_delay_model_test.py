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

EXTRACT_SAMPLES_PATH = runfiles.get_path(
    'xls/estimators/delay_model/extract_sample_points_from_delay_model'
)

TEST_DELAY_MODEL = """
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
op_models { op: "kSub" estimator { alias_op: "kAdd" } }
data_points {
  operation {
    op: "kAdd"
    bit_count: 256
    operands { bit_count: 256 }
    operands { bit_count: 256 }
  }
  delay: 4242
  delay_offset: 225
}
data_points {
  operation {
    op: "kAdd"
    bit_count: 128
    operands { bit_count: 128 }
    operands { bit_count: 128 }
  }
  delay: 4200
  delay_offset: 225
}
"""


class DelayExtractSamplesFromDelayModelTest(test_base.TestCase):

  def test_basic(self):
    """Test tool with simple input no error."""
    dm_file = self.create_tempfile(content=TEST_DELAY_MODEL)

    stdout = subprocess.check_output([
        EXTRACT_SAMPLES_PATH,
        '--input',
        dm_file.full_path,
    ]).decode('utf-8')
    self.assertEqual(
        stdout,
        """# proto-file: xls/estimators/delay_model/delay_model.proto
# proto-message: xls.delay_model.OpSamples
op_samples {
  op: "kAdd"
  samples {
    result_width: 256
    operand_widths: 256
    operand_widths: 256
  }
  samples {
    result_width: 128
    operand_widths: 128
    operand_widths: 128
  }
}

""",
    )


if __name__ == '__main__':
  test_base.main()
