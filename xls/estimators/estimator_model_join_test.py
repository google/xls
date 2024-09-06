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
"""Tests for xls.estimators.estimator_model_join."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

ESTIMATOR_MODEL_JOIN_PATH = runfiles.get_path(
    'xls/estimators/estimator_model_join'
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

TEST_OPMODEL_ADD_AND_UMUL = """
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
op_models {
  op: "kUMul"
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
  total_area: 23.5
}
"""

TEST_DATAPOINT_ADD2 = """data_points {
  operation {
    op: "kAdd"
    bit_count: 32
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
    operands {
      bit_count: 32
    }
  }
  delay: 1205
  total_area: 34.5
}
"""

TEST_DATAPOINT_UMUL = """data_points {
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
  total_area: 167.8
}
"""

TEST_DELAY_MODEL_ADD = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel
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
  total_area: 23.5
}
metric: DELAY_METRIC
"""

TEST_DELAY_MODEL_ADD_REPLACE = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel
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
    operands {
      bit_count: 32
    }
  }
  delay: 1205
  total_area: 34.5
}
metric: DELAY_METRIC
"""

TEST_DELAY_MODEL_ADD_UMUL = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel
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
op_models {
  op: "kUMul"
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
  total_area: 23.5
}
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
  total_area: 167.8
}
metric: DELAY_METRIC
"""
TEST_AREA_MODEL_ADD = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel
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
  total_area: 23.5
}
metric: AREA_METRIC
"""

TEST_AREA_MODEL_ADD_REPLACE = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel
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
    operands {
      bit_count: 32
    }
  }
  delay: 1205
  total_area: 34.5
}
metric: AREA_METRIC
"""

TEST_AREA_MODEL_ADD_UMUL = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel
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
op_models {
  op: "kUMul"
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
  total_area: 23.5
}
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
  total_area: 167.8
}
metric: AREA_METRIC
"""


class DelayExtractSamplesFromDelayModelTest(test_base.TestCase):

  def test_basic(self):
    """Test tool with correct input no warning."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)

    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_file.full_path,
        '--data_points',
        dp_add_file.full_path,
        '--metric',
        'DELAY_METRIC',
    ]).decode('utf-8')
    self.assertEqual(stdout, TEST_DELAY_MODEL_ADD)
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_update_add_point(self):
    """Add a new data point via update_points."""
    out_file = self.create_tempfile()
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)
    dp_add2_file = self.create_tempfile(content=TEST_DATAPOINT_ADD2)

    subprocess.run(
        [
            ESTIMATOR_MODEL_JOIN_PATH,
            '--op_models',
            opm_add_file.full_path,
            '--data_points',
            dp_add_file.full_path,
            '--output',
            out_file.full_path,
            '--metric',
            'DELAY_METRIC',
        ],
        check=True,
    )
    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_file.full_path,
        '--data_points',
        dp_add2_file.full_path,
        '--output',
        out_file.full_path,
        '--update_mode=update_points',
        '--metric',
        'DELAY_METRIC',
    ]).decode('utf-8')
    # We need to move the "metric: DELAY_METRIC" line to the end of file
    self.assertEqual(
        stdout,
        TEST_DELAY_MODEL_ADD.replace('\nmetric: DELAY_METRIC', '')
        + TEST_DATAPOINT_ADD2
        + 'metric: DELAY_METRIC\n',
    )
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_update_add_op(self):
    """Add a new op point via --update_whole_ops."""
    out_file = self.create_tempfile()
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    opm_add_umul_file = self.create_tempfile(content=TEST_OPMODEL_ADD_AND_UMUL)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)
    dp_umul_file = self.create_tempfile(content=TEST_DATAPOINT_UMUL)

    subprocess.run(
        [
            ESTIMATOR_MODEL_JOIN_PATH,
            '--op_models',
            opm_add_file.full_path,
            '--data_points',
            dp_add_file.full_path,
            '--output',
            out_file.full_path,
            '--metric',
            'DELAY_METRIC',
        ],
        check=True,
    )
    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_umul_file.full_path,
        '--data_points',
        dp_umul_file.full_path,
        '--output',
        out_file.full_path,
        '--update_mode=update_whole_ops',
        '--metric',
        'DELAY_METRIC',
    ]).decode('utf-8')
    self.assertEqual(stdout, TEST_DELAY_MODEL_ADD_UMUL)
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_update_replace_op(self):
    """Replace a whole op's data points via update_whole_ops."""
    out_file = self.create_tempfile()
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)
    dp_add2_file = self.create_tempfile(content=TEST_DATAPOINT_ADD2)

    subprocess.run(
        [
            ESTIMATOR_MODEL_JOIN_PATH,
            '--op_models',
            opm_add_file.full_path,
            '--data_points',
            dp_add_file.full_path,
            '--output',
            out_file.full_path,
            '--metric',
            'DELAY_METRIC',
        ],
        check=True,
    )
    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_file.full_path,
        '--data_points',
        dp_add2_file.full_path,
        '--output',
        out_file.full_path,
        '--update_mode=update_whole_ops',
        '--metric',
        'DELAY_METRIC',
    ]).decode('utf-8')
    self.assertEqual(stdout, TEST_DELAY_MODEL_ADD_REPLACE)
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_mismatch(self):
    """Test tool with when no metric was specified."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_umul_file = self.create_tempfile(content=TEST_DATAPOINT_UMUL)

    try:
      sp = subprocess.run(
          [
              ESTIMATOR_MODEL_JOIN_PATH,
              '--op_models',
              opm_add_file.full_path,
              '--data_points',
              dp_umul_file.full_path,
              '--metric',
              'DELAY_METRIC',
          ],
          check=True,
          capture_output=True,
      )
      return_code = sp.returncode
      stderr = sp.stderr.decode('utf-8')
    except subprocess.CalledProcessError as e:
      return_code = e.returncode
      stderr = e.stderr.decode('utf-8')

    self.assertNotEqual(return_code, 0)
    self.assertIn('Issues found', stderr)
    self.assertIn('has a regression estimator but no data points', stderr)
    self.assertIn(
        "has data points but doesn't have a regression estimator", stderr
    )

  def test_no_metric(self):
    """Test tool with mismatched op_models and data_points."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)

    try:
      sp = subprocess.run(
          [
              ESTIMATOR_MODEL_JOIN_PATH,
              '--op_models',
              opm_add_file.full_path,
              '--data_points',
              dp_add_file.full_path,
          ],
          check=True,
          capture_output=True,
      )
      return_code = sp.returncode
      stderr = sp.stderr.decode('utf-8')
    except subprocess.CalledProcessError as e:
      return_code = e.returncode
      stderr = e.stderr.decode('utf-8')

    self.assertNotEqual(return_code, 0)
    self.assertIn(
        'Flags parsing error: flag --metric=None: Flag --metric must have a'
        ' value other than None.',
        stderr,
    )


class AreaExtractSamplesFromAreaModelTest(test_base.TestCase):

  def test_basic(self):
    """Test tool with correct input no warning."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)

    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_file.full_path,
        '--data_points',
        dp_add_file.full_path,
        '--metric',
        'AREA_METRIC',
    ]).decode('utf-8')
    self.assertEqual(stdout, TEST_AREA_MODEL_ADD)
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_update_add_point(self):
    """Add a new data point via update_points."""
    out_file = self.create_tempfile()
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)
    dp_add2_file = self.create_tempfile(content=TEST_DATAPOINT_ADD2)

    subprocess.run(
        [
            ESTIMATOR_MODEL_JOIN_PATH,
            '--op_models',
            opm_add_file.full_path,
            '--data_points',
            dp_add_file.full_path,
            '--output',
            out_file.full_path,
            '--metric',
            'AREA_METRIC',
        ],
        check=True,
    )
    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_file.full_path,
        '--data_points',
        dp_add2_file.full_path,
        '--output',
        out_file.full_path,
        '--update_mode=update_points',
        '--metric',
        'AREA_METRIC',
    ]).decode('utf-8')
    # We need to move the "metric: AREA_METRIC" line to the end of file
    self.assertEqual(
        stdout,
        TEST_AREA_MODEL_ADD.replace('\nmetric: AREA_METRIC', '')
        + TEST_DATAPOINT_ADD2
        + 'metric: AREA_METRIC\n',
    )
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_update_add_op(self):
    """Add a new op point via --update_whole_ops."""
    out_file = self.create_tempfile()
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    opm_add_umul_file = self.create_tempfile(content=TEST_OPMODEL_ADD_AND_UMUL)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)
    dp_umul_file = self.create_tempfile(content=TEST_DATAPOINT_UMUL)

    subprocess.run(
        [
            ESTIMATOR_MODEL_JOIN_PATH,
            '--op_models',
            opm_add_file.full_path,
            '--data_points',
            dp_add_file.full_path,
            '--output',
            out_file.full_path,
            '--metric',
            'AREA_METRIC',
        ],
        check=True,
    )
    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_umul_file.full_path,
        '--data_points',
        dp_umul_file.full_path,
        '--output',
        out_file.full_path,
        '--update_mode=update_whole_ops',
        '--metric',
        'AREA_METRIC',
    ]).decode('utf-8')
    self.assertEqual(stdout, TEST_AREA_MODEL_ADD_UMUL)
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_update_replace_op(self):
    """Replace a whole op's data points via update_whole_ops."""
    out_file = self.create_tempfile()
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_add_file = self.create_tempfile(content=TEST_DATAPOINT_ADD)
    dp_add2_file = self.create_tempfile(content=TEST_DATAPOINT_ADD2)

    subprocess.run(
        [
            ESTIMATOR_MODEL_JOIN_PATH,
            '--op_models',
            opm_add_file.full_path,
            '--data_points',
            dp_add_file.full_path,
            '--output',
            out_file.full_path,
            '--metric',
            'AREA_METRIC',
        ],
        check=True,
    )
    stdout = subprocess.check_output([
        ESTIMATOR_MODEL_JOIN_PATH,
        '--op_models',
        opm_add_file.full_path,
        '--data_points',
        dp_add2_file.full_path,
        '--output',
        out_file.full_path,
        '--update_mode=update_whole_ops',
        '--metric',
        'AREA_METRIC',
    ]).decode('utf-8')
    self.assertEqual(stdout, TEST_AREA_MODEL_ADD_REPLACE)
    self.assertNotIn('ISSUES FOUND', stdout)

  def test_mismatch(self):
    """Test tool with when no metric was specified."""
    opm_add_file = self.create_tempfile(content=TEST_OPMODEL_ADD)
    dp_umul_file = self.create_tempfile(content=TEST_DATAPOINT_UMUL)

    try:
      sp = subprocess.run(
          [
              ESTIMATOR_MODEL_JOIN_PATH,
              '--op_models',
              opm_add_file.full_path,
              '--data_points',
              dp_umul_file.full_path,
              '--metric',
              'AREA_METRIC',
          ],
          check=True,
          capture_output=True,
      )
      return_code = sp.returncode
      stderr = sp.stderr.decode('utf-8')
    except subprocess.CalledProcessError as e:
      return_code = e.returncode
      stderr = e.stderr.decode('utf-8')

    self.assertNotEqual(return_code, 0)
    self.assertIn('Issues found', stderr)
    self.assertIn('has a regression estimator but no data points', stderr)
    self.assertIn(
        "has data points but doesn't have a regression estimator", stderr
    )


if __name__ == '__main__':
  test_base.main()
