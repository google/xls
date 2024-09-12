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

from google.protobuf import text_format
from xls.common import test_base
from xls.estimators import estimator_model
from xls.estimators import estimator_model_pb2
from xls.estimators.area_model import area_model_utils


TEST_UNIT_AREA_MODEL = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel

op_models { op: "kIdentity" estimator { fixed: 1 } }

metric: AREA_METRIC
"""

TEST_AREA_MODEL_WITH_KIDENTITY_DATA_POINTS = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel

op_models {
  op: "kIdentity"
  estimator {
    area_regression {
    }
  }
}

data_points {
  operation {
    op: "kIdentity"
    bit_count: 1
    operands {
      bit_count: 1
    }
  }
  delay: 241
  delay_offset: 241
  total_area: 40.0384
  sequential_area: 40.0384
}

metric: AREA_METRIC
"""

TEST_NONUNIT_AREA_MODEL_WITHOUT_KIDENTITY_DATA_POINTS = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel

op_models { op: "kAdd" estimator { fixed: 1 } }

metric: AREA_METRIC
"""

TEST_DELAY_MODEL = """# proto-file: xls/estimators/estimator_model.proto
# proto-message: xls.estimator_model.EstimatorModel

data_points {
  operation {
    op: "kNot"
    bit_count: 4
    operands {
      bit_count: 4
    }
  }
  delay: 256
  delay_offset: 241
}

metric: DELAY_METRIC
"""


class AreaModelUtilsTest(test_base.TestCase):

  def test_unit_model_register_area(self):
    # em = estimator_model.EstimatorModel.from_textproto(TEST_UNIT_AREA_MODEL)
    em = estimator_model.EstimatorModel(
        text_format.Parse(
            TEST_UNIT_AREA_MODEL, estimator_model_pb2.EstimatorModel()
        )
    )
    self.assertEqual(area_model_utils.get_one_bit_register_area(em), 1)

  def test_nonunit_model_without_kidentity_data_points_register_area(self):
    em = estimator_model.EstimatorModel(
        text_format.Parse(
            TEST_NONUNIT_AREA_MODEL_WITHOUT_KIDENTITY_DATA_POINTS,
            estimator_model_pb2.EstimatorModel(),
        )
    )
    with self.assertRaises(ValueError):
      area_model_utils.get_one_bit_register_area(em)

  def test_nonunit_model_with_kidentity_data_points_register_area(self):
    em = estimator_model.EstimatorModel(
        text_format.Parse(
            TEST_AREA_MODEL_WITH_KIDENTITY_DATA_POINTS,
            estimator_model_pb2.EstimatorModel(),
        )
    )
    self.assertEqual(
        area_model_utils.get_one_bit_register_area(em), 40.0384 / 2
    )

  def test_delay_model_register_area(self):
    em = estimator_model.EstimatorModel(
        text_format.Parse(
            TEST_DELAY_MODEL,
            estimator_model_pb2.EstimatorModel(),
        )
    )
    with self.assertRaises(ValueError):
      area_model_utils.get_one_bit_register_area(em)


if __name__ == "__main__":
  test_base.main()
