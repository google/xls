# Copyright 2026 The XLS Authors
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

"""Compiles an EstimatorModel textproto into a CompiledEstimatorModel textproto."""

from collections.abc import Sequence

from absl import app
from absl import flags
from google.protobuf import text_format

from xls.estimators import estimator_model
from xls.estimators import estimator_model_pb2
from xls.estimators.area_model import area_model_utils

_REFERENCE_GOLDEN = flags.DEFINE_string(
    'reference_golden',
    None,
    'Optional path to an existing CompiledEstimatorModel textproto. If the'
    ' newly compiled model matches within floating-point solver tolerance, the'
    ' reference golden is emitted unchanged to avoid cross-platform solver'
    ' noise.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) != 2:
    raise app.UsageError('Expected exactly one input textproto argument.')

  with open(argv[1], 'rb') as f:
    em = estimator_model.EstimatorModel(
        text_format.Parse(f.read(), estimator_model_pb2.EstimatorModel())
    )

  one_bit_register_area = None
  if em.is_area_model():
    one_bit_register_area = area_model_utils.get_one_bit_register_area(em)

  compiled = em.to_compiled_proto(one_bit_register_area=one_bit_register_area)
  if _REFERENCE_GOLDEN.value:
    with open(_REFERENCE_GOLDEN.value, 'rb') as f:
      raw_golden = f.read().decode('utf-8')
    golden_proto = text_format.Parse(
        raw_golden, estimator_model_pb2.CompiledEstimatorModel()
    )
    if estimator_model.compiled_protos_are_close(compiled, golden_proto):
      print(raw_golden, end='')
      return
  print('# proto-file: xls/estimators/estimator_model.proto')
  print('# proto-message: xls.estimator_model.CompiledEstimatorModel')
  print('#')
  print('# Generated File. Do Not Edit.')
  print('# To Regenerate run xls/dev_tools/rebuild_golden_files.sh')
  print()
  print(text_format.MessageToString(compiled), end='')


if __name__ == '__main__':
  app.run(main)
