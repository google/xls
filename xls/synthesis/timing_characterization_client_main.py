# Lint as: python3
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
"""Sweeps to characterize datapoints from a synthesis server.

These datapoints can be used in a delay model (where they will be interpolated)
-- the results emitted on stdout are in xls.delay_model.DelayModel prototext
format.
"""

from typing import List

from absl import app
from absl import flags
from absl import logging

import grpc

from xls.delay_model import delay_model_pb2
from xls.delay_model import op_module_generator
from xls.synthesis import client_credentials
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'Port to connect to synthesis server on.')

SLE_ALIASES = 'kSLt kSGe kSGt kULe kULt kUGe kNe kEq kNeg kDecode'.split()
FREE_OPS = ('kBitSlice kArray kArrayConcat kConcat kIdentity kLiteral kParam '
            'kReverse kTuple kTupleIndex kZeroExt').split()


def _synth(stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
           verilog_text: str,
           top_module_name: str) -> synthesis_pb2.CompileResponse:
  request = synthesis_pb2.CompileRequest()
  request.module_text = verilog_text
  request.top_module_name = top_module_name
  logging.vlog(3, '--- Request')
  logging.vlog(3, request)

  return stub.Compile(request)


def _run_binop(
    op: str, kop: str, stub: synthesis_service_pb2_grpc.SynthesisServiceStub
) -> List[delay_model_pb2.DataPoint]:
  """Characterize a binary operation via synthesis server.

  Args:
    op: Operation name to use for generating an IR package; e.g. 'add'.
    kop: Operation name to emit into datapoints, generally in kConstant form for
      use in the delay model; e.g. 'kAdd'.
    stub: Handle to the synthesis server.

  Returns:
    List of characterized datapoints via the synthesis server.

  Note that if a synthesis error occurs, it is assumed to be "out of resources"
  and is quashed, with datapoints gathered to that point returned.
  """
  results = []
  for bit_count in [1] + list(range(2, 257, 2)):
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type,
                                                      (op_type, op_type))
    module_name = f'{op}_{bit_count}'
    mod_generator_result = op_module_generator.generate_verilog_module(
        module_name, ir_text)
    top_name = module_name + '_wrapper'
    verilog_text = op_module_generator.generate_parallel_module(
        [mod_generator_result], top_name)
    try:
      result = _synth(stub, verilog_text, top_name)
    except grpc.RpcError:
      break

    ps = 1e12 / result.max_frequency_hz
    result = delay_model_pb2.DataPoint()
    result.operation.op = kop
    result.operation.bit_count = bit_count
    for _ in range(2):
      operand = result.operation.operands.add()
      operand.bit_count = bit_count
    result.delay = int(ps)
    results.append(result)

  return results


def _run_binop_and_add(
    op: str, kop: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization for the given binop and adds it to the model."""
  add_op_model = model.op_models.add(op=kop)
  expression = add_op_model.estimator.regression.expressions.add()
  expression.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expression.factor.operand_number = 0
  model.data_points.extend(_run_binop(op, kop, stub))


def run_characterization(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Runs characterization via 'stub', DelayModel to stdout as prototext."""
  model = delay_model_pb2.DelayModel()

  _run_binop_and_add('add', 'kAdd', model, stub)
  _run_binop_and_add('shll', 'kShll', model, stub)

  def add_alias(from_op: str, to_op: str):
    entry = model.op_models.add(op=from_op)
    entry.estimator.alias_op = to_op

  add_alias('kSub', to_op='kAdd')
  add_alias('kShrl', to_op='kShll')
  add_alias('kShra', to_op='kShll')
  add_alias('kDynamicBitSlice', to_op='kShll')
  add_alias('kArrayUpdate', to_op='kArrayIndex')
  for sle_alias in SLE_ALIASES:
    add_alias(sle_alias, to_op='kSLe')

  for free_op in FREE_OPS:
    entry = model.op_models.add(op=free_op)
    entry.estimator.fixed = 0

  print('# proto-file: xls/delay_model/delay_model.proto')
  print('# proto-message: xls.delay_model.DelayModel')
  print(model)


def main(argv):
  if len(argv) != 1:
    raise app.UsageError('Unexpected arguments.')

  channel_creds = client_credentials.get_credentials()
  with grpc.secure_channel(f'localhost:{FLAGS.port}', channel_creds) as channel:
    grpc.channel_ready_future(channel).result()
    stub = synthesis_service_pb2_grpc.SynthesisServiceStub(channel)

    run_characterization(stub)


if __name__ == '__main__':
  app.run(main)
