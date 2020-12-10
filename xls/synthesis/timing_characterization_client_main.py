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

from itertools import product
from typing import Callable
from typing import List
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import grpc

from xls.delay_model import op_module_generator
from xls.delay_model import delay_model_pb2
from xls.ir.op_specification import OPS
from xls.synthesis import client_credentials
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc


FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'Port to connect to synthesis server on.')
flags.DEFINE_integer('max_width', 8, 'Max width in bits to sweep.')


ENUM2NAME_MAP = dict((op.enum_name, op.name) for op in OPS)

# https://google.github.io/xls/ir_semantics/#unary-bitwise-operations
UNARY_BITWISE = 'kIdentity kNot'.split()

# https://google.github.io/xls/ir_semantics/#variadic-bitwise-operations
VARIADIC_BITWISE = 'kAnd kOr kXor'.split()

# https://google.github.io/xls/ir_semantics/#arithmetic-unary-operations
ARITHMETIC_UNARY = 'kNeg'.split()

# https://google.github.io/xls/ir_semantics/#arithmetic-binary-operations
ARITHMETIC_BINARY = 'kAdd kSDiv kSMul kSub kUDiv kUMul'.split()

# https://google.github.io/xls/ir_semantics/#comparison-operations
COMPARISON = 'kEq kNe kSGe kSGt kSLe kSLt kUGe kUGt kULe kULt'.split()

# https://google.github.io/xls/ir_semantics/#shift-operations
SHIFT = 'kShll kShra kShrl'.split()

# https://google.github.io/xls/ir_semantics/#extension-operations
EXTENSION = 'kZeroExt kSignExt'.split()

# https://google.github.io/xls/ir_semantics/#miscellaneous-operations
MISCELLANEOUS = 'kArray kArrayIndex kArrayUpdate kBitSlice kDynamicBitSlice kConcat kDecode kEncode kOneHot kOneHotSel kParam kReverse kSel kTuple kTupleIndex'.split()


def get_bit_widths():
  return [1] + list(range(2, FLAGS.max_width + 1, 2))


def _synth(stub: synthesis_service_pb2_grpc.SynthesisServiceStub, verilog_text: str, top_module_name: str) -> synthesis_pb2.CompileResponse:
  request = synthesis_pb2.CompileRequest()
  request.module_text = verilog_text
  request.top_module_name = top_module_name
  logging.vlog(3, '--- Request')
  logging.vlog(3, request)

  return stub.Compile(request)


def _synthesize_ir(stub: synthesis_service_pb2_grpc.SynthesisServiceStub, ir_text: str, op: str, result_bit_count: int,
                  operand_bit_counts = Sequence[int]) -> delay_model_pb2.DataPoint:
  module_name = 'top'
  mod_generator_result = op_module_generator.generate_verilog_module(module_name, ir_text)
  verilog_text = mod_generator_result.verilog_text
  result = _synth(stub, verilog_text, module_name)
  ps = 1e12 / result.max_frequency_hz
  result = delay_model_pb2.DataPoint()
  result.operation.op = op
  result.operation.bit_count = result_bit_count
  for bit_count in operand_bit_counts:
    operand = result.operation.operands.add()
    operand.bit_count = bit_count
  result.delay = int(ps)
  return result


def _run_unary_bitwise(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  # Compute samples
  results = []
  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type, (op_type,))
    result = _synthesize_ir(stub, ir_text, op, bit_count, (bit_count,))
    results.append(result)
  model.data_points.extend(results)


def _run_variadic_bitwise(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  # Compute samples
  widths = get_bit_widths()
  arity = list(range(2, 8))
  combs = product(widths, arity)
  results = []
  for bit_count, arity in combs:
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type, (op_type,) * arity)
    result = _synthesize_ir(stub, ir_text, op, bit_count, (bit_count,) * arity)
    results.append(result)
  model.data_points.extend(results)


def _run_arithmetic_unary(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  results = []
  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type, (op_type,))
    result = _synthesize_ir(stub, ir_text, op, bit_count, (bit_count,))
    results.append(result)
  model.data_points.extend(results)


def _run_arithmetic_binary(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  results = []
  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type, (op_type, op_type))
    result = _synthesize_ir(stub, ir_text, op, bit_count, (bit_count, bit_count))
    results.append(result)
  model.data_points.extend(results)


def _run_comparison(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  results = []
  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ret_type = f'bits[1]'
    ir_text = op_module_generator.generate_ir_package(op, ret_type, (op_type, op_type))
    result = _synthesize_ir(stub, ir_text, op, 1, (bit_count, bit_count))
    results.append(result)
  model.data_points.extend(results)


def _run_shift(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  # Compute samples
  results = []
  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(
      op, op_type, (op_type, op_type))
    result = _synthesize_ir(stub, ir_text, op, bit_count, (bit_count,))
    results.append(result)
  model.data_points.extend(results)


def _run_extension(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  add_op_model.estimator.regression.factors.add(
    source=delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT,
    operand_number=0)

  op = ENUM2NAME_MAP[op]

  # Compute samples
  widths = get_bit_widths()
  combs = filter(lambda bits: bits[1] > bits[0], product(widths, widths))
  results = []
  for bit_count, new_bit_count in combs:
    op_type = f'bits[{bit_count}]'
    ret_type = f'bits[{new_bit_count}]'
    ir_text = op_module_generator.generate_ir_package(
      op, ret_type, (op_type,),
      attributes=[('new_bit_count', new_bit_count)])
    result = _synthesize_ir(stub, ir_text, op, new_bit_count, (bit_count, 64))
    results.append(result)
  model.data_points.extend(results)


def _run_miscellaneous(
    op: str, model: delay_model_pb2.DelayModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  pass


OPS_RUNNERS = [
  (UNARY_BITWISE, _run_unary_bitwise),
  (VARIADIC_BITWISE, _run_variadic_bitwise),
  (ARITHMETIC_UNARY, _run_arithmetic_unary),
  (ARITHMETIC_BINARY, _run_arithmetic_binary),
  (COMPARISON, _run_comparison),
  (SHIFT, _run_shift),
  (EXTENSION, _run_extension),
  (MISCELLANEOUS, _run_miscellaneous)
]


def run_characterization(stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  model = delay_model_pb2.DelayModel()

  for ops, runner in OPS_RUNNERS:
    for op in ops:
      runner(op, model, stub)

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
