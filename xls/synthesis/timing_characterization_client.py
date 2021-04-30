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

import itertools

from typing import Dict, Sequence, Set, Tuple

from absl import flags
from absl import logging

from google.protobuf import text_format
from xls.common import gfile
from xls.common.file.python import filesystem
from xls.delay_model import delay_model_pb2
from xls.delay_model import op_module_generator
from xls.ir.op_specification import OPS
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_width', 8, 'Max width in bits to sweep.')
flags.DEFINE_integer('min_freq_mhz', 500, 'Minimum frequency to test.')
flags.DEFINE_integer('max_freq_mhz', 5000, 'Maximum frequency to test.')
flags.DEFINE_string(
    'checkpoint_path', '', 'Path at which to load and save checkpoints. ' +
    'Checkpoints will not be kept if unspecified.')

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
MISCELLANEOUS = ('kArray kArrayIndex kArrayUpdate kBitSlice kDynamicBitSlice '
                 'kConcat kDecode kEncode kOneHot kOneHotSel kParam kReverse '
                 'kSel kTuple kTupleIndex').split()


def get_bit_widths():
  return [1] + list(range(2, FLAGS.max_width + 1, 2))


def save_checkpoint(model: delay_model_pb2.DelayModel, checkpoint_path: str):
  if checkpoint_path:
    with gfile.open(checkpoint_path, 'w') as f:
      f.write(text_format.MessageToString(model))


def _synth(stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
           verilog_text: str,
           top_module_name: str) -> synthesis_pb2.CompileResponse:
  """Bisects the space of frequencies and sends requests to the server."""
  high = FLAGS.max_freq_mhz
  low = FLAGS.min_freq_mhz
  epsilon = 10

  best_result = synthesis_pb2.CompileResponse()
  while high - low > epsilon:
    current = (high + low) / 2
    request = synthesis_pb2.CompileRequest()
    request.target_frequency_hz = int(current * 10e6)
    request.module_text = verilog_text
    request.top_module_name = top_module_name
    logging.vlog(3, '--- Request')
    logging.vlog(3, request)
    response = stub.Compile(request)
    if response.slack_ps >= 0:
      logging.info('PASS at %dMHz (slack %d).', current, response.slack_ps)
      low = current
      if current > best_result.max_frequency_hz:
        best_result = response
    else:
      logging.info('FAIL at %dMHz (slack %d).', current, response.slack_ps)
      high = current

  best_result.max_frequency_hz = int(current * 10e6)
  return best_result


def _synthesize_ir(stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
                   model: delay_model_pb2.DelayModel,
                   data_points: Dict[str, Set[str]], ir_text: str, op: str,
                   result_bit_count: int,
                   operand_bit_counts: Sequence[int]) -> None:
  """Synthesizes the given IR text and returns a data point."""
  if op not in data_points:
    data_points[op] = set()

  bit_count_strs = []
  for bit_count in operand_bit_counts:
    operand = delay_model_pb2.Operation.Operand(
        bit_count=bit_count, element_count=0)
    bit_count_strs.append(str(operand))
  key = ', '.join([str(result_bit_count)] + bit_count_strs)
  if key in data_points[op]:
    return
  data_points[op].add(key)

  logging.info('Running %s with %d / %s', op, result_bit_count,
               ', '.join([str(x) for x in operand_bit_counts]))
  module_name = 'top'
  mod_generator_result = op_module_generator.generate_verilog_module(
      module_name, ir_text)
  verilog_text = mod_generator_result.verilog_text
  result = _synth(stub, verilog_text, module_name)
  logging.info('Result: %s', result)
  ps = 1e12 / result.max_frequency_hz
  result = delay_model_pb2.DataPoint()
  result.operation.op = op
  result.operation.bit_count = result_bit_count
  for bit_count in operand_bit_counts:
    operand = result.operation.operands.add()
    operand.bit_count = bit_count
  result.delay = int(ps)

  # Checkpoint after every run.
  model.data_points.append(result)
  save_checkpoint(model, FLAGS.checkpoint_path)


def _run_unary_bitwise(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize unary bitwise ops."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expr = add_op_model.estimator.regression.expressions.add()
  expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expr.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type, (op_type,))
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_variadic_bitwise(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize variadic bitwise ops."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expr = add_op_model.estimator.regression.expressions.add()
  expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expr.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  # Compute samples
  widths = get_bit_widths()
  arity = list(range(2, 8))
  combs = itertools.product(widths, arity)
  for bit_count, arity in combs:
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type,
                                                      (op_type,) * arity)
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_arithmetic_unary(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize unary ops."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expr = add_op_model.estimator.regression.expressions.add()
  expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expr.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type, (op_type,))
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_arithmetic_binary(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize arithmetic ops."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expr = add_op_model.estimator.regression.expressions.add()
  expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expr.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type,
                                                      (op_type, op_type))
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_comparison(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize comparison ops."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expr = add_op_model.estimator.regression.expressions.add()
  expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expr.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ret_type = 'bits[1]'
    ir_text = op_module_generator.generate_ir_package(op, ret_type,
                                                      (op_type, op_type))
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_shift(op: str, model: delay_model_pb2.DelayModel,
               data_points: Dict[str, Set[str]],
               stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize shift ops."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expr = add_op_model.estimator.regression.expressions.add()
  expr.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expr.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  # Compute samples
  for bit_count in get_bit_widths():
    op_type = f'bits[{bit_count}]'
    ir_text = op_module_generator.generate_ir_package(op, op_type,
                                                      (op_type, op_type))
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_extension(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Characterize extension ops (sign- and zero-)."""
  # Add op_model to protobuf message
  add_op_model = model.op_models.add(op=op)
  expression = add_op_model.estimator.regression.expressions.add()
  expression.factor.source = delay_model_pb2.DelayFactor.OPERAND_BIT_COUNT
  expression.factor.operand_number = 0

  op = ENUM2NAME_MAP[op]

  # Compute samples
  widths = get_bit_widths()
  combs = filter(lambda b: b[1] > b[0], itertools.product(widths, widths))
  for bit_count, new_bit_count in combs:
    op_type = f'bits[{bit_count}]'
    ret_type = f'bits[{new_bit_count}]'
    ir_text = op_module_generator.generate_ir_package(
        op, ret_type, (op_type,), attributes=[('new_bit_count', new_bit_count)])
    _synthesize_ir(stub, model, data_points, ir_text, op, bit_count,
                   (bit_count,))


def _run_miscellaneous(
    op: str, model: delay_model_pb2.DelayModel, data_points: Dict[str,
                                                                  Set[str]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  del op, model, data_points, stub
  pass


OPS_RUNNERS = [(UNARY_BITWISE, _run_unary_bitwise),
               (VARIADIC_BITWISE, _run_variadic_bitwise),
               (ARITHMETIC_UNARY, _run_arithmetic_unary),
               (ARITHMETIC_BINARY, _run_arithmetic_binary),
               (COMPARISON, _run_comparison), (SHIFT, _run_shift),
               (EXTENSION, _run_extension), (MISCELLANEOUS, _run_miscellaneous)]


def init_data(
    checkpoint_path: str
) -> Tuple[Dict[str, Set[str]], delay_model_pb2.DelayModel]:
  """Return new state, loading data from a checkpoint, if available."""
  data_points = {}
  model = delay_model_pb2.DelayModel()
  if checkpoint_path:
    filesystem.parse_text_proto_file(checkpoint_path, model)
    for data_point in model.data_points:
      op = data_point.operation
      if op.op not in data_points:
        data_points[op.op] = set()
      types_str = ', '.join([str(op.bit_count)] +
                            [str(x.bit_count) for x in op.operands])
      data_points[op.op].add(types_str)
  return data_points, model


def run_characterization(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub) -> None:
  """Run characterization with the given synthesis service."""
  data_points, model = init_data(FLAGS.checkpoint_path)
  for ops, runner in OPS_RUNNERS:
    for op in ops:
      runner(op, model, data_points, stub)

  print('# proto-file: xls/delay_model/delay_model.proto')
  print('# proto-message: xls.delay_model.DelayModel')
  print(model)
