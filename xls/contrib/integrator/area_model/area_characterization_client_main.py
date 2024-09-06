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

These datapoints can be used in an area model (where they may be interpolated)
-- the results emitted on stdout are in xls.estimator_model.EstimatorModel
prototext
format.
"""

import functools
import math
import operator
from typing import List, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
import grpc

from xls.estimators import estimator_model
from xls.estimators import estimator_model_pb2
from xls.estimators.delay_model import op_module_generator
from xls.synthesis import client_credentials
from xls.synthesis import synthesis_pb2
from xls.synthesis import synthesis_service_pb2_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'Port to connect to synthesis server on.')

FREE_OPS = (
    'kArray kArrayConcat kBitSlice kConcat kIdentity '
    'kLiteral kParam kReverse kTuple kTupleIndex kZeroExt kSignExt'
).split()

logging.set_verbosity(logging.INFO)

# Configure cross fold validation.
NUM_CROSS_VALIDATION_FOLDS = 5
MAX_FOLD_GEOMEAN_ERROR = 0.15

# Standard bitwidth strides.
# Use bigger strides for more-nested / slower-running loops.
BITWIDTH_STRIDE_DEGREES = [2, 4, 12, 16]


def _array_elements_sweep():
  """Yields a standard range of array elements along a single dimension."""

  # array_elements <= 8 noisy
  # larger array_elements extremely slow to synthesize (especially w/ nesting)
  for dim_size in range(12, 37, 12):
    yield dim_size


def _bitwidth_sweep(stride_degree: int = 0):
  """Yields a standard range of bitwidth.

  Increasing stride_degree increases the stride of the range (maximum
  stride_degree = 3).

  Args:
    stride_degree: Index into BITWIDTH_STRIDE_DEGREES indicating the stride
      amount.
  """
  for bit_count in range(8, 71, BITWIDTH_STRIDE_DEGREES[stride_degree]):
    yield bit_count


def _operand_count_sweep():
  """Yields a standard range of operand counts."""
  for input_count in range(2, 16, 1):
    yield input_count


# Helper functions for setting expression fields
def _set_add_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.ADD


def _set_sub_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.SUB


def _set_divide_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.DIVIDE


def _set_max_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.MAX


def _set_min_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.MIN


def _set_multiply_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.MULTIPLY


def _set_power_expression(expr: estimator_model_pb2.EstimatorExpression):
  expr.bin_op = estimator_model_pb2.EstimatorExpression.BinaryOperation.POWER


def _set_constant_expression(
    expr: estimator_model_pb2.EstimatorExpression, value: int
):
  expr.constant = value


def _set_result_bit_count_expression_factor(
    expr: estimator_model_pb2.EstimatorExpression, add_constant=0
):
  if add_constant != 0:
    _set_add_expression(expr)
    _set_result_bit_count_expression_factor(expr.rhs_expression)
    _set_constant_expression(expr.lhs_expression, add_constant)
  else:
    expr.factor.source = estimator_model_pb2.EstimatorFactor.RESULT_BIT_COUNT


def _set_addressable_element_count_factor(
    expr: estimator_model_pb2.EstimatorExpression,
):
  expr.factor.source = estimator_model_pb2.EstimatorFactor.OPERAND_ELEMENT_COUNT


def _set_operand_count_expression_factor(
    expr: estimator_model_pb2.EstimatorExpression, add_constant=0
):
  if add_constant != 0:
    _set_add_expression(expr)
    _set_operand_count_expression_factor(expr.rhs_expression)
    _set_constant_expression(expr.lhs_expression, add_constant)
  else:
    expr.factor.source = estimator_model_pb2.EstimatorFactor.OPERAND_COUNT


def _set_operand_bit_count_expression_factor(
    expr: estimator_model_pb2.EstimatorExpression,
    operand_idx: int,
    add_constant=0,
):
  """Sets the operand bit count of the expression factor."""
  if add_constant != 0:
    _set_add_expression(expr)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, operand_idx)
    _set_constant_expression(expr.lhs_expression, add_constant)
  else:
    expr.factor.source = estimator_model_pb2.EstimatorFactor.OPERAND_BIT_COUNT
    expr.factor.operand_number = operand_idx


def _new_expression(
    op_model: estimator_model_pb2.OpModel,
) -> estimator_model_pb2.EstimatorExpression:
  """Add a new expression to op_model and return."""
  return op_model.estimator.regression.expressions.add()


def _new_regression_op_model(
    model: estimator_model_pb2.EstimatorModel,
    kop: str,
    result_bit_count: bool = False,
    operand_count: bool = False,
    operand_bit_counts: Sequence[int] = (),
) -> estimator_model_pb2.OpModel:
  """Add to 'model' a new regression model for op 'kop'."""
  add_op_model = model.op_models.add(op=kop)
  add_op_model.estimator.regression.kfold_validator.num_cross_validation_folds = (
      NUM_CROSS_VALIDATION_FOLDS
  )
  add_op_model.estimator.regression.kfold_validator.max_fold_geomean_error = (
      MAX_FOLD_GEOMEAN_ERROR
  )
  if result_bit_count:
    result_bits_expr = _new_expression(add_op_model)
    _set_result_bit_count_expression_factor(result_bits_expr)
  if operand_count:
    operand_count_expr = _new_expression(add_op_model)
    _set_operand_count_expression_factor(operand_count_expr)
  for operand_idx in operand_bit_counts:
    operand_bit_count_expr = _new_expression(add_op_model)
    _set_operand_bit_count_expression_factor(
        operand_bit_count_expr, operand_idx
    )
  return add_op_model


def _synth(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    verilog_text: str,
    top_module_name: str,
) -> synthesis_pb2.CompileResponse:
  request = synthesis_pb2.CompileRequest()
  request.module_text = verilog_text
  request.top_module_name = top_module_name
  logging.vlog(3, '--- Request')
  logging.vlog(3, request)

  return stub.Compile(request)


def _record_area(
    response: synthesis_pb2.CompileResponse,
    result: estimator_model_pb2.DataPoint,
):
  """Extracts area from the server response and writes it to the datapoint."""
  cell_histogram = response.instance_count.cell_histogram
  if 'SB_LUT4' in cell_histogram:
    result.delay = cell_histogram['SB_LUT4']
  else:
    result.delay = 0


def _get_type_from_dimensions(dimensions: List[int]):
  """Return a bits or nested bit array type with 'dimensions' dimensions."""
  bits_type = 'bits'
  for dim in dimensions:
    bits_type += f'[{dim}]'
  return bits_type


def _synthesize_op_and_make_bare_data_point(
    op: str,
    kop: str,
    op_type: str,
    operand_types: List[str],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    attributes: Sequence[Tuple[str, str]] = (),
    literal_operand: Optional[int] = None,
) -> estimator_model_pb2.DataPoint:
  """Characterize an operation via synthesis server.

  Sets the area and op of the data point but not any other information about the
  node / operands.

  Args:
    op: Operation name to use for generating an IR package; e.g. 'add'.
    kop: Operation name to emit into datapoints, generally in kConstant form for
      use in the delay model; e.g. 'kAdd'.
    op_type: The type of the operation result.
    operand_types: The type of each operation.
    stub: Handle to the synthesis server.
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operations. Forwarded to generate_ir_package.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter. Forwarded to generate_ir_package.

  Returns:
    datapoint produced via the synthesis server with the area and op (but no
    other fields)
    set.
  """
  ir_text = op_module_generator.generate_ir_package(
      op, op_type, operand_types, attributes, literal_operand
  )
  module_name_safe_op_type = op_type.replace('[', '_').replace(']', '')
  module_name = f'{op}_{module_name_safe_op_type}'
  mod_generator_result = op_module_generator.generate_verilog_module(
      module_name, ir_text
  )
  top_name = module_name + '_wrapper'
  verilog_text = op_module_generator.generate_parallel_module(
      [mod_generator_result], top_name
  )

  response = _synth(stub, verilog_text, top_name)
  result = estimator_model_pb2.DataPoint()
  _record_area(response, result)
  result.operation.op = kop

  return result


def _build_data_point(
    op: str,
    kop: str,
    node_dimensions: List[int],
    operand_dimensions: List[List[int]],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    attributes: Sequence[Tuple[str, str]] = (),
    literal_operand: Optional[int] = None,
) -> estimator_model_pb2.DataPoint:
  """Characterize an operation via synthesis server.

  Sets the area and op of the data point but not any other information about the
  node / operands.

  Args:
    op: Operation name to use for generating an IR package; e.g. 'add'.
    kop: Operation name to emit into datapoints, generally in kConstant form for
      use in the delay model; e.g. 'kAdd'.
    node_dimensions: The dimensions of the operation result (bits type or nested
      array of bits).
    operand_dimensions: The dimensions of each operation (bits type or nested
      array of bits).
    stub: Handle to the synthesis server.
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operations. Forwarded to generate_ir_package.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter. Forwarded to generate_ir_package:

  Returns:
    datapoint produced via the synthesis server with the area and op (but no
    other fields)
    set.
  """
  op_type = _get_type_from_dimensions(node_dimensions)
  operand_types = []
  for operand_dims in operand_dimensions:
    operand_types.append(_get_type_from_dimensions(operand_dims))

  result = _synthesize_op_and_make_bare_data_point(
      op, kop, op_type, operand_types, stub, attributes, literal_operand
  )
  return result


def _build_data_point_bit_types(
    op: str,
    kop: str,
    num_node_bits: int,
    num_operand_bits: List[int],
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    attributes: Sequence[Tuple[str, str]] = (),
    literal_operand: Optional[int] = None,
) -> estimator_model_pb2.DataPoint:
  """Characterize an operation with bit type input and output via synthesis server.

  Args:
    op: Operation name to use for generating an IR package; e.g. 'add'.
    kop: Operation name to emit into datapoints, generally in kConstant form for
      use in the delay model; e.g. 'kAdd'.
    num_node_bits: The number of bits of the operation result.
    num_operand_bits: The number of bits of each operation.
    stub: Handle to the synthesis server.
    attributes: Attributes to include in the operation mnemonic. For example,
      "new_bit_count" in extend operations. Forwarded to generate_ir_package.
    literal_operand: Optionally specifies that the given operand number should
      be substituted with a randomly generated literal instead of a function
      parameter. Forwarded to generate_ir_package.

  Returns:
    Complete datapoint for the op, including bitwidths of tehe node and
    operands.
  """
  op_type = _get_type_from_dimensions([num_node_bits])
  operand_types = []
  for operand_bit_count in num_operand_bits:
    operand_types.append(_get_type_from_dimensions([operand_bit_count]))

  result = _synthesize_op_and_make_bare_data_point(
      op, kop, op_type, operand_types, stub, attributes, literal_operand
  )
  result.operation.bit_count = num_node_bits
  for operand_bit_count in num_operand_bits:
    operand = result.operation.operands.add()
    operand.bit_count = operand_bit_count
  return result


def _run_nary_op(
    op: str,
    kop: str,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    num_inputs: int,
) -> List[estimator_model_pb2.DataPoint]:
  """Characterizes an nary op."""
  results = []
  for bit_count in _bitwidth_sweep(0):
    results.append(
        _build_data_point_bit_types(
            op, kop, bit_count, [bit_count] * num_inputs, stub
        )
    )
    logging.info(
        '# nary_op: %s, %s bits, %s inputs --> %s',
        op,
        str(bit_count),
        str(num_inputs),
        str(results[-1].delay),
    )

  return results


def _run_linear_bin_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the given linear bin_op and adds it to the model."""
  _new_regression_op_model(model, kop, result_bit_count=True)
  model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=2))
  # Validate model using k-fold validation in RegressionEstimator
  estimator_model.EstimatorModel(model)


def _run_quadratic_bin_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    signed: bool = False,
) -> None:
  """Runs characterization for the quadratic bin_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop)

  # result_bit_count * result_bit_count
  # This is because the sign bit is special.
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  constant = -1 if signed else 0
  _set_result_bit_count_expression_factor(
      expr.lhs_expression, add_constant=constant
  )
  _set_result_bit_count_expression_factor(
      expr.rhs_expression, add_constant=constant
  )

  model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=2))
  # Validate model
  estimator_model.EstimatorModel(model)


def _run_unary_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    signed=False,
) -> None:
  """Runs characterization for the given unary_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop)
  expr = _new_expression(add_op_model)
  constant = -1 if signed else 0
  _set_result_bit_count_expression_factor(expr, add_constant=constant)

  model.data_points.extend(_run_nary_op(op, kop, stub, num_inputs=1))
  # Validate model
  estimator_model.EstimatorModel(model)


def _run_nary_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the given nary_op and adds it to the model."""
  add_op_model = _new_regression_op_model(model, kop)
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_result_bit_count_expression_factor(expr.lhs_expression)
  # Note - for most operand counts, this works much better as
  # operand_count-1.  However, for low operand count (e.g. 2,4)
  # we can fit multiple ops inside a LUT, so the number of operands
  # has little effect until higher operand counts. So, weirdly,
  # we get lower error overall by just using operand_count...
  _set_operand_count_expression_factor(expr.rhs_expression)

  for input_count in _operand_count_sweep():
    model.data_points.extend(
        _run_nary_op(op, kop, stub, num_inputs=input_count)
    )
  # Validate model
  estimator_model.EstimatorModel(model)


def _run_single_bit_result_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
    num_inputs: int,
) -> None:
  """Runs characterization for an op that always produce a single-bit output."""
  _new_regression_op_model(model, kop, operand_bit_counts=[0])

  for input_bits in _bitwidth_sweep(0):
    logging.info('# reduction_op: %s, %s bits', op, str(input_bits))
    model.data_points.append(
        _build_data_point_bit_types(op, kop, 1, [input_bits] * num_inputs, stub)
    )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_reduction_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  return _run_single_bit_result_op_and_add(op, kop, model, stub, num_inputs=1)


def _run_comparison_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  return _run_single_bit_result_op_and_add(op, kop, model, stub, num_inputs=2)


def _run_select_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the select op."""
  add_op_model = _new_regression_op_model(model, kop)

  # operand_count * result_bit_count
  # Alternatively, try pow(2, operand_bit_count(0)) * result_bit_count
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_operand_count_expression_factor(expr.lhs_expression, add_constant=-2)
  _set_result_bit_count_expression_factor(expr.rhs_expression)

  # Enumerate cases and bitwidth.
  # Note: at 7 and 8 cases, there is a weird dip in LUTs at around 40 bits wide
  # Why? No idea...
  for num_cases in _operand_count_sweep():
    for bit_count in _bitwidth_sweep(0):

      # Handle differently if num_cases is a power of 2.
      select_bits = (num_cases - 1).bit_length()
      if math.pow(2, select_bits) == num_cases:
        model.data_points.append(
            _build_data_point_bit_types(
                op,
                kop,
                bit_count,
                [select_bits] + ([bit_count] * num_cases),
                stub,
            )
        )
      else:
        model.data_points.append(
            _build_data_point_bit_types(
                op,
                kop,
                bit_count,
                [select_bits] + ([bit_count] * (num_cases + 1)),
                stub,
            )
        )
        logging.info(
            '# select_op: %s, %s bits, %s cases --> %s',
            op,
            str(bit_count),
            str(num_cases),
            str(model.data_points[-1].delay),
        )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_one_hot_select_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the one hot select op."""
  add_op_model = _new_regression_op_model(model, kop)

  # operand_bit_count(0) * result_bit_count
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
  _set_result_bit_count_expression_factor(expr.rhs_expression)

  # Enumerate cases and bitwidth.
  for num_cases in _operand_count_sweep():
    for bit_count in _bitwidth_sweep(0):
      select_bits = num_cases
      model.data_points.append(
          _build_data_point_bit_types(
              op,
              kop,
              bit_count,
              [select_bits] + ([bit_count] * num_cases),
              stub,
          )
      )
      logging.info(
          '# one_hot_select_op: %s, %s bits, %s cases --> %s',
          op,
          str(bit_count),
          str(num_cases),
          str(model.data_points[-1].delay),
      )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_encode_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the encode op."""
  _new_regression_op_model(model, kop, operand_bit_counts=[0])

  # input_bits should be at least 2 bits.
  for input_bits in _bitwidth_sweep(0):
    node_bits = (input_bits - 1).bit_length()
    model.data_points.append(
        _build_data_point_bit_types(op, kop, node_bits, [input_bits], stub)
    )
    logging.info(
        '# encode_op: %s, %s input bits --> %s',
        op,
        str(input_bits),
        str(model.data_points[-1].delay),
    )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_decode_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the decode op."""
  _new_regression_op_model(model, kop, result_bit_count=True)

  # node_bits should be at least 2 bits.
  for node_bits in _bitwidth_sweep(0):
    input_bits = (node_bits - 1).bit_length()
    model.data_points.append(
        _build_data_point_bit_types(
            op,
            kop,
            node_bits,
            [input_bits],
            stub,
            attributes=[('width', str(node_bits))],
        )
    )
    logging.info(
        '# encode_op: %s, %s bits --> %s',
        op,
        str(node_bits),
        str(model.data_points[-1].delay),
    )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_dynamic_bit_slice_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the dynamic bit slice op."""
  add_op_model = _new_regression_op_model(model, kop)

  # ~= result_bit_count * operand_bit_count[1] (start bits)
  # Hard to model this well - in theory, this should be something
  # more like result_bit_count * 2 ^ start bits.  However,
  # as we add more result bits, more work gets eliminated / reduced
  # (iff 2 ^ start bits + result width > input bits).
  mul_expr = _new_expression(add_op_model)
  _set_multiply_expression(mul_expr)
  _set_result_bit_count_expression_factor(mul_expr.lhs_expression)
  _set_operand_bit_count_expression_factor(mul_expr.rhs_expression, 1)

  # input_bits should be at least 2 bits
  idx = 0
  for input_bits in _bitwidth_sweep(2):
    for start_bits in range(3, (input_bits - 1).bit_length() + 1):
      for node_bits in range(1, input_bits, BITWIDTH_STRIDE_DEGREES[2]):
        model.data_points.append(
            _build_data_point_bit_types(
                op,
                kop,
                node_bits,
                [input_bits, start_bits],
                stub,
                attributes=[('width', str(node_bits))],
            )
        )
        logging.info(
            '# idx: %s, dynamic_bit_slice_op: %s, %s start bits, '
            '%s input bits, %s width --> %s',
            str(idx),
            op,
            str(start_bits),
            str(input_bits),
            str(node_bits),
            str(model.data_points[-1].delay),
        )
        idx = idx + 1

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_one_hot_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the one hot op."""
  add_op_model = _new_regression_op_model(model, kop)

  # operand_bit_counts[0] * operand_bit_count[0]
  # The highest priority gate has ~1 gate (actually, a wire)
  # The lowest priority gate has O(n) gates (reduction tree of higher
  # priority inputs)
  # sum(1->n) is quadratic
  expr = _new_expression(add_op_model)
  _set_multiply_expression(expr)
  _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
  _set_operand_bit_count_expression_factor(expr.rhs_expression, 0)

  for bit_count in _bitwidth_sweep(0):
    # lsb / msb priority or the same logic but mirror image.
    model.data_points.append(
        _build_data_point_bit_types(
            op,
            kop,
            bit_count + 1,
            [bit_count],
            stub,
            attributes=[('lsb_prio', 'true')],
        )
    )
    logging.info(
        '# one_hot: %s, %s input bits --> %s',
        op,
        str(bit_count),
        str(model.data_points[-1].delay),
    )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_mul_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for a mul op."""

  def _get_big_op_bitcount_expr():
    """Returns larger bit count of the two operands."""
    expr = estimator_model_pb2.EstimatorExpression()
    _set_max_expression(expr)
    _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, 1)
    return expr

  def _get_small_op_bitcount_expr():
    """Returns smaller bit count of the two operands."""
    expr = estimator_model_pb2.EstimatorExpression()
    _set_min_expression(expr)
    _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, 1)
    return expr

  def _get_zero_expr():
    """Returns a constant 0 expression."""
    expr = estimator_model_pb2.EstimatorExpression()
    _set_constant_expression(expr, 0)
    return expr

  def _get_meaningful_width_bits_expr():
    """Returns the maximum number of meaningful result bits."""
    expr = estimator_model_pb2.EstimatorExpression()
    _set_add_expression(expr)
    _set_operand_bit_count_expression_factor(expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(expr.rhs_expression, 1)
    return expr

  def _get_bounded_width_offset_domain(begin_expr, end_expr):
    """Gives the bounded offset of result_bit_count.

    Gives the bounded offset of result_bit_count into the
    range[begin_expr, end_expr] e.g.:

    for begin_expr = 2, end_expr = 4, we map: 1 --> 0 2 --> 0 3 --> 1 4
    --> 2 5 --> 2 6 --> 2 etc.

    expr = min(end_expr, max(begin_expr, result_bit_count)) - begin_expr

    Args:
      begin_expr: begin of range.
      end_expr: end of range.

    Returns:
      Bounded offset of result_bit_count.
    """
    expr = estimator_model_pb2.EstimatorExpression()
    _set_sub_expression(expr)
    expr.rhs_expression.CopyFrom(begin_expr)

    min_expr = expr.lhs_expression
    _set_min_expression(min_expr)
    min_expr.lhs_expression.CopyFrom(end_expr)

    max_expr = min_expr.rhs_expression
    _set_max_expression(max_expr)
    max_expr.lhs_expression.CopyFrom(begin_expr)
    _set_result_bit_count_expression_factor(max_expr.rhs_expression)

    return expr

  def _get_rectangle_area(width_expr, height_expr):
    """Return the area of a rectangle of dimensions width_expr * height_expr."""
    expr = estimator_model_pb2.EstimatorExpression()
    _set_multiply_expression(expr)
    expr.lhs_expression.CopyFrom(width_expr)
    expr.rhs_expression.CopyFrom(height_expr)
    return expr

  def _get_triangle_area(width_expr):
    """Return the area of a isosceles right triangle.

    Width_expr expression: _get_triangle_area(width_expr, width_expr) / 2

    Args:
      width_expr: Width expression.

    Returns:
      The area of a isosceles right triangle.
    """
    expr = estimator_model_pb2.EstimatorExpression()
    _set_divide_expression(expr)
    sqr_expression = _get_rectangle_area(width_expr, width_expr)
    expr.lhs_expression.CopyFrom(sqr_expression)
    _set_constant_expression(expr.rhs_expression, 2)
    return expr

  def _get_partial_triangle_area(width_expr, max_width_expr):
    """Return the area of a partial isosceles right triangle.

         |    /|    -|
         |   / |     |
         |  /  |     |
         | /   |     |
         |/    |     |
         |     |     | heigh = maximum_width
        /|     |     |
       / |area |     |
      /  |     |     |
     /   |     |     |
    /____|_____|    -|
         |

         |_____|
          width

    |___________|
    maximum_width

    expr = rectangle_area(width, maximum_width) - triangle_area(width)

    Args:
      width_expr: Width expression.
      max_width_expr: Max width expression.

    Returns:
      Area of partial isosceles right triangle.
    """
    expr = estimator_model_pb2.EstimatorExpression()
    _set_sub_expression(expr)

    rectangle_expr = _get_rectangle_area(width_expr, max_width_expr)
    expr.lhs_expression.CopyFrom(rectangle_expr)

    triangle_expr = _get_triangle_area(width_expr)
    expr.rhs_expression.CopyFrom(triangle_expr)
    return expr

  # Compute for multiply can be divided into 3 regions.

  # Regions:
  #  A         B      C
  #      |---------|-----   -
  #     /|         |    /    |
  #    / |         |   /     |
  #   /  |         |  /      | height = min(op[0], op[1])
  #  /   |         | /       |
  # /    |         |/        |
  # -----|---------|        -

  # |______________|
  # max(op[0],op[1])

  # |____|
  # min(op[0], op[1])

  # |____________________|
  # max(op[0],op[1]) + min(op[0], op[1])

  # *Math works out the same whether op[0] or [op1] is larger.

  # expr = area(region C) + (area(region B) + area(region A))
  # Top level add
  add_op_model = _new_regression_op_model(model, kop)
  outer_add_expr = _new_expression(add_op_model)
  _set_add_expression(outer_add_expr)

  # precompute min/max(op[0], op[1])
  big_op_bitcount_expr = _get_big_op_bitcount_expr()
  small_op_bitcount_expr = _get_small_op_bitcount_expr()

  # Region C
  region_c_domain_expr = _get_bounded_width_offset_domain(
      _get_zero_expr(), small_op_bitcount_expr
  )
  region_c_area_expr = _get_triangle_area(region_c_domain_expr)
  outer_add_expr.lhs_expression.CopyFrom(region_c_area_expr)

  # Inner add
  inner_add_expr = outer_add_expr.rhs_expression
  _set_add_expression(inner_add_expr)

  # Region B
  region_b_domain_expr = _get_bounded_width_offset_domain(
      small_op_bitcount_expr, big_op_bitcount_expr
  )
  region_b_area_expr = _get_rectangle_area(
      region_b_domain_expr, small_op_bitcount_expr
  )
  inner_add_expr.lhs_expression.CopyFrom(region_b_area_expr)

  # Region A
  region_a_domain_expr = _get_bounded_width_offset_domain(
      big_op_bitcount_expr, _get_meaningful_width_bits_expr()
  )
  region_a_area_expr = _get_partial_triangle_area(
      region_a_domain_expr, small_op_bitcount_expr
  )
  inner_add_expr.rhs_expression.CopyFrom(region_a_area_expr)

  # All bit counts should be at least 2
  for mplier_count in _bitwidth_sweep(2):
    for mcand_count in _bitwidth_sweep(2):
      for node_count in _bitwidth_sweep(2):
        model.data_points.append(
            _build_data_point_bit_types(
                op, kop, node_count, [mplier_count, mcand_count], stub
            )
        )
        logging.info(
            '# mul: %s, %s * %s, %s node count, result_bits --> %s',
            op,
            str(mplier_count),
            str(mcand_count),
            str(node_count),
            str(model.data_points[-1].delay),
        )

  # Validate model
  estimator_model.EstimatorModel(model)


def _yield_array_dimension_sizes_helper(
    num_dimensions: int, dimension_sizes: List[int]
):
  """Yields a sweep of array dimension size lists.

  Yields a sweep of array dimension size lists for 'num_dimension' number of
  dimensions'. 'dimension_sizes' is used to recursively build up the size list.

  Args:
    num_dimensions: Number of dimensions.
    dimension_sizes: Sizes of dimensions.

  Yields:
    A sweep of array dimension size lists.
  """
  if num_dimensions <= 0:
    yield dimension_sizes
  else:
    for dim_size in _array_elements_sweep():
      dimension_sizes[-num_dimensions] = dim_size
      yield from _yield_array_dimension_sizes_helper(
          num_dimensions - 1, dimension_sizes
      )


def _yield_array_dimension_sizes(num_dimensions: int):
  """Yields a sweep of array dimension size lists for 'num_dimension' number of dimensions'."""
  dimension_sizes = [-1] * num_dimensions
  yield from _yield_array_dimension_sizes_helper(
      num_dimensions, dimension_sizes
  )


def _get_array_num_elements(
    array_dims: List[int], index_depth: Optional[int] = None
):
  """Returns the number of elements in a (nested) array.

  Returns the number of elements in a (nested) array with dimensions
  'array_dims'.  If the array is indexed 'index_depth' times. If 'index_depth'
  is not specified, the maximum number of possible indices is assumed.

  Args:
    array_dims: Array dimensions.
    index_depth: Depth of index.

  Returns:
    The number of elements in a (nested) array with dimensions 'array_dims'.
  """
  if index_depth is None:
    index_depth = len(array_dims) - 1

  elem_count = 1
  for idx in range(len(array_dims) - index_depth, len(array_dims)):
    elem_count = elem_count * array_dims[idx]
  return elem_count


def _run_array_index_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the ArrayIndex op."""
  add_op_model = _new_regression_op_model(model, kop)

  # Area is a function of #elements*weight + elements*bitwidth*weight.
  #
  # This seems to hold across a range of element counts, bitwidth, and number
  # of dimensions i.e.
  #
  # The weight isn't an artifact of where we sampled data - It is actually
  # ~constant rather than being something like the ratio of #elements to
  # #bitwidths or similar.

  def _set_addressable_element_count_expression(elm_expr):
    _set_divide_expression(elm_expr)
    _set_operand_bit_count_expression_factor(elm_expr.lhs_expression, 0)
    _set_result_bit_count_expression_factor(elm_expr.rhs_expression)

  elm_expr = _new_expression(add_op_model)
  _set_addressable_element_count_expression(elm_expr)
  mul_expr = _new_expression(add_op_model)
  _set_multiply_expression(mul_expr)
  _set_addressable_element_count_expression(mul_expr.lhs_expression)
  _set_result_bit_count_expression_factor(mul_expr.rhs_expression)

  for num_dims in range(1, 3):
    for array_dimension_sizes in _yield_array_dimension_sizes(num_dims):

      # If single-dimension array, increase number of elements.
      if num_dims == 1:
        assert len(array_dimension_sizes) == 1
        array_dimension_sizes[0] = array_dimension_sizes[0] * 2

      for element_bit_count in _bitwidth_sweep(3):
        array_and_element_dimensions = [
            element_bit_count
        ] + array_dimension_sizes

        # Format dimension args
        operand_dimensions = [array_and_element_dimensions]
        for dim in reversed(array_dimension_sizes):
          operand_dimensions.append([(dim - 1).bit_length()])

        # Record data point
        result = _build_data_point(
            op, kop, [element_bit_count], operand_dimensions, stub
        )
        result.operation.bit_count = element_bit_count
        operand = result.operation.operands.add()
        operand.bit_count = functools.reduce(
            operator.mul, array_and_element_dimensions, 1
        )
        model.data_points.append(result)

        logging.info(
            '%s: %s --> %s',
            str(kop),
            ','.join(str(item) for item in operand_dimensions),
            str(result.delay),
        )

  # Validate model
  estimator_model.EstimatorModel(model)


def _run_array_update_op_and_add(
    op: str,
    kop: str,
    model: estimator_model_pb2.EstimatorModel,
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization for the ArrayUpdate op."""
  add_op_model = _new_regression_op_model(model, kop)

  # Area is a function of #elements*weight + elements*bitwidth*weight.
  #
  # This seems to hold across a range of element counts, bitwidth, and number
  # of dimensions i.e.
  #
  # The weight isn't an artifact of where we sampled data - It is actually
  # ~constant rather than being something like the ratio of #elements to
  # #bitwidths or similar.

  def _set_addressable_element_count_expression(elm_expr):
    _set_divide_expression(elm_expr)
    _set_operand_bit_count_expression_factor(elm_expr.lhs_expression, 0)
    _set_operand_bit_count_expression_factor(elm_expr.rhs_expression, 1)

  elm_expr = _new_expression(add_op_model)
  _set_addressable_element_count_expression(elm_expr)
  mul_expr = _new_expression(add_op_model)
  _set_multiply_expression(mul_expr)
  _set_addressable_element_count_expression(mul_expr.lhs_expression)
  _set_operand_bit_count_expression_factor(mul_expr.rhs_expression, 1)

  for num_dims in range(1, 3):
    for array_dimension_sizes in _yield_array_dimension_sizes(num_dims):

      # If single-dimension array, increase number of elements.
      if num_dims == 1:
        assert len(array_dimension_sizes) == 1
        array_dimension_sizes[0] = array_dimension_sizes[0] * 2

      for element_bit_count in _bitwidth_sweep(3):
        array_and_element_dimensions = [
            element_bit_count
        ] + array_dimension_sizes

        # Format dimension args
        operand_dimensions = [array_and_element_dimensions]
        operand_dimensions.append([element_bit_count])
        for dim in reversed(array_dimension_sizes):
          operand_dimensions.append([(dim - 1).bit_length()])

        # Record data point
        result = _build_data_point(
            op, kop, array_and_element_dimensions, operand_dimensions, stub
        )
        array_operand = result.operation.operands.add()
        array_operand.bit_count = functools.reduce(
            operator.mul, array_and_element_dimensions, 1
        )
        new_elm_operand = result.operation.operands.add()
        new_elm_operand.bit_count = element_bit_count
        model.data_points.append(result)

        logging.info(
            '%s: %s --> %s',
            str(kop),
            ','.join(str(item) for item in operand_dimensions),
            str(result.delay),
        )

  # Validate model
  estimator_model.EstimatorModel(model)


def run_characterization(
    stub: synthesis_service_pb2_grpc.SynthesisServiceStub,
) -> None:
  """Runs characterization via 'stub', DelayModel to stdout as prototext."""
  model = estimator_model_pb2.EstimatorModel()

  # Bin ops
  _run_linear_bin_op_and_add('add', 'kAdd', model, stub)
  _run_linear_bin_op_and_add('sub', 'kSub', model, stub)
  # Observed shift data is noisy.
  _run_linear_bin_op_and_add('shll', 'kShll', model, stub)
  _run_linear_bin_op_and_add('shrl', 'kShrl', model, stub)
  _run_linear_bin_op_and_add('shra', 'kShra', model, stub)

  _run_quadratic_bin_op_and_add('sdiv', 'kSDiv', model, stub, signed=True)
  _run_quadratic_bin_op_and_add('smod', 'kSMod', model, stub, signed=True)
  _run_quadratic_bin_op_and_add('udiv', 'kUDiv', model, stub)
  _run_quadratic_bin_op_and_add('umod', 'kUMod', model, stub)

  # Unary ops
  _run_unary_op_and_add('neg', 'kNeg', model, stub, signed=True)
  _run_unary_op_and_add('not', 'kNot', model, stub)

  # Nary ops
  _run_nary_op_and_add('and', 'kAnd', model, stub)
  _run_nary_op_and_add('nand', 'kNand', model, stub)
  _run_nary_op_and_add('nor', 'kNor', model, stub)
  _run_nary_op_and_add('or', 'kOr', model, stub)
  _run_nary_op_and_add('xor', 'kXor', model, stub)

  # Reduction ops
  _run_reduction_op_and_add('and_reduce', 'kAndReduce', model, stub)
  _run_reduction_op_and_add('or_reduce', 'kOrReduce', model, stub)
  _run_reduction_op_and_add('xor_reduce', 'kXorReduce', model, stub)

  # Comparison ops
  _run_comparison_op_and_add('eq', 'kEq', model, stub)
  _run_comparison_op_and_add('ne', 'kNe', model, stub)
  # Note: Could optimize for sign - accuracy gains from
  # sign have been marginal so far, though. These ops
  # also cost less than smul / sdiv anyway.
  _run_comparison_op_and_add('sge', 'kSGe', model, stub)
  _run_comparison_op_and_add('sgt', 'kSGt', model, stub)
  _run_comparison_op_and_add('sle', 'kSLe', model, stub)
  _run_comparison_op_and_add('slt', 'kSLt', model, stub)
  _run_comparison_op_and_add('uge', 'kUGe', model, stub)
  _run_comparison_op_and_add('ugt', 'kUGt', model, stub)
  _run_comparison_op_and_add('ule', 'kULe', model, stub)
  _run_comparison_op_and_add('ult', 'kULt', model, stub)

  # Select ops
  # For functions only called for 1 op, could just encode
  # op and kOp into function.  However, perfer consistency
  # and readability of passing them in as args.
  # Note: Select op observed data is really weird, see _run_select_op_and_add
  _run_select_op_and_add('sel', 'kSel', model, stub)
  _run_one_hot_select_op_and_add('one_hot_sel', 'kOneHotSel', model, stub)

  # Encode ops
  _run_encode_op_and_add('encode', 'kEncode', model, stub)
  _run_decode_op_and_add('decode', 'kDecode', model, stub)

  # Dynamic bit slice op
  _run_dynamic_bit_slice_op_and_add(
      'dynamic_bit_slice', 'kDynamicBitSlice', model, stub
  )

  # One hot op
  _run_one_hot_op_and_add('one_hot', 'kOneHot', model, stub)

  # Mul ops
  # Note: Modeling smul w/ sign bit as with sdiv decreases accuracy.
  _run_mul_op_and_add('smul', 'kSMul', model, stub)
  _run_mul_op_and_add('umul', 'kUMul', model, stub)

  # Array ops
  _run_array_index_op_and_add('array_index', 'kArrayIndex', model, stub)
  _run_array_update_op_and_add('array_update', 'kArrayUpdate', model, stub)

  # Add free ops.
  for free_op in FREE_OPS:
    entry = model.op_models.add(op=free_op)
    entry.estimator.fixed = 0

  # Final validation
  estimator_model.EstimatorModel(model)

  print('# proto-file: xls/estimators/estimator_model.proto')
  print('# proto-message: xls.estimator_model.EstimatorModel')
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
