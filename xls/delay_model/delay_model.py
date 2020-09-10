# Copyright 2020 Google LLC
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

# Lint as: python3
# pylint: disable=g-long-lambda
"""Delay model for XLS operations.

The delay model estimates the delay (latency) of XLS operations when synthesized
in hardware. The delay model can both generates C++ code to compute delay as
well as provide delay estimates in Python.
"""

import abc

from typing import Sequence, Text, Tuple, Callable
import warnings

import numpy as np
from scipy import optimize as opt

from xls.delay_model import delay_model_pb2


class Error(Exception):
  pass


class Estimator(metaclass=abc.ABCMeta):
  """Base class for delay estimators.

    An Estimator provides and estimate of XLS operation delay based on
    parameters of the operation.

    Attributes:
      op: The XLS op modeled by this delay estimator. The value should
        match the name of the XLS Op enum value.  Example: 'kAdd'.
  """

  def __init__(self, op: Text):
    self.op = op

  @abc.abstractmethod
  def cpp_delay_code(self, node_identifier: Text) -> Text:
    """Returns the sequence of C++ statements which compute the delay.

    Args:
      node_identifier: The string identifier of the Node* value whose delay is
        being estimated..

    Returns:
      Sequence of C++ statements to compute the delay. The delay
      should be returned as an int64 in the C++ code. For example:

        if (node->BitCountOrDie() == 1) { return 0; }
        return 2 * node->operand_count();
    """
    raise NotImplementedError

  @abc.abstractmethod
  def operation_delay(self, operation: delay_model_pb2.Operation) -> int:
    """Returns the estimated delay for the given operation."""
    raise NotImplementedError


class FixedEstimator(Estimator):
  """A delay estimator which always returns a fixed delay."""

  def __init__(self, op, delay: int):
    super(FixedEstimator, self).__init__(op)
    self.fixed_delay = delay

  def operation_delay(self, operation: delay_model_pb2.Operation) -> int:
    return self.fixed_delay

  def cpp_delay_code(self, node_identifier: Text) -> Text:
    return 'return {};'.format(self.fixed_delay)


class AliasEstimator(Estimator):
  """An estimator which aliases another estimator for a different op.

    Operations which have very similar or identical delay characteristics (for
    example, kSub and kAdd) can be modeled using an alias. For example, the
    estimator for kSub could be an AliasEstimator which refers to kAdd.
  """

  def __init__(self, op, aliased_op: Text):
    super(AliasEstimator, self).__init__(op)
    self.aliased_op = aliased_op

  def cpp_delay_code(self, node_identifier: Text) -> Text:
    return 'return {}Delay({});'.format(
        self.aliased_op.lstrip('k'), node_identifier)

  def operation_delay(self, operation: delay_model_pb2.Operation) -> int:
    raise NotImplementedError


def delay_factor_description(factor: delay_model_pb2.DelayFactor) -> Text:
  """Returns a brief description of a delay factor."""
  e = delay_model_pb2.DelayFactor.Source
  return {
      e.RESULT_BIT_COUNT:
          'Result bit count',
      e.OPERAND_BIT_COUNT:
          'Operand %d bit count' % factor.operand_number,
      e.OPERAND_COUNT:
          'Operand count',
      e.OPERAND_ELEMENT_COUNT:
          'Operand %d element count' % factor.operand_number,
      e.OPERAND_ELEMENT_BIT_COUNT:
          'Operand %d element bit count' % factor.operand_number
  }[factor.source]


def _operation_delay_factor(factor: delay_model_pb2.DelayFactor,
                            operation: delay_model_pb2.Operation) -> int:
  """Returns the value of a delay factor extracted from an operation."""
  e = delay_model_pb2.DelayFactor.Source
  return {
      e.RESULT_BIT_COUNT:
          lambda: operation.bit_count,
      e.OPERAND_BIT_COUNT:
          lambda: operation.operands[factor.operand_number].bit_count,
      e.OPERAND_COUNT:
          lambda: len(operation.operands),
      e.OPERAND_ELEMENT_COUNT:
          lambda: operation.operands[factor.operand_number].element_count,
      e.OPERAND_ELEMENT_BIT_COUNT:
          lambda: operation.operands[factor.operand_number].bit_count,
  }[factor.source]()


def _delay_factor_cpp_expression(factor: delay_model_pb2.DelayFactor,
                                 node_identifier: Text) -> Text:
  """Returns a C++ expression which computes a delay factor of an XLS Node*.

  Args:
    factor: The delay factor to extract.
    node_identifier: The identifier of the xls::Node* to extract the factor
      from.

  Returns:
    C++ expression computing the delay factor of a node. For example, if
    the delay factor is OPERAND_COUNT, the method might return:
    'node->operand_count()'.
  """
  e = delay_model_pb2.DelayFactor.Source
  return {
      e.RESULT_BIT_COUNT:
          lambda: '{}->GetType()->GetFlatBitCount()'.format(node_identifier),
      e.OPERAND_BIT_COUNT:
          lambda: '{}->operand({})->GetType()->GetFlatBitCount()'.format(
              node_identifier, factor.operand_number),
      e.OPERAND_COUNT:
          lambda: '{}->operand_count()'.format(node_identifier),
      e.OPERAND_ELEMENT_COUNT:
          lambda: '{}->operand({})->GetType()->AsArrayOrDie()->size()'.format(
              node_identifier, factor.operand_number),
      e.OPERAND_ELEMENT_BIT_COUNT:
          lambda:
          '{}->operand({})->GetType()->AsArrayOrDie()->element_type()->GetFlatBitCount()'
          .format(node_identifier, factor.operand_number),
  }[factor.source]()


class RegressionEstimator(Estimator):
  """An estimator which uses curve fitting of measured data points.

  The curve has the form:

    delay_est = P_0 + P_1 * factor_0 + P_2 * factor_0 +
                      P_3 * factor_1 + P_4 * factor_1 +
                      ...

  Where P_i are learned parameters and factor_i are the delay factors
  extracted from the operation (for example, operand count or result bit
  count). The model supports an arbitrary number of factors.

  Attributes:
    delay_factors: The factors used in curve fitting.
    data_points: Delay measurements used by the model as DataPoint protos.
    raw_data_points: Delay measurements as tuples of ints. The first elements in
      the tuple are the delay factors and the last element is the delay.
    delay_function: The curve-fitted function which computes the estimated delay
      given the factors as floats.
    params: The list of learned parameters.
  """

  def __init__(self, op, delay_factors: Sequence[delay_model_pb2.DelayFactor],
               data_points: Sequence[delay_model_pb2.DataPoint]):
    super(RegressionEstimator, self).__init__(op)
    self.delay_factors = list(delay_factors)
    self.data_points = list(data_points)

    # Compute the raw data points for curve fitting. Each raw data point is a
    # tuple of numbers representing the delay factors and the delay. For
    # example: (factor_0, factor_1, delay).
    self.raw_data_points = []
    for dp in self.data_points:
      self.raw_data_points.append(
          tuple(
              _operation_delay_factor(f, dp.operation)
              for f in self.delay_factors) + (dp.delay - dp.delay_offset,))
    self.delay_function, self.params = self._fit_curve(self.raw_data_points)

  def _fit_curve(
      self, raw_data_points: Sequence[Tuple[int]]
  ) -> Tuple[Callable[[Sequence[float]], float], Sequence[float]]:
    """Fits a curve to the given data points.

    Args:
      raw_data_points: A sequence of tuples where each tuple is a single
        measurement point. In each tuple, independent variables are listed first
        and the dependent variable is last.

    Returns:
      A tuple containing the fitted function and the sequence of learned
      parameters.
    """
    # Split the raw data points into independent (xdata) and dependent variables
    # (ydata). Each raw data point has the form: (x_0, x_1, ... x_n, y)
    data_by_dim = list(zip(*raw_data_points))
    xdata = data_by_dim[0:-1]
    ydata = data_by_dim[-1]

    def delay_f(x, *params) -> float:
      s = params[0]
      for i in range(len(x)):
        s += params[2 * i + 1] * x[i] + params[2 * i + 2] * np.log2(x[i])
      return s

    with warnings.catch_warnings():
      warnings.filterwarnings('ignore')
      num_params = 1 + 2 * len(xdata)
      params, _ = opt.curve_fit(
          delay_f, xdata, ydata, p0=[1] * num_params, bounds=(0, np.inf))

    return lambda x: delay_f(x, *params), params

  def operation_delay(self, operation: delay_model_pb2.Operation) -> int:
    factors = tuple(
        _operation_delay_factor(f, operation) for f in self.delay_factors)
    return int(self.delay_function(factors))

  def raw_delay(self, xargs: Sequence[float]) -> float:
    """Returns the delay with delay factors passed in as floats."""
    return self.delay_function(xargs)

  def cpp_delay_code(self, node_identifier: Text) -> Text:
    terms = [str(self.params[0])]
    for i, factor in enumerate(self.delay_factors):
      f_str = _delay_factor_cpp_expression(factor, node_identifier)
      terms.append('{} * {}'.format(self.params[2 * i + 1], f_str))
      terms.append('{} * std::log2({})'.format(self.params[2 * i + 2], f_str))
    return 'return std::round({});'.format(' + '.join(terms))


class BoundingBoxEstimator(Estimator):
  """Bounding box estimator."""

  def __init__(self, op, factors: Sequence[delay_model_pb2.DelayFactor],
               data_points: Sequence[delay_model_pb2.DataPoint]):
    super(BoundingBoxEstimator, self).__init__(op)
    self.delay_factors = factors
    self.data_points = list(data_points)
    self.raw_data_points = []
    for dp in self.data_points:
      self.raw_data_points.append(
          tuple(
              _operation_delay_factor(f, dp.operation)
              for f in self.delay_factors) + (dp.delay - dp.delay_offset,))

  def cpp_delay_code(self, node_identifier: Text) -> Text:
    lines = []
    for raw_data_point in self.raw_data_points:
      test_expr_terms = []
      for i, x_value in enumerate(raw_data_point[0:-1]):
        test_expr_terms.append('%s <= %d' % (_delay_factor_cpp_expression(
            self.delay_factors[i], node_identifier), x_value))
      lines.append('if (%s) { return %d; }' %
                   (' && '.join(test_expr_terms), raw_data_point[-1]))
    lines.append(
        'return absl::UnimplementedError("Unhandled node for delay estimation: " '
        '+ {}->ToStringWithOperandTypes());'.format(node_identifier))
    return '\n'.join(lines)

  def operation_delay(self, operation: delay_model_pb2.Operation) -> int:
    xargs = tuple(
        _operation_delay_factor(f, operation) for f in self.delay_factors)
    return int(self.raw_delay(xargs))

  def raw_delay(self, xargs):
    """Returns the delay with delay factors passed in as floats."""
    for raw_data_point in self.raw_data_points:
      x_values = raw_data_point[0:-1]
      if all(a <= b for (a, b) in zip(xargs, x_values)):
        return raw_data_point[-1]
    raise Error('Operation outside bounding box')


class LogicalEffortEstimator(Estimator):
  """A delay estimator which uses logical effort computation.

  Attributes:
    tau_in_ps: The delay of a single inverter in ps.
  """

  def __init__(self, op, tau_in_ps: int):
    super(LogicalEffortEstimator, self).__init__(op)
    self.tau_in_ps = tau_in_ps

  def operation_delay(self, operation: delay_model_pb2.Operation) -> int:
    raise NotImplementedError

  def cpp_delay_code(self, node_identifier: Text) -> Text:
    lines = []
    lines.append(
        'absl::StatusOr<int64> delay_in_ps = DelayEstimator::GetLogicalEffortDelayInPs({}, {});'
        .format(node_identifier, self.tau_in_ps))
    lines.append('if (delay_in_ps.ok()) {')
    lines.append('  return delay_in_ps.value();')
    lines.append('}')
    lines.append('return delay_in_ps.status();')
    return '\n'.join(lines)


def _estimator_from_proto(op: Text, proto: delay_model_pb2.Estimator,
                          data_points: Sequence[delay_model_pb2.DataPoint]):
  """Create an Estimator from a proto."""
  if proto.HasField('fixed'):
    assert not data_points
    return FixedEstimator(op, proto.fixed)
  if proto.HasField('alias_op'):
    assert not data_points
    return AliasEstimator(op, proto.alias_op)
  if proto.HasField('regression'):
    assert data_points
    return RegressionEstimator(op, proto.regression.factors, data_points)
  if proto.HasField('bounding_box'):
    assert data_points
    return BoundingBoxEstimator(op, proto.bounding_box.factors, data_points)
  assert proto.HasField('logical_effort')
  assert not data_points
  return LogicalEffortEstimator(op, proto.logical_effort.tau_in_ps)


class OpModel:
  """Delay model for a single XLS op (e.g., kAdd).

  This abstraction mirrors the OpModel proto message in delay_model.proto.

  Attributes:
    op: The op for this model (e.g., 'kAdd').
    specializations: A map from SpecializationKind to Estimator which contains
      any specializations of the delay model of the op.
    estimator: The non-specialized Estimator to use fo this op in the general
      case.
  """

  def __init__(self, proto: delay_model_pb2.OpModel,
               data_points: Sequence[delay_model_pb2.DataPoint]):
    self.op = proto.op
    data_points = list(data_points)
    # Build a separate estimator for each specialization, if any.
    self.specializations = {}
    for specialization in proto.specializations:
      # pylint: disable=cell-var-from-loop
      pred = lambda dp: dp.operation.specialization == specialization.kind
      # Filter out the data points which correspond to the specialization.
      special_data_points = [dp for dp in data_points if pred(dp)]
      data_points = [dp for dp in data_points if not pred(dp)]
      self.specializations[specialization.kind] = _estimator_from_proto(
          self.op, specialization.estimator, special_data_points)
    self.estimator = _estimator_from_proto(self.op, proto.estimator,
                                           data_points)

  def cpp_delay_function(self) -> Text:
    """Return a C++ function which computes delay for an operation."""
    lines = []
    lines.append('absl::StatusOr<int64> %s(Node* node) {' %
                 self.cpp_delay_function_name())
    for kind, estimator in self.specializations.items():
      if kind == delay_model_pb2.SpecializationKind.OPERANDS_IDENTICAL:
        cond = ('std::all_of(node->operands().begin(), node->operands().end(), '
                '[&](Node* n) { return n == node->operand(0); })')
      elif kind == delay_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND:
        cond = ('std::any_of(node->operands().begin(), node->operands().end(), '
                '[](Node* n) { return n->Is<Literal>(); })')
      else:
        raise NotImplementedError
      lines.append('if (%s) {' % cond)
      lines.append(estimator.cpp_delay_code('node'))
      lines.append('}')
    lines.append(self.estimator.cpp_delay_code('node'))
    lines.append('}')
    return '\n'.join(lines)

  def cpp_delay_function_name(self) -> Text:
    return self.op.lstrip('k') + 'Delay'

  def cpp_delay_function_declaration(self) -> Text:
    return 'absl::StatusOr<int64> {}(Node* node);'.format(
        self.cpp_delay_function_name())


class DelayModel:
  """Delay model representing a particular hardware technology.

  Attributes:
    op_models: A map from xls::Op (e.g., 'kAdd') to the OpModel for that op.
  """

  def __init__(self, proto: delay_model_pb2.DelayModel):
    op_data_points = {}
    for data_point in proto.data_points:
      op = data_point.operation.op
      op_data_points[op] = op_data_points.get(op, []) + [data_point]

    self.op_models = {}
    for op_model in proto.op_models:
      self.op_models[op_model.op] = OpModel(op_model,
                                            op_data_points.get(op_model.op, ()))

  def ops(self) -> Sequence[Text]:
    return sorted(self.op_models.keys())

  def op_model(self, op: Text) -> OpModel:
    return self.op_models[op]
