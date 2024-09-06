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

# pylint: disable=g-long-lambda
"""Estimator model for XLS operations.

The estimator model estimates a metric (e.g., latency) of XLS operations when
synthesized in hardware. The estimator model can both generates C++ code to
compute the metric as well as provide metric estimates in Python.
"""

import abc
import dataclasses
import enum
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize as opt

from xls.estimators import estimator_model_pb2


class Error(Exception):
  pass


class Metric(enum.Enum):
  """String Enum representing metric.

  Attributes:
    DELAY_METRIC: The delay metric.
    AREA_METRIC: The area metric.
  """

  DELAY_METRIC = 'delay'
  AREA_METRIC = 'area'

  @classmethod
  def from_metric_proto(cls, proto: estimator_model_pb2.Metric) -> 'Metric':
    if proto == estimator_model_pb2.UNSPECIFIED_METRIC:
      raise ValueError('The UNSPECIFIED metric is not allowed.')
    return {
        estimator_model_pb2.Metric.DELAY_METRIC: cls.DELAY_METRIC,
        estimator_model_pb2.Metric.AREA_METRIC: cls.AREA_METRIC,
    }[proto]


class Estimator(metaclass=abc.ABCMeta):
  """Base class for estimators.

  An Estimator provides and estimate of an XLS operation metric based on
  parameters of the operation.

  Attributes:
    op: The XLS op modeled by this metric estimator. The value should match the
      name of the XLS Op enum value.  Example: 'kAdd'.
    metric: The metric computed by this estimator.
  """

  def __init__(self, op: str, metric: Metric):
    self.op = op
    self.metric = metric

  @abc.abstractmethod
  def cpp_estimation_code(self, node_identifier: str) -> str:
    """Returns the sequence of C++ statements which compute the metric.

    Args:
      node_identifier: The string identifier of the Node* value whose metric is
        being estimated.

    Returns:
      Sequence of C++ statements to compute the metric. In C++ code, the metric
        should be returned as an int64_t for delay, and double for area.
      For example, a C++ code for delay could be written as follows,

        if (node->BitCountOrDie() == 1) { return 0; }
        return 2 * node->operand_count();
    """
    raise NotImplementedError

  @abc.abstractmethod
  def operation_estimation(
      self, operation: estimator_model_pb2.Operation
  ) -> Union[int, float]:
    """Returns the estimated metric for the given operation."""
    raise NotImplementedError


class FixedEstimator(Estimator):
  """A metric estimator which always returns a constant."""

  def __init__(self, op, metric: Metric, constant: Union[int, float]):
    super().__init__(op, metric)
    self.constant = constant

  def cpp_estimation_code(self, node_identifier: str) -> str:
    return 'return {};'.format(self.constant)

  def operation_estimation(
      self, operation: estimator_model_pb2.Operation
  ) -> Union[int, float]:
    return self.constant


class AliasEstimator(Estimator):
  """An estimator which aliases another estimator for a different op.

  Operations which have very similar or identical metric characteristics (for
  example, the delay of kSub and the delay of kAdd) can be modeled using an
  alias. For example, the estimator for kSub could be an AliasEstimator which
  refers to kAdd.
  """

  def __init__(self, op, metric: Metric, aliased_op: str):
    super().__init__(op, metric)
    self.aliased_op = aliased_op

  def cpp_estimation_code(self, node_identifier: str) -> str:
    aliased_op_name = self.aliased_op.lstrip('k')
    if self.metric == Metric.DELAY_METRIC:
      metric_name = 'Delay'
    elif self.metric == Metric.AREA_METRIC:
      metric_name = 'Area'
    else:
      raise NotImplementedError(
          'OpModel.cpp_estimation_function_name for metric'
          f' "{self.metric}" is not supported!'
      )
    return f'return {aliased_op_name}{metric_name}({node_identifier});'

  def operation_estimation(
      self, operation: estimator_model_pb2.Operation
  ) -> Union[int, float]:
    raise NotImplementedError


def estimator_factor_description(
    factor: estimator_model_pb2.EstimatorFactor,
) -> str:
  """Returns a brief description of a factor."""
  e = estimator_model_pb2.EstimatorFactor.Source
  return {
      e.RESULT_BIT_COUNT: 'Result bit count',
      e.OPERAND_BIT_COUNT: 'Operand %d bit count' % factor.operand_number,
      e.OPERAND_COUNT: 'Operand count',
      e.OPERAND_ELEMENT_COUNT: (
          'Operand %d element count' % factor.operand_number
      ),
      e.OPERAND_ELEMENT_BIT_COUNT: (
          'Operand %d element bit count' % factor.operand_number
      ),
  }[factor.source]


def estimator_expression_description(
    exp: estimator_model_pb2.EstimatorExpression,
) -> str:
  """Returns a brief description of an expression."""
  if exp.HasField('bin_op'):
    lhs = estimator_expression_description(exp.lhs_expression)
    rhs = estimator_expression_description(exp.rhs_expression)
    e = estimator_model_pb2.EstimatorExpression.BinaryOperation
    if exp.bin_op == e.ADD:
      return '({} + {})'.format(lhs, rhs)
    elif exp.bin_op == e.DIVIDE:
      return '({lhs} / ({rhs} < 1.0 ? 1.0 : {rhs})'.format(lhs=lhs, rhs=rhs)
    elif exp.bin_op == e.MAX:
      return 'max({}, {})'.format(lhs, rhs)
    elif exp.bin_op == e.MIN:
      return 'min({}, {})'.format(lhs, rhs)
    elif exp.bin_op == e.MULTIPLY:
      return '({} * {})'.format(lhs, rhs)
    elif exp.bin_op == e.POWER:
      return 'pow({}, {})'.format(lhs, rhs)
    else:
      assert exp.bin_op == e.SUB
      return '({} - {})'.format(lhs, rhs)
  elif exp.HasField('factor'):
    return estimator_factor_description(exp.factor)
  else:
    assert exp.HasField('constant')
    return str(exp.constant)


def _operation_estimator_factor(
    factor: estimator_model_pb2.EstimatorFactor,
    operation: estimator_model_pb2.Operation,
) -> int:
  """Returns the value of an estimator factor extracted from an operation."""
  e = estimator_model_pb2.EstimatorFactor.Source
  if factor.source == e.RESULT_BIT_COUNT:
    return operation.bit_count
  elif factor.source == e.OPERAND_BIT_COUNT:
    operand = operation.operands[factor.operand_number]
    if operand.element_count > 0:
      return operand.bit_count * operand.element_count
    else:
      return operand.bit_count
  elif factor.source == e.OPERAND_COUNT:
    return len(operation.operands)
  elif factor.source == e.OPERAND_ELEMENT_COUNT:
    return operation.operands[factor.operand_number].element_count
  else:
    assert factor.source == e.OPERAND_ELEMENT_BIT_COUNT
    return operation.operands[factor.operand_number].bit_count


def _operation_estimator_expression(
    expression: estimator_model_pb2.EstimatorExpression,
    operation: estimator_model_pb2.Operation,
) -> int:
  """Returns the value of a estimator expression extracted from an operation."""
  if expression.HasField('bin_op'):
    assert expression.HasField('lhs_expression')
    assert expression.HasField('rhs_expression')
    lhs_value = _operation_estimator_expression(
        expression.lhs_expression, operation
    )
    rhs_value = _operation_estimator_expression(
        expression.rhs_expression, operation
    )
    e = estimator_model_pb2.EstimatorExpression.BinaryOperation
    return {
        e.ADD: lambda: lhs_value + rhs_value,
        e.DIVIDE: lambda: lhs_value / (1.0 if rhs_value < 1.0 else rhs_value),
        e.MAX: lambda: max(lhs_value, rhs_value),
        e.MIN: lambda: min(lhs_value, rhs_value),
        e.MULTIPLY: lambda: lhs_value * rhs_value,
        e.POWER: lambda: lhs_value**rhs_value,
        e.SUB: lambda: lhs_value - rhs_value,
    }[expression.bin_op]()

  if expression.HasField('factor'):
    return _operation_estimator_factor(expression.factor, operation)

  assert expression.HasField('constant')
  return expression.constant


def _estimator_factor_cpp_expression(
    factor: estimator_model_pb2.EstimatorFactor, node_identifier: str
) -> str:
  """Returns a C++ expression which computes a estimator factor of an XLS Node*.

  Args:
    factor: The estimator factor to extract.
    node_identifier: The identifier of the xls::Node* to extract the factor
      from.

  Returns:
    C++ expression computing the estimator factor of a node. For example, if
    the estimator factor is OPERAND_COUNT, the method might return:
    'node->operand_count()'.
  """
  e = estimator_model_pb2.EstimatorFactor.Source
  return {
      e.RESULT_BIT_COUNT: lambda: '{}->GetType()->GetFlatBitCount()'.format(
          node_identifier
      ),
      e.OPERAND_BIT_COUNT: lambda: (
          '{}->operand({})->GetType()->GetFlatBitCount()'.format(
              node_identifier, factor.operand_number
          )
      ),
      e.OPERAND_COUNT: lambda: '{}->operand_count()'.format(node_identifier),
      e.OPERAND_ELEMENT_COUNT: lambda: (
          '{}->operand({})->GetType()->AsArrayOrDie()->size()'.format(
              node_identifier, factor.operand_number
          )
      ),
      e.OPERAND_ELEMENT_BIT_COUNT: lambda: '{}->operand({})->GetType()->AsArrayOrDie()->element_type()->GetFlatBitCount()'.format(
          node_identifier, factor.operand_number
      ),
  }[factor.source]()


def _estimator_expression_cpp_expression(
    expression: estimator_model_pb2.EstimatorExpression, node_identifier: str
) -> str:
  """Returns a C++ expression which computes an estimator expression of an XLS Node*.

  Args:
    expression: The expression to extract.
    node_identifier: The identifier of the xls::Node* to extract the factor
      from.

  Returns:
    C++ expression computing the estimator expression of a node.
  """
  if expression.HasField('bin_op'):
    assert expression.HasField('lhs_expression')
    assert expression.HasField('rhs_expression')
    lhs_value = _estimator_expression_cpp_expression(
        expression.lhs_expression, node_identifier
    )
    rhs_value = _estimator_expression_cpp_expression(
        expression.rhs_expression, node_identifier
    )
    e = estimator_model_pb2.EstimatorExpression.BinaryOperation
    return {
        e.ADD: lambda: '({} + {})'.format(lhs_value, rhs_value),
        e.DIVIDE: lambda: '({lhs} / ({rhs} < 1.0 ? 1.0 : {rhs}))'.format(
            lhs=lhs_value, rhs=rhs_value
        ),
        e.MAX: lambda: 'std::max({}, {})'.format(lhs_value, rhs_value),
        e.MIN: lambda: 'std::min({}, {})'.format(lhs_value, rhs_value),
        e.MULTIPLY: lambda: '({} * {})'.format(lhs_value, rhs_value),
        e.POWER: lambda: 'pow({}, {})'.format(lhs_value, rhs_value),
        e.SUB: lambda: '({} - {})'.format(lhs_value, rhs_value),
    }[expression.bin_op]()

  if expression.HasField('factor'):
    return 'static_cast<float>({})'.format(
        _estimator_factor_cpp_expression(expression.factor, node_identifier)
    )

  assert expression.HasField('constant')
  return 'static_cast<float>({})'.format(expression.constant)


@dataclasses.dataclass
class RawDataPoint:
  """Measurements used by RegressionEstimator and BoundingBoxEstimator."""

  factors: Union[List[int], List[float]]  # x in y = f(x)
  measurement: Union[int, float]  # y in y = f(x)

  @classmethod
  def from_data_point_proto(
      cls,
      metric: Metric,
      dp: estimator_model_pb2.DataPoint,
      estimator_factors: Union[List[int], List[float]],
  ) -> 'RawDataPoint':
    """As a DataPoint contains information for all metrics, this method extracts a RawDataPoint for a given metric."""
    if metric == Metric.DELAY_METRIC:
      measurement = dp.delay - dp.delay_offset
    elif metric == Metric.AREA_METRIC:
      measurement = dp.total_area - dp.sequential_area
    else:
      raise NotImplementedError(
          f'DataPoint to RawDataPoint for metric "{metric}" is not supported!'
      )
    return cls(
        factors=estimator_factors,
        measurement=measurement,
    )


class RegressionEstimator(Estimator):
  """An estimator which uses curve fitting of measured data points.

  The curve has the form:

    delay_est = P_0 + P_1 * factor_0 + P_2 * log2(factor_0) +
                      P_3 * factor_1 + P_4 * log2(factor_1) +
                      ...

  Where P_i are learned parameters and factor_i are the estimator expressions
  extracted from the operation (for example, operand count or result bit
  count or some mathematical combination thereof). The model supports an
  arbitrary number of expressions.

  Attributes:
    estimator_expressions: The expressions used in curve fitting.
    data_points: Measurements used by the model as DataPoint protos.
    raw_data_points: Measurements stored in RawDataPoint structures. The
      .factors list contains the estimator expressions, and the .measurement
      field is the measured data of a given metric.
    estimator_function: The curve-fitted function which computes the estimated
      metric given the expressions as floats.
    params: The list of learned parameters.
    num_cross_validation_folds: The number of folds to use for cross validation.
    max_data_point_error: The maximum allowable absolute error for any single
      data point.
    max_fold_geomean_error: The maximum allowable geomean absolute error over
      all data points in a given test set.
  """

  def __init__(
      self,
      op,
      metric: Metric,
      estimator_expressions: Sequence[estimator_model_pb2.EstimatorExpression],
      data_points: Sequence[estimator_model_pb2.DataPoint],
      num_cross_validation_folds: int = 5,
      max_data_point_error: float = np.inf,
      max_fold_geomean_error: float = np.inf,
  ):
    super().__init__(op, metric)
    self.estimator_expressions = list(estimator_expressions)
    self.data_points = list(data_points)

    # Compute the raw data points for curve fitting. Each raw data point is a
    # tuple of numbers representing the estimator expressions and the
    # measurement. For example: (expression_0, expression_1, measurement).
    self.raw_data_points: List[RawDataPoint] = []
    for dp in self.data_points:
      factors = [
          _operation_estimator_expression(expression, dp.operation)
          for expression in estimator_expressions
      ]
      self.raw_data_points.append(
          RawDataPoint.from_data_point_proto(metric, dp, factors)
      )

    self._k_fold_cross_validation(
        self.raw_data_points,
        num_cross_validation_folds,
        max_data_point_error,
        max_fold_geomean_error,
    )
    self.estimator_function, self.params = self._fit_curve(self.raw_data_points)

  @classmethod
  def generate_k_fold_cross_validation_train_and_test_sets(
      cls,
      raw_data_points: Sequence[RawDataPoint],
      num_cross_validation_folds: int,
  ):
    """Yields training and testing datasets for cross validation.

    Args:
      raw_data_points: The sequence of data points.
      num_cross_validation_folds: Number of cross-validation folds.

    Yields:
      Yields training and testing datasets for cross
      validation. 'num_cross_validation_folds' number of training and testing
      datasets for use in cross validation.
    """

    # Separate data into num_cross_validation_folds sets
    random.seed(0)
    randomized_data_points = random.sample(
        raw_data_points, len(raw_data_points)
    )
    folds = []
    for fold_idx in range(num_cross_validation_folds):
      folds.append([
          dp
          for idx, dp in enumerate(randomized_data_points)
          if idx % num_cross_validation_folds == fold_idx
      ])

    # Generate train  and test data points.
    for test_fold_idx in range(num_cross_validation_folds):
      training_dps = []
      for fold_idx, fold_dps in enumerate(folds):
        if fold_idx == test_fold_idx:
          continue
        training_dps.extend(fold_dps)
      yield training_dps, folds[test_fold_idx]

  def _k_fold_cross_validation(
      self,
      raw_data_points: Sequence[RawDataPoint],
      num_cross_validation_folds: int,
      max_data_point_error: float,
      max_fold_geomean_error: float,
  ):
    """Performs k-fold cross validation to verify the model.

    An exception is raised if the model does not pass cross validation.  Note
    that this function modifies self.delay_function to perform regression on
    partial data sets.

    Args:
      raw_data_points: A sequence of RawDataPoints, where each is a single
        measurement point.  Independent variables are in the .factors field, and
        the dependent variable is in the .measurement field.
      num_cross_validation_folds: The number of folds to use for cross
        validation.
      max_data_point_error: The maximum allowable absolute error for any single
        data point.
      max_fold_geomean_error: The maximum allowable geomean absolute error over
        all data points in a given test set.

    Raises:
      Error: Raised if the model does not pass cross validation.  Note
        that this function modifies self.delay_function to perform regression on
        partial data sets.
    """
    if max_data_point_error == np.inf and max_fold_geomean_error == np.inf:
      return
    if num_cross_validation_folds > len(raw_data_points):
      raise Error(
          '{}: Too few data points to cross validate: '
          '{} data points, {} folds'.format(
              self.op, len(raw_data_points), num_cross_validation_folds
          )
      )

    # Perform validation for each training and testing set.
    for (
        training_dps,
        testing_dps,
    ) in RegressionEstimator.generate_k_fold_cross_validation_train_and_test_sets(
        raw_data_points, num_cross_validation_folds=num_cross_validation_folds
    ):

      # Train.
      self.estimator_function, self.params = self._fit_curve(training_dps)

      # Test.
      error_product = 1.0
      for dp in testing_dps:
        xdata = dp.factors
        ydata = dp.measurement
        predicted_y = self.raw_estimation(xdata)
        abs_dp_error = abs((predicted_y - ydata) / ydata)
        error_product = error_product * abs_dp_error
        if abs_dp_error > max_data_point_error:
          raise Error(
              '{}: Regression model failed k-fold cross validation for '
              'data point {} with absolute error {} > max {}'.format(
                  self.op, dp, abs_dp_error, max_data_point_error
              )
          )
      geomean_error = error_product ** (1.0 / len(testing_dps))
      if geomean_error > max_fold_geomean_error:
        raise Error(
            '{}: Regression model failed k-fold cross validation for '
            'test set with geomean error {} > max {}'.format(
                self.op, geomean_error, max_fold_geomean_error
            )
        )

  def _fit_curve(
      self, raw_data_points: Sequence[RawDataPoint]
  ) -> Tuple[Callable[[Sequence[float]], float], np.ndarray]:
    """Fits a curve to the given data points.

    Args:
      raw_data_points: A sequence of RawDataPoints, where each is a single
        measurement point. Independent variables are in the .factors field, and
        the dependent variable is in the .measurement field.

    Returns:
      A tuple containing the fitted function and the sequence of learned
      parameters.
    """
    # Split the raw data points into independent (xdata) and dependent variables
    # (ydata).
    raw_xdata = np.array(
        [pt.factors for pt in raw_data_points], dtype=np.float64
    )
    ydata = np.transpose([pt.measurement for pt in raw_data_points])

    # Construct our augmented "independent" variables in a matrix:
    # xdata = [1, x0, log2(x0), x1, log2(x1), ...]
    def augment_xdata(x_arr: np.ndarray) -> np.ndarray:
      x_augmented = np.ones(
          (x_arr.shape[0], 1 + 2 * x_arr.shape[1]), dtype=np.float64
      )
      x_augmented[::, 1::2] = x_arr
      x_augmented[::, 2::2] = np.log2(np.maximum(1.0, x_arr))
      return x_augmented

    xdata = augment_xdata(raw_xdata)

    # Now, the least-squares solution to the equation xdata @ p = ydata is
    # exactly the set of parameters for our model! EXCEPT: we want to make sure
    # none of the weights are negative, since we expect all terms to have net
    # positive contribution. This helps make sure extrapolations are reasonable.
    params = opt.nnls(xdata, ydata)[0]

    def f(x) -> float:
      x_augmented = augment_xdata(np.array([x], dtype=np.float64))
      return np.dot(x_augmented, params)[0]

    return f, params.flatten()

  def operation_estimation(
      self, operation: estimator_model_pb2.Operation
  ) -> Union[int, float]:
    expressions = tuple(
        _operation_estimator_expression(e, operation)
        for e in self.estimator_expressions
    )
    if self.metric == Metric.DELAY_METRIC:
      return int(self.estimator_function(expressions))
    elif self.metric == Metric.AREA_METRIC:
      return self.estimator_function(expressions)
    else:
      raise NotImplementedError(
          f'RegressionEstimator.operation_estimation for metric "{self.metric}"'
          ' is not supported!'
      )

  def raw_estimation(self, xargs: Sequence[float]) -> float:
    """Returns the estimation with estimator expressions passed in as floats."""
    return self.estimator_function(xargs)

  def cpp_estimation_code(self, node_identifier: str) -> str:
    terms = [repr(float(self.params[0]))]
    for i, expression in enumerate(self.estimator_expressions):
      e_str = _estimator_expression_cpp_expression(expression, node_identifier)
      terms.append('{!r} * {}'.format(float(self.params[2 * i + 1]), e_str))
      terms.append(
          '{w!r} * std::log2({e} < 1.0 ? 1.0 : {e})'.format(
              w=float(self.params[2 * i + 2]), e=e_str
          )
      )
    terms_str = ' + '.join(terms)
    if self.metric == Metric.DELAY_METRIC:
      return f'return std::round({terms_str});'
    elif self.metric == Metric.AREA_METRIC:
      return f'return {terms_str};'
    else:
      raise NotImplementedError(
          f'RegressionEstimator.cpp_estimation_code for metric "{self.metric}"'
          ' is not supported!'
      )


class BoundingBoxEstimator(Estimator):
  """Bounding box estimator."""

  def __init__(
      self,
      op,
      metric: Metric,
      estimator_factors: Sequence[estimator_model_pb2.EstimatorFactor],
      data_points: Sequence[estimator_model_pb2.DataPoint],
  ):
    super().__init__(op, metric)
    self.estimator_factors = estimator_factors
    self.data_points = list(data_points)
    self.raw_data_points = []
    for dp in self.data_points:
      factors = [
          _operation_estimator_factor(estimator_factor, dp.operation)
          for estimator_factor in self.estimator_factors
      ]
      self.raw_data_points.append(
          RawDataPoint.from_data_point_proto(metric, dp, factors)
      )

  def cpp_estimation_code(self, node_identifier: str) -> str:
    lines = []
    for raw_data_point in self.raw_data_points:
      test_expr_terms = []
      for i, x_value in enumerate(raw_data_point.factors):
        test_expr_terms.append(
            '%s <= %d'
            % (
                _estimator_factor_cpp_expression(
                    self.estimator_factors[i], node_identifier
                ),
                x_value,
            )
        )
      lines.append(
          'if (%s) { return %d; }'
          % (' && '.join(test_expr_terms), raw_data_point.measurement)
      )
    lines.append(
        'return absl::UnimplementedError('
        f'"Unhandled node for {str(self.metric.value)} estimation: " '
        '+ {}->ToStringWithOperandTypes());'.format(node_identifier)
    )
    return '\n'.join(lines)

  def operation_estimation(
      self, operation: estimator_model_pb2.Operation
  ) -> Union[int, float]:
    xargs = tuple(
        _operation_estimator_factor(f, operation)
        for f in self.estimator_factors
    )
    return int(self.raw_estimation(xargs))

  def raw_estimation(self, xargs):
    """Returns the estimation with factors passed in as floats."""
    for raw_data_point in self.raw_data_points:
      x_values = raw_data_point.factors
      if all(a <= b for (a, b) in zip(xargs, x_values)):
        return raw_data_point.measurement
    raise Error('Operation outside bounding box')


class LogicalEffortEstimator(Estimator):
  """A delay estimator which uses logical effort computation.

  This is specific to delay estimation. Other metrics are not supported.

  Attributes:
    tau_in_ps: The delay of a single inverter in ps.
  """

  def __init__(self, op, metric: Metric, tau_in_ps: int):
    super().__init__(op, metric)
    if metric != Metric.DELAY_METRIC:
      raise NotImplementedError(
          f'LogicalEffortEstimator for metric "{metric}" is not supported!'
      )
    self.tau_in_ps = tau_in_ps

  def operation_estimation(
      self, operation: estimator_model_pb2.Operation
  ) -> Union[int, float]:
    raise NotImplementedError

  def cpp_estimation_code(self, node_identifier: str) -> str:
    lines = []
    lines.append(
        'absl::StatusOr<int64_t> delay_in_ps = '
        'DelayEstimator::GetLogicalEffortDelayInPs({}, {});'.format(
            node_identifier, self.tau_in_ps
        )
    )
    lines.append('if (delay_in_ps.ok()) {')
    lines.append('  return delay_in_ps.value();')
    lines.append('}')
    lines.append('return delay_in_ps.status();')
    return '\n'.join(lines)


def _estimator_from_proto(
    op: str,
    metric: Metric,
    proto: estimator_model_pb2.Estimator,
    data_points: Sequence[estimator_model_pb2.DataPoint],
):
  """Create an Estimator from a proto."""
  if proto.HasField('fixed'):
    assert not data_points
    return FixedEstimator(op, metric, proto.fixed)
  if proto.HasField('alias_op'):
    assert not data_points
    return AliasEstimator(op, metric, proto.alias_op)
  if proto.HasField('regression'):
    assert data_points
    keyword_dict = dict()
    if proto.regression.HasField('kfold_validator'):
      optional_args = [
          'num_cross_validation_folds',
          'max_data_point_error',
          'max_fold_geomean_error',
      ]
      for arg in optional_args:
        if proto.regression.kfold_validator.HasField(arg):
          keyword_dict[arg] = getattr(proto.regression.kfold_validator, arg)
    return RegressionEstimator(
        op, metric, proto.regression.expressions, data_points, **keyword_dict
    )
  if proto.HasField('bounding_box'):
    assert data_points
    return BoundingBoxEstimator(
        op, metric, proto.bounding_box.factors, data_points
    )
  assert proto.HasField('logical_effort')
  assert not data_points
  return LogicalEffortEstimator(op, metric, proto.logical_effort.tau_in_ps)


def _is_matching_operation(
    op: estimator_model_pb2.Operation,
    specialization: estimator_model_pb2.OpModel.Specialization,
):
  """Check if the operation matches the specialization in kind & details."""
  if op.specialization != specialization.kind:
    return False

  if (
      specialization.kind
      == estimator_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND
  ):
    spec_details = specialization.details.literal_operand_details
    op_details = op.literal_operand_details
    if spec_details.allowed_nonliteral_operand and not set(
        op_details.nonliteral_operand
    ).issubset(spec_details.allowed_nonliteral_operand):
      return False
    if spec_details.required_literal_operand and not set(
        op_details.literal_operand
    ).issuperset(spec_details.required_literal_operand):
      return False

  return True


@dataclasses.dataclass(frozen=True)
class LiteralOperandDetails:
  """A container for multi-modal output features of data providers."""

  allowed_nonliteral_operands: frozenset[int] = frozenset()
  required_literal_operands: frozenset[int] = frozenset()


@dataclasses.dataclass(frozen=True)
class SpecializationDetails:
  """A container for multi-modal output features of data providers."""

  literal_operand_details: Optional[LiteralOperandDetails] = None


class OpModel:
  """Estimator model for a single XLS op (e.g., kAdd).

  This abstraction mirrors the OpModel proto message in estimator_model.proto.

  Attributes:
    op: The op for this model (e.g., 'kAdd').
    metric: The metric this model is estimating.
    specializations: A map from SpecializationKind to Estimator which contains
      any specializations of the estimator model of the op.
    estimator: The non-specialized Estimator to use for this op in the general
      case.
  """

  def __init__(
      self,
      metric: Metric,
      proto: estimator_model_pb2.OpModel,
      data_points: Sequence[estimator_model_pb2.DataPoint],
  ):
    self.op = proto.op
    self.metric = metric
    data_points = list(data_points)
    # Build a separate estimator for each specialization, if any.
    self.specializations = {}
    for specialization in proto.specializations:
      # pylint: disable=cell-var-from-loop
      pred = lambda dp: _is_matching_operation(dp.operation, specialization)

      # Filter out the data points which correspond to the specialization.
      special_data_points = [dp for dp in data_points if pred(dp)]
      data_points = [dp for dp in data_points if not pred(dp)]
      details = None
      if (
          specialization.kind
          == estimator_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND
      ):
        details = SpecializationDetails(
            LiteralOperandDetails(
                allowed_nonliteral_operands=frozenset(
                    specialization.details.literal_operand_details.allowed_nonliteral_operand
                ),
                required_literal_operands=frozenset(
                    specialization.details.literal_operand_details.required_literal_operand
                ),
            )
        )
      self.specializations[(specialization.kind, details)] = (
          _estimator_from_proto(
              self.op,
              self.metric,
              specialization.estimator,
              special_data_points,
          )
      )
    self.estimator = _estimator_from_proto(
        self.op, self.metric, proto.estimator, data_points
    )

  def cpp_estimation_function(self) -> str:
    """Return a C++ function which estimates a metric for an operation."""
    metric_return_type = {
        Metric.DELAY_METRIC: 'int64_t',
        Metric.AREA_METRIC: 'double',
    }[self.metric]

    lines = []
    lines.append(
        f'absl::StatusOr<{metric_return_type}>'
        f' {self.cpp_estimation_function_name()}(Node* node) {{'
    )
    nonliteral_operands_tracked = False
    for (kind, details), estimator in self.specializations.items():
      if kind == estimator_model_pb2.SpecializationKind.OPERANDS_IDENTICAL:
        cond = (
            'std::all_of(node->operands().begin(), node->operands().end(), '
            '[&](Node* n) { return n == node->operand(0); })'
        )
      elif kind == estimator_model_pb2.SpecializationKind.HAS_LITERAL_OPERAND:
        cond = (
            'std::any_of(node->operands().begin(), node->operands().end(), '
            '[](Node* n) { return n->Is<Literal>(); })'
        )
        if details and details.literal_operand_details:
          literal_operand_details = details.literal_operand_details
          if literal_operand_details.required_literal_operands:
            cond = ' && '.join([
                f'node->operand({i})->Is<Literal>()'
                for i in literal_operand_details.required_literal_operands
            ])
          if literal_operand_details.allowed_nonliteral_operands:
            if not nonliteral_operands_tracked:
              lines.extend([
                  (
                      f'absl::flat_hash_set<{metric_return_type}>'
                      ' nonliteral_operands;'
                  ),
                  'for (int64_t i = 0; i < node->operands().size(); ++i) {',
                  '  if (!node->operand(i)->Is<Literal>()) { ',
                  '    nonliteral_operands.insert(i);',
                  '  }',
                  '}',
              ])
              nonliteral_operands_tracked = True
            cond += (
                ' && std::all_of('
                'nonliteral_operands.begin(), '
                'nonliteral_operands.end(), '
                '[&](int64_t i) {{ '
                '  return {};'
                '}})'
            ).format(
                ' || '.join(
                    f'i == {operand}'
                    for operand in literal_operand_details.allowed_nonliteral_operands
                )
            )
      else:
        raise NotImplementedError
      lines.append('if (%s) {' % cond)
      lines.append(estimator.cpp_estimation_code('node'))
      lines.append('}')
    lines.append(self.estimator.cpp_estimation_code('node'))
    lines.append('}')
    return '\n'.join(lines)

  def cpp_estimation_function_name(self) -> str:
    if self.metric == Metric.DELAY_METRIC:
      suffix = 'Delay'
    elif self.metric == Metric.AREA_METRIC:
      suffix = 'Area'
    else:
      raise NotImplementedError(
          'OpModel.cpp_estimation_function_name for metric'
          f' "{self.metric}" is not supported!'
      )
    return self.op.lstrip('k') + suffix

  def cpp_estimation_function_declaration(self) -> str:
    if self.metric == Metric.DELAY_METRIC:
      return 'absl::StatusOr<int64_t> {}(Node* node);'.format(
          self.cpp_estimation_function_name()
      )
    elif self.metric == Metric.AREA_METRIC:
      return 'absl::StatusOr<double> {}(Node* node);'.format(
          self.cpp_estimation_function_name()
      )
    else:
      raise NotImplementedError(
          'OpModel.cpp_estimation_function_declaration for metric'
          f' "{self.metric}" is not supported!'
      )


class EstimatorModel:
  """Delay model representing a particular hardware technology.

  Attributes:
    metric: The metric this model is estimating.
    op_models: A map from xls::Op (e.g., 'kAdd') to the OpModel for that op.
  """

  def __init__(self, proto: estimator_model_pb2.EstimatorModel):
    op_data_points = {}
    self.metric = Metric.from_metric_proto(proto.metric)
    for data_point in proto.data_points:
      op = data_point.operation.op
      op_data_points[op] = op_data_points.get(op, []) + [data_point]

    self.op_models = {}
    for op_model in proto.op_models:
      self.op_models[op_model.op] = OpModel(
          self.metric, op_model, op_data_points.get(op_model.op, ())
      )

  def ops(self) -> Sequence[str]:
    return sorted(self.op_models.keys())

  def op_model(self, op: str) -> OpModel:
    return self.op_models[op]

  def is_delay_model(self) -> bool:
    return self.metric == Metric.DELAY_METRIC

  def is_area_model(self) -> bool:
    return self.metric == Metric.AREA_METRIC
