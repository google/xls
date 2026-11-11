// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/estimators/proto_estimator_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/estimator_model.pb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/op_list.h"
#include "xls/ir/type.h"

namespace xls {

absl::StatusOr<Op> EnumNameToOp(std::string_view enum_name) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string_view, Op>>
      kEnumMap({
#define FROM_OP_ENUM_NAME(name, a, b, c) {#name, Op::name},
          XLS_FOR_EACH_OP_TYPE(FROM_OP_ENUM_NAME)
#undef FROM_OP_ENUM_NAME
      });
  auto it = kEnumMap->find(enum_name);
  if (it == kEnumMap->end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown Op enum name: ", enum_name));
  }
  return it->second;
}

namespace {

absl::StatusOr<int64_t> EvalEstimatorFactor(
    const estimator_model::EstimatorFactor& factor, Node* node) {
  auto get_operand_type = [&](int64_t operand_num) -> absl::StatusOr<Type*> {
    if (operand_num < 0 || operand_num >= node->operand_count()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Operand number %d out of bounds for node %s with %d operands.",
          operand_num, node->GetName(), node->operand_count()));
    }
    return node->operand(operand_num)->GetType();
  };
  switch (factor.source()) {
    case estimator_model::EstimatorFactor::RESULT_BIT_COUNT:
      return node->GetType()->GetFlatBitCount();
    case estimator_model::EstimatorFactor::OPERAND_BIT_COUNT: {
      XLS_ASSIGN_OR_RETURN(Type * operand_type,
                           get_operand_type(factor.operand_number()));
      return operand_type->GetFlatBitCount();
    }
    case estimator_model::EstimatorFactor::OPERAND_COUNT:
      return node->operand_count();
    case estimator_model::EstimatorFactor::OPERAND_ELEMENT_COUNT: {
      XLS_ASSIGN_OR_RETURN(Type * operand_type,
                           get_operand_type(factor.operand_number()));
      if (!operand_type->IsArray()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Operand %d of node %s has non-array type %s for "
                            "OPERAND_ELEMENT_COUNT.",
                            factor.operand_number(), node->GetName(),
                            operand_type->ToString()));
      }
      return operand_type->AsArrayOrDie()->size();
    }
    case estimator_model::EstimatorFactor::OPERAND_ELEMENT_BIT_COUNT: {
      XLS_ASSIGN_OR_RETURN(Type * operand_type,
                           get_operand_type(factor.operand_number()));
      if (!operand_type->IsArray()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Operand %d of node %s has non-array type %s for "
                            "OPERAND_ELEMENT_BIT_COUNT.",
                            factor.operand_number(), node->GetName(),
                            operand_type->ToString()));
      }
      return operand_type->AsArrayOrDie()->element_type()->GetFlatBitCount();
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported EstimatorFactor source: ",
                       static_cast<int>(factor.source())));
  }
}

absl::StatusOr<double> EvalEstimatorExpression(
    const estimator_model::EstimatorExpression& expr, Node* node) {
  switch (expr.expression_type_case()) {
    case estimator_model::EstimatorExpression::kBinOp: {
      XLS_ASSIGN_OR_RETURN(
          double lhs, EvalEstimatorExpression(expr.lhs_expression(), node));
      XLS_ASSIGN_OR_RETURN(
          double rhs, EvalEstimatorExpression(expr.rhs_expression(), node));
      switch (expr.bin_op()) {
        case estimator_model::EstimatorExpression::ADD:
          return lhs + rhs;
        case estimator_model::EstimatorExpression::DIVIDE:
          return lhs / (rhs < 1.0 ? 1.0 : rhs);
        case estimator_model::EstimatorExpression::MAX:
          return std::max(lhs, rhs);
        case estimator_model::EstimatorExpression::MIN:
          return std::min(lhs, rhs);
        case estimator_model::EstimatorExpression::MULTIPLY:
          return lhs * rhs;
        case estimator_model::EstimatorExpression::POWER:
          return std::pow(lhs, rhs);
        case estimator_model::EstimatorExpression::SUB:
          return lhs - rhs;
        default:
          return absl::InvalidArgumentError(
              absl::StrCat("Unsupported EstimatorExpression BinaryOperation: ",
                           static_cast<int>(expr.bin_op())));
      }
    }
    case estimator_model::EstimatorExpression::kFactor: {
      XLS_ASSIGN_OR_RETURN(int64_t f, EvalEstimatorFactor(expr.factor(), node));
      return static_cast<double>(static_cast<float>(f));
    }
    case estimator_model::EstimatorExpression::kConstant:
      return static_cast<double>(static_cast<float>(expr.constant()));
    default:
      return absl::InvalidArgumentError(
          "EstimatorExpression has unset expression_type.");
  }
}

absl::StatusOr<bool> MatchesSpecialization(
    const estimator_model::CompiledOpModel::Specialization& spec, Node* node) {
  switch (spec.kind()) {
    case estimator_model::SpecializationKind::OPERANDS_IDENTICAL:
      return std::all_of(node->operands().begin(), node->operands().end(),
                         [&](Node* n) { return n == node->operand(0); });
    case estimator_model::SpecializationKind::HAS_LITERAL_OPERAND: {
      if (spec.has_details() && spec.details().has_literal_operand_details()) {
        const auto& lit_details = spec.details().literal_operand_details();
        if (!lit_details.required_literal_operand().empty()) {
          for (int64_t req_idx : lit_details.required_literal_operand()) {
            if (req_idx < 0 || req_idx >= node->operand_count() ||
                !node->operand(req_idx)->Is<Literal>()) {
              return false;
            }
          }
        } else {
          if (!std::any_of(node->operands().begin(), node->operands().end(),
                           [](Node* n) { return n->Is<Literal>(); })) {
            return false;
          }
        }
        if (!lit_details.allowed_nonliteral_operand().empty()) {
          for (int64_t i = 0; i < node->operand_count(); ++i) {
            if (!node->operand(i)->Is<Literal>()) {
              bool allowed = std::any_of(
                  lit_details.allowed_nonliteral_operand().begin(),
                  lit_details.allowed_nonliteral_operand().end(),
                  [i](int64_t allowed_idx) { return i == allowed_idx; });
              if (!allowed) {
                return false;
              }
            }
          }
        }
        return true;
      }
      return std::any_of(node->operands().begin(), node->operands().end(),
                         [](Node* n) { return n->Is<Literal>(); });
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported SpecializationKind: ", static_cast<int>(spec.kind())));
  }
}

}  // namespace

absl::StatusOr<const estimator_model::CompiledEstimator*> SelectEstimator(
    const estimator_model::CompiledOpModel& op_model, Node* node) {
  for (const auto& spec : op_model.specializations()) {
    XLS_ASSIGN_OR_RETURN(bool matches, MatchesSpecialization(spec, node));
    if (matches) {
      return &spec.estimator();
    }
  }
  return &op_model.estimator();
}

absl::StatusOr<double> EvalRegressionEstimator(
    const estimator_model::CompiledEstimator::Regression& reg, Node* node) {
  double result = reg.constant();
  for (const auto& term : reg.terms()) {
    XLS_ASSIGN_OR_RETURN(double e,
                         EvalEstimatorExpression(term.expression(), node));
    result += term.linear() * e;
    result += term.log2() * std::log2(std::max(1.0, e));
  }
  return result;
}

absl::StatusOr<double> EvalAreaRegressionEstimator(
    const estimator_model::CompiledEstimator::AreaRegression& reg, Node* node) {
  double result = reg.constant();
  for (const auto& term : reg.terms()) {
    XLS_ASSIGN_OR_RETURN(double e,
                         EvalEstimatorExpression(term.expression(), node));
    double log_e = std::log2(std::max(1.0, e));
    result += term.square() * e * e;
    result += term.linear_log2() * e * log_e;
    result += term.linear() * e;
    result += term.log2() * log_e;
  }
  return result;
}

absl::StatusOr<double> EvalBoundingBoxEstimator(
    const estimator_model::CompiledEstimator::BoundingBox& bbox, Node* node,
    std::string_view metric_name) {
  for (const auto& point : bbox.points()) {
    bool matches = true;
    for (const auto& bound : point.bounds()) {
      XLS_ASSIGN_OR_RETURN(int64_t val,
                           EvalEstimatorFactor(bound.factor(), node));
      if (val > bound.max_value()) {
        matches = false;
        break;
      }
    }
    if (matches) {
      switch (point.result_case()) {
        case estimator_model::CompiledEstimator::BoundingBox::Point::kDelayPs:
          return static_cast<double>(point.delay_ps());
        case estimator_model::CompiledEstimator::BoundingBox::Point::
            kAreaSqMicrons:
          return point.area_sq_microns();
        default:
          return absl::InvalidArgumentError(
              "BoundingBox point is missing a result.");
      }
    }
  }
  return absl::UnimplementedError(
      absl::StrCat("Unhandled node for ", metric_name,
                   " estimation: ", node->ToStringWithOperandTypes()));
}

}  // namespace xls
