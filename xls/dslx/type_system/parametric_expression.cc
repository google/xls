// Copyright 2020 The XLS Authors
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

#include "xls/dslx/type_system/parametric_expression.h"

#include <memory>
#include <utility>
#include <variant>

#include "absl/status/statusor.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

/* static */ ParametricExpression::Evaluated ParametricExpression::ToEvaluated(
    const EnvValue& value) {
  if (std::holds_alternative<InterpValue>(value)) {
    return std::get<InterpValue>(value);
  }
  return std::get<const ParametricExpression*>(value)->Clone();
}

/* static */ ParametricExpression::EnvValue ParametricExpression::ToEnvValue(
    const Evaluated& v) {
  if (std::holds_alternative<InterpValue>(v)) {
    return std::get<InterpValue>(v);
  }
  return std::get<std::unique_ptr<ParametricExpression>>(v).get();
}

std::unique_ptr<ParametricExpression> ParametricExpression::ToOwned(
    const std::variant<const ParametricExpression*, InterpValue>& operand) {
  if (std::holds_alternative<InterpValue>(operand)) {
    return std::make_unique<ParametricConstant>(std::get<InterpValue>(operand));
  }
  return std::get<const ParametricExpression*>(operand)->Clone();
}

std::unique_ptr<ParametricExpression> ParametricExpression::Add(
    const EnvValue& lhs, const EnvValue& rhs) {
  if (std::holds_alternative<InterpValue>(lhs) &&
      std::holds_alternative<InterpValue>(rhs)) {
    return std::make_unique<ParametricConstant>(
        std::get<InterpValue>(lhs).Add(std::get<InterpValue>(rhs)).value());
  }
  return std::make_unique<ParametricAdd>(ToOwned(lhs), ToOwned(rhs));
}
std::unique_ptr<ParametricExpression> ParametricExpression::Mul(
    const EnvValue& lhs, const EnvValue& rhs) {
  if (std::holds_alternative<InterpValue>(lhs) &&
      std::holds_alternative<InterpValue>(rhs)) {
    return std::make_unique<ParametricConstant>(
        std::get<InterpValue>(lhs).Mul(std::get<InterpValue>(rhs)).value());
  }
  return std::make_unique<ParametricMul>(ToOwned(lhs), ToOwned(rhs));
}
std::unique_ptr<ParametricExpression> ParametricExpression::Div(
    const EnvValue& lhs, const EnvValue& rhs) {
  if (std::holds_alternative<InterpValue>(lhs) &&
      std::holds_alternative<InterpValue>(rhs)) {
    return std::make_unique<ParametricConstant>(
        std::get<InterpValue>(lhs)
            .FloorDiv(std::get<InterpValue>(rhs))
            .value());
  }
  return std::make_unique<ParametricDiv>(ToOwned(lhs), ToOwned(rhs));
}
std::unique_ptr<ParametricExpression> ParametricExpression::CeilOfLog2(
    const EnvValue& arg) {
  if (std::holds_alternative<InterpValue>(arg)) {
    return std::make_unique<ParametricConstant>(
        std::get<InterpValue>(arg).CeilOfLog2().value());
  }
  return std::make_unique<ParametricWidth>(ToOwned(arg));
}
ParametricExpression::Evaluated ParametricExpression::TryUnwrapConstant(
    std::unique_ptr<ParametricExpression> e) {
  if (auto* c = dynamic_cast<ParametricConstant*>(e.get())) {
    return c->value();
  }
  return std::move(e);
}

}  // namespace xls::dslx
