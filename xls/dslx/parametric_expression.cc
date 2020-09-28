// Copyright 2020 Google LLC
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

#include "xls/dslx/parametric_expression.h"

namespace xls::dslx {

/* static */ ParametricExpression::Evaluated ParametricExpression::ToEvaluated(
    const EnvValue& value) {
  if (absl::holds_alternative<int64>(value)) {
    return absl::get<int64>(value);
  }
  return absl::get<const ParametricExpression*>(value)->Clone();
}

/* static */ ParametricExpression::EnvValue ParametricExpression::ToEnvValue(
    const Evaluated& v) {
  if (absl::holds_alternative<int64>(v)) {
    return absl::get<int64>(v);
  }
  return absl::get<std::unique_ptr<ParametricExpression>>(v).get();
}

std::unique_ptr<ParametricExpression> ParametricExpression::ToOwned(
    const std::variant<const ParametricExpression*, int64>& operand) {
  if (absl::holds_alternative<int64>(operand)) {
    return absl::make_unique<ParametricConstant>(absl::get<int64>(operand));
  }
  return absl::get<const ParametricExpression*>(operand)->Clone();
}

std::unique_ptr<ParametricExpression> ParametricExpression::Add(
    const EnvValue& lhs, const EnvValue& rhs) {
  if (absl::holds_alternative<int64>(lhs) &&
      absl::holds_alternative<int64>(rhs)) {
    return absl::make_unique<ParametricConstant>(absl::get<int64>(lhs) +
                                                 absl::get<int64>(rhs));
  }
  return absl::make_unique<ParametricAdd>(ToOwned(lhs), ToOwned(rhs));
}
std::unique_ptr<ParametricExpression> ParametricExpression::Mul(
    const EnvValue& lhs, const EnvValue& rhs) {
  if (absl::holds_alternative<int64>(lhs) &&
      absl::holds_alternative<int64>(rhs)) {
    return absl::make_unique<ParametricConstant>(absl::get<int64>(lhs) *
                                                 absl::get<int64>(rhs));
  }
  return absl::make_unique<ParametricMul>(ToOwned(lhs), ToOwned(rhs));
}
ParametricExpression::Evaluated ParametricExpression::TryUnwrapConstant(
    std::unique_ptr<ParametricExpression> e) {
  if (auto* c = dynamic_cast<ParametricConstant*>(e.get())) {
    return c->value();
  }
  return std::move(e);
}

}  // namespace xls::dslx
