// Copyright 2023 The XLS Authors
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

#include "xls/dslx/type_system/parametric_constraint.h"

namespace xls::dslx {

ParametricConstraint::ParametricConstraint(const ParametricBinding& binding,
                                           std::unique_ptr<ConcreteType> type)
    : ParametricConstraint(binding, std::move(type), binding.expr()) {}

ParametricConstraint::ParametricConstraint(const ParametricBinding& binding,
                                           std::unique_ptr<ConcreteType> type,
                                           Expr* expr)
    : binding_(&binding), type_(std::move(type)), expr_(expr) {}

std::string ParametricConstraint::ToString() const {
  if (expr_ == nullptr) {
    return absl::StrFormat("%s: %s", identifier(), type().ToString());
  }
  return absl::StrFormat("%s: %s = %s", identifier(), type().ToString(),
                         expr()->ToString());
}

}  // namespace xls::dslx
