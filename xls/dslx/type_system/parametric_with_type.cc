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

#include "xls/dslx/type_system/parametric_with_type.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_format.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

ParametricWithType::ParametricWithType(const ParametricBinding& binding,
                                       std::unique_ptr<Type> type)
    : binding_(binding), type_(std::move(type)) {}

std::string ParametricWithType::ToString() const {
  if (expr() == nullptr) {
    return absl::StrFormat("%s: %s", identifier(), type().ToString());
  }
  return absl::StrFormat("%s: %s = %s", identifier(), type().ToString(),
                         expr()->ToString());
}

}  // namespace xls::dslx
