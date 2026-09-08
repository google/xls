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

#include "xls/solvers/symex/concolic_input_spec.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "xls/ir/value.h"

namespace xls::solvers::symex {

ConcolicInputSpec& ConcolicInputSpec::BindParam(std::string_view name,
                                                Value value) {
  param_values_[std::string(name)] = std::move(value);
  return *this;
}

bool ConcolicInputSpec::HasParam(std::string_view name) const {
  return param_values_.contains(name);
}

std::optional<Value> ConcolicInputSpec::GetParam(std::string_view name) const {
  auto it = param_values_.find(name);
  if (it != param_values_.end()) {
    return it->second;
  }
  return std::nullopt;
}

}  // namespace xls::solvers::symex
