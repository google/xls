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

#ifndef XLS_SOLVERS_SYMEX_CONCOLIC_INPUT_SPEC_H_
#define XLS_SOLVERS_SYMEX_CONCOLIC_INPUT_SPEC_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/value.h"

namespace xls::solvers::symex {

// Specification for concrete parameter inputs in concolic / symbolic
// execution. Parameters bound in this specification are treated as known
// concrete constants during path exploration rather than unconstrained
// symbolic variables.
class ConcolicInputSpec {
 public:
  ConcolicInputSpec() = default;

  // Binds a parameter by name to a concrete XLS Value. Returns *this for
  // method chaining.
  ConcolicInputSpec& BindParam(std::string_view name, Value value);

  bool empty() const { return param_values_.empty(); }
  int64_t size() const { return param_values_.size(); }
  bool HasParam(std::string_view name) const;
  std::optional<Value> GetParam(std::string_view name) const;

  const absl::flat_hash_map<std::string, Value>& param_values() const {
    return param_values_;
  }

 private:
  absl::flat_hash_map<std::string, Value> param_values_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_CONCOLIC_INPUT_SPEC_H_
