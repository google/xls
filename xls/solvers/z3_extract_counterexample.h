// Copyright 2024 The XLS Authors
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

#ifndef XLS_SOLVERS_Z3_EXTRACT_COUNTEREXAMPLE_H_
#define XLS_SOLVERS_Z3_EXTRACT_COUNTEREXAMPLE_H_

#include <string>
#include <string_view>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls::solvers::z3 {

struct IrParamSpec {
  std::string name;
  absl::Nonnull<const Type*> type;
};

// Given a message that Z3 produces when finding a counterexample, attempts to
// extract the model as an XLS value representation.
//
// Returns a map from parameter name to the value that was parsed for that
// value. Note that if not all parameters are obsered in the counterexample
// message text, an error is returned.
absl::StatusOr<absl::flat_hash_map<std::string, Value>> ExtractCounterexample(
    std::string_view message, absl::Span<const IrParamSpec> params);

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_EXTRACT_COUNTEREXAMPLE_H_
