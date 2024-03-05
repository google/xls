// Copyright 2021 The XLS Authors
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

#ifndef XLS_DSLX_MANGLE_H_
#define XLS_DSLX_MANGLE_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {

enum class CallingConvention : uint8_t {
  // The IR converted parameters are identical to the DSL parameters in their
  // type, number, and name.
  kTypical,

  // DSL functions that have `fail!()` operations inside are IR converted to
  // automatically take a `(seq: token, activated: bool)` as initial parameters,
  // so that caller contexts can say whether the function is activated (such
  // that an assertion should actually cause a failure when the predicate is
  // false).
  kImplicitToken,

  // The IR calling convention for a proc next function.
  kProcNext,
};

// Returns the mangled name of function with the given parametric bindings.
absl::StatusOr<std::string> MangleDslxName(
    std::string_view module_name, std::string_view function_name,
    CallingConvention convention,
    const absl::btree_set<std::string>& free_keys = {},
    const ParametricEnv* parametric_env = nullptr);
}  // namespace xls::dslx

#endif  // XLS_DSLX_MANGLE_H_
