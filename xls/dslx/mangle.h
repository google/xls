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

#include "xls/dslx/ast.h"
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {

enum class CallingConvention {
  // The IR converted parameters are identical to the DSL parameters in their
  // type, number, and name.
  kTypical,

  // DSL functions that have `fail!()` operations inside are IR converted to
  // automatically take a `(seq: token, activated: bool)` as initial parameters,
  // so that caller contexts can say whether the function is activated (such
  // that an assertion should actually cause a failure when the predicate is
  // false).
  kImplicitToken,
};

// Returns the mangled name of function with the given parametric bindings.
absl::StatusOr<std::string> MangleDslxName(
    absl::string_view module_name, absl::string_view function_name,
    CallingConvention convention,
    const absl::btree_set<std::string>& free_keys = {},
    const SymbolicBindings* symbolic_bindings = nullptr);
}  // namespace xls::dslx

#endif  // XLS_DSLX_MANGLE_H_
