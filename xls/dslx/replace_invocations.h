// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_REPLACE_INVOCATIONS_H_
#define XLS_DSLX_REPLACE_INVOCATIONS_H_

#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"  // For TypecheckedModule
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {

// Describes a rule for rewriting invocations irrespective of caller.
// Used by the bulk API where a set of callers is provided separately.
struct InvocationRewriteRule {
  // Replace invocations whose resolved callee is exactly this function.
  const Function* from_callee = nullptr;

  // The function that should be used as the new callee.
  const Function* to_callee = nullptr;

  // Optional filter: only invocations whose callee-side ParametricEnv equals
  // this value will be replaced. If not set, matches all instantiations.
  std::optional<ParametricEnv> match_callee_env;

  // Optional explicit env for the replacement callee. If not set, retains the
  // original invocation's explicit parametrics. If set to an empty env, emits
  // no explicit parametrics (rely on deduction).
  std::optional<ParametricEnv> to_callee_env;
};

// Returns a cloned module where invocations inside `callers` that resolve to
// a rule's `from_callee` (and optionally match the specified callee-side
// parametric environment) have their callee expression replaced so they invoke
// the corresponding `to_callee` instead.
//
// Bulk variant: applies multiple rewrite rules across a set of caller
// functions. The first rule that matches an invocation is applied. The
// `type_info` is consulted to resolve parametric binding information for each
// invocation.
absl::StatusOr<TypecheckedModule> ReplaceInvocationsInModule(
    const TypecheckedModule& tm, absl::Span<const Function* const> callers,
    absl::Span<const InvocationRewriteRule> rules, ImportData& import_data,
    std::string_view install_subject);

// Non-bulk convenience overload that delegates to the bulk variant.
absl::StatusOr<TypecheckedModule> ReplaceInvocationsInModule(
    const TypecheckedModule& tm, const Function* caller,
    const InvocationRewriteRule& rule, ImportData& import_data,
    std::string_view install_subject);

}  // namespace xls::dslx

#endif  // XLS_DSLX_REPLACE_INVOCATIONS_H_
