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

#include "xls/dslx/deduce_ctx.h"

#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/variant.h"
#include "xls/common/string_to_int.h"

namespace xls::dslx {

std::string FnStackEntry::ToReprString() const {
  return absl::StrFormat("FnStackEntry{\"%s\", %s}", name_,
                         symbolic_bindings_.ToString());
}

DeduceCtx::DeduceCtx(TypeInfo* type_info, Module* module,
                     DeduceFn deduce_function,
                     TypecheckFunctionFn typecheck_function,
                     TypecheckModuleFn typecheck_module,
                     TypecheckInvocationFn typecheck_invocation,
                     ImportData* import_data, WarningCollector* warnings)
    : type_info_(type_info),
      module_(module),
      deduce_function_(std::move(XLS_DIE_IF_NULL(deduce_function))),
      typecheck_function_(std::move(typecheck_function)),
      typecheck_module_(std::move(typecheck_module)),
      typecheck_invocation_(std::move(typecheck_invocation)),
      import_data_(import_data),
      warnings_(warnings) {}

// Helper that converts the symbolic bindings to a parametric expression
// environment (for parametric evaluation).
ParametricExpression::Env ToParametricEnv(
    const SymbolicBindings& symbolic_bindings) {
  ParametricExpression::Env env;
  for (const SymbolicBinding& binding : symbolic_bindings.bindings()) {
    env[binding.identifier] = binding.value;
  }
  return env;
}

}  // namespace xls::dslx
