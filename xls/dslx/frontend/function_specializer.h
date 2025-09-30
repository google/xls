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

#ifndef XLS_DSLX_FRONTEND_FUNCTION_SPECIALIZER_H_
#define XLS_DSLX_FRONTEND_FUNCTION_SPECIALIZER_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {

// Creates a non-parametric clone of `source_function` using the bindings
// captured in `param_env` and inserts it into the owning module under
// `specialized_name`.
//
// The returned function is appended to the owning module. Each reference to a
// parametric binding is replaced with a literal value derived from the
// environment.
absl::StatusOr<Function*> InsertFunctionSpecialization(
    Function* source_function, const ParametricEnv& param_env,
    std::string_view specialized_name);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_FUNCTION_SPECIALIZER_H_
