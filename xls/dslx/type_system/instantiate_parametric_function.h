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

#ifndef XLS_DSLX_TYPE_SYSTEM_INSTANTIATE_PARAMETRIC_FUNCTION_H_
#define XLS_DSLX_TYPE_SYSTEM_INSTANTIATE_PARAMETRIC_FUNCTION_H_

#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"

namespace xls::dslx {

// Sets up parametric args (based on actual args) and explicit bindings (based
// on explicitly given parametric args) and runs the parametric instantiator.
absl::StatusOr<TypeAndParametricEnv> InstantiateParametricFunction(
    DeduceCtx* ctx, DeduceCtx* parent_ctx, const Invocation* invocation,
    Function& callee_fn, const FunctionType& fn_type,
    const std::vector<InstantiateArg>& instantiate_args);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_INSTANTIATE_PARAMETRIC_FUNCTION_H_
