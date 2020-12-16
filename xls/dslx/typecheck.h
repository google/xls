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

#ifndef XLS_DSLX_TYPECHECK_H_
#define XLS_DSLX_TYPECHECK_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/python/cpp_ast.h"

namespace xls::dslx {

// Checks the function's parametrics' and arguments' types.
//
// Returns the sequence of parameter types.
absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>> CheckFunctionParams(
    Function* f, DeduceCtx* ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPECHECK_H_
