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

// Validates type annotations on parameters / return type of `f` are consistent.
//
// Returns a XlsTypeErrorStatus when the return type deduced is inconsistent
// with the return type annotation on `f`.
absl::Status CheckFunction(Function* f, DeduceCtx* ctx);

// Validates a test (body) within a module.
absl::Status CheckTest(Test* t, DeduceCtx* ctx);

// Instantiates a builtin parametric invocation; e.g. `update()`.
absl::StatusOr<NameDef*> InstantiateBuiltinParametric(
    BuiltinNameDef* builtin_name, Invocation* invocation, DeduceCtx* ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPECHECK_H_
