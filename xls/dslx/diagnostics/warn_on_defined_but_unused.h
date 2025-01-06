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

#ifndef XLS_DSLX_DIAGNOSTICS_WARN_ON_DEFINED_BUT_UNUSED_H_
#define XLS_DSLX_DIAGNOSTICS_WARN_ON_DEFINED_BUT_UNUSED_H_

#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/deduce_ctx.h"

namespace xls::dslx {

// Emit warnings (into the WarningCollector on "ctx") for any bindings
// considered defined-but-unused variables present in function "f".
absl::Status WarnOnDefinedButUnused(Function& f, DeduceCtx* ctx);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DIAGNOSTICS_WARN_ON_DEFINED_BUT_UNUSED_H_
