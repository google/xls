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

#include "xls/dslx/type_system/scoped_fn_stack_entry.h"

#include <cstdint>

#include "absl/log/check.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_env.h"

namespace xls::dslx {

ScopedFnStackEntry::ScopedFnStackEntry(DeduceCtx* ctx, Module* module)
    : ctx_(ctx), depth_before_(ctx->fn_stack().size()), expect_popped_(false) {
  ctx->AddFnStackEntry(FnStackEntry::MakeTop(module));
}

ScopedFnStackEntry::ScopedFnStackEntry(Function& fn, DeduceCtx* ctx,
                                       WithinProc within_proc,
                                       bool expect_popped)
    : ctx_(ctx),
      depth_before_(ctx->fn_stack().size()),
      expect_popped_(expect_popped) {
  ctx->AddFnStackEntry(FnStackEntry::Make(fn, ParametricEnv(), within_proc));
}

void ScopedFnStackEntry::Finish() {
  if (expect_popped_) {
    CHECK_EQ(ctx_->fn_stack().size(), depth_before_);
  } else {
    int64_t depth_after_push = depth_before_ + 1;
    CHECK_EQ(ctx_->fn_stack().size(), depth_after_push);
    ctx_->PopFnStackEntry();
  }
}

}  // namespace xls::dslx
