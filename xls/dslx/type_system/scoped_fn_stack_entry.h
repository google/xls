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

#ifndef XLS_DSLX_TYPE_SYSTEM_SCOPED_FN_STACK_ENTRY_H_
#define XLS_DSLX_TYPE_SYSTEM_SCOPED_FN_STACK_ENTRY_H_

#include <cstdint>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/deduce_ctx.h"

namespace xls::dslx {

// Helper type to place on the stack when we intend to pop off a FnStackEntry
// when done, or expect a caller to pop it off for us. That is, this helps us
// check fn_stack() invariants are as expected.
//
// WARNING: this not a true RAII guard, but rather a helper with
// constructor/Finish() providing the book-ends on account of Status-returns
// violating the anticipated invariants, the call to Finish() should be used on
// the "happy paths".
class ScopedFnStackEntry {
 public:
  ScopedFnStackEntry(DeduceCtx* ctx, Module* module);

  // Args:
  //  expect_popped: Indicates that we expect, on the call to Finish(),
  //    that the entry will have already been popped. Generally this is `false`
  //    since we expect the entry to be on the top of the fn stack in the
  //    call to Finish(), in which case we automatically pop it.
  ScopedFnStackEntry(Function& fn, DeduceCtx* ctx, WithinProc within_proc,
                     bool expect_popped = false);

  // Called when we close out a scope. We can't use this object as a scope
  // guard easily because we want to be able to detect if we return an
  // absl::Status early, so we have to manually put end-of-scope calls at usage
  // points.
  void Finish();

 private:
  DeduceCtx* ctx_;
  int64_t depth_before_;
  bool expect_popped_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_SCOPED_FN_STACK_ENTRY_H_
