// Copyright 2026 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_LAMBDA_REWRITER_H_
#define XLS_DSLX_FRONTEND_LAMBDA_REWRITER_H_

#include "absl/status/status.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// Converts all lambdas into structs and impls. For example,
//
//   fn add_two(arr: u32[5]) -> u32[5] {
//     let x = u32:2;
//     map(arr, |i: u32| -> u32 { x + i })
//   }
//
// becomes:
//
//   struct lambda_capture { x: u32 }
//
//   impl lambda_capture {
//     fn call(self) -> u32 { self.x + i }
//   }
//
//   fn add_two(arr: u32[5]) -> u32[5] {
//     let x = u32:2;
//     map(arr, lambda_capture{x: x}.call)
//   }
absl::Status RewriteLambdas(Module& module, const FileTable& file_table);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_LAMBDA_REWRITER_H_
