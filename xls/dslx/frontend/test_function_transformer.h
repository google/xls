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

#ifndef XLS_DSLX_FRONTEND_TEST_FUNCTION_TRANSFORMER_H_
#define XLS_DSLX_FRONTEND_TEST_FUNCTION_TRANSFORMER_H_

#include <memory>

#include "absl/status/status.h"
#include "xls/dslx/frontend/module.h"

namespace xls {
namespace dslx {

// Transforms test functions that spawn procs into TestProcs in the cloned
// module. The original functions will be removed from the cloned module.
absl::StatusOr<std::unique_ptr<Module>> TransformTestFunctions(
    const Module& module);

}  // namespace dslx
}  // namespace xls

#endif  // XLS_DSLX_FRONTEND_TEST_FUNCTION_TRANSFORMER_H_
