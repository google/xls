// Copyright 2022 The XLS Authors
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

#ifndef XLS_CODEGEN_BDD_IO_ANALYSIS_H_
#define XLS_CODEGEN_BDD_IO_ANALYSIS_H_

#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// Determines if streaming outputs are mutually exclusive.
//
// TODO(tedhong): 2022-02-09 Add analysis of I/O dependencies
// TODO(tedhong): 2022-02-09 Add additional exclusivity analysis
absl::StatusOr<bool> AreStreamingOutputsMutuallyExclusive(FunctionBase* f);

}  // namespace xls

#endif  // XLS_CODEGEN_BDD_IO_ANALYSIS_H_
