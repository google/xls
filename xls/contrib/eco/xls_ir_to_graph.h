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

#ifndef XLS_ECO_XLS_IR_TO_GRAPH_H_
#define XLS_ECO_XLS_IR_TO_GRAPH_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/contrib/eco/graph.h"
#include "xls/ir/function_base.h"

namespace xls {

absl::StatusOr<XLSGraph> XlsIrToGraph(FunctionBase* function_base);
absl::StatusOr<XLSGraph> ParseIrFileToGraph(std::string_view ir_path);

}  // namespace xls

#endif  // XLS_ECO_XLS_IR_TO_GRAPH_H_
