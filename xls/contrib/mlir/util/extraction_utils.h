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

#ifndef XLS_CONTRIB_MLIR_UTIL_EXTRACTION_UTILS_H_
#define XLS_CONTRIB_MLIR_UTIL_EXTRACTION_UTILS_H_

#include "mlir/include/mlir/Analysis/CallGraph.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

// Extracts the given operation as a top-level module.
//
// The returned module contains all functions and symbols that are transitively
// called.
//
// The input module is not modified.
//
// The input CallGraph is not invalidated.
ModuleOp extractAsTopLevelModule(EprocOp op, mlir::CallGraph& callGraph);
ModuleOp extractAsTopLevelModule(mlir::func::FuncOp op,
                                 mlir::CallGraph& callGraph);

// Registers the test pass for extracting as a top level module.
namespace test {
void registerTestExtractAsTopLevelModulePass();
}

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_UTIL_EXTRACTION_UTILS_H_
