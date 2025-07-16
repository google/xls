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

#ifndef GDM_HW_MLIR_XLS_TOOLS_XLS_TRANSLATE_XLS_TRANSLATE_TO_MLIR_H_
#define GDM_HW_MLIR_XLS_TOOLS_XLS_TRANSLATE_XLS_TRANSLATE_TO_MLIR_H_

#include "llvm/include/llvm/Support/SourceMgr.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "xls/ir/package.h"

namespace xls {
class Package;
}  // namespace xls

namespace mlir::xls {

OwningOpRef<Operation*> XlsToMlirXlsTranslate(llvm::SourceMgr& mgr,
                                              MLIRContext* ctx);

OwningOpRef<Operation*> XlsToMlirXlsTranslate(const ::xls::Package& package,
                                              MLIRContext* ctx);

}  // namespace mlir::xls

#endif  // GDM_HW_MLIR_XLS_TOOLS_XLS_TRANSLATE_XLS_TRANSLATE_TO_MLIR_H_
