// Copyright 2025 The XLS Authors
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

#ifndef XLS_CONTRIB_MLIR_UTIL_IDENTIFIER_H_
#define XLS_CONTRIB_MLIR_UTIL_IDENTIFIER_H_

#include <string>

#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir::xls {

std::string CleanupIdentifier(StringRef name);

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_UTIL_IDENTIFIER_H_
