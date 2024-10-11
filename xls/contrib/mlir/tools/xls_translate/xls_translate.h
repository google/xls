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

#ifndef GDM_HW_MLIR_XLS_TOOLS_XLS_TRANSLATE_XLS_TRANSLATE_H_
#define GDM_HW_MLIR_XLS_TOOLS_XLS_TRANSLATE_XLS_TRANSLATE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
class Operation;
}  // namespace mlir

namespace llvm {
class raw_ostream;
}  // namespace llvm

namespace xls {
class Package;
}  // namespace xls

namespace mlir::xls {

// Caches DSLX translation results. DSLX translation is expensive, so this
// avoids re-translating the same file multiple times across calls to
// MlirXlsToXlsTranslate.
class DslxPackageCache {
 public:
  // Imports `fileName` as a DSLX file.
  absl::StatusOr<std::shared_ptr<const ::xls::Package>> import(
      const std::string& fileName);

 private:
  absl::flat_hash_map<std::string, std::shared_ptr<const ::xls::Package>> cache;
};

struct MlirXlsToXlsTranslateOptions {
  // The name of the main function to translate.
  llvm::StringRef main_function = "";

  // The search path for DSLX files.
  llvm::StringRef dslx_search_path = "";

  // Whether to run XLS's optimizer post translation but before emitting.
  bool optimize_ir = false;

  // Whether to generate Verilog.
  bool generate_verilog = false;

  // Whether to privatize all non-top functions and run SymbolDCE first.
  bool privatize_and_dce_functions = false;

  // Optional cache for DSLX translation results.
  DslxPackageCache* dslx_cache = nullptr;

  // Verilog emission options.

  // The number of pipeline stages to (attempt to) generate. If zero, the
  // combinational generator will be used.
  int pipeline_stages = 0;

  // The delay model to use.
  std::string delay_model = "asap7";

  // The name of the reset signal.
  std::string reset_signal_name = "rst";
};

// Translates an operation with XLS dialect to DSLX.
LogicalResult MlirXlsToXlsTranslate(Operation* op, llvm::raw_ostream& output,
                                    MlirXlsToXlsTranslateOptions options = {});

}  // namespace mlir::xls

#endif  // GDM_HW_MLIR_XLS_TOOLS_XLS_TRANSLATE_XLS_TRANSLATE_H_
