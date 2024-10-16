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

#ifndef GDM_HW_MLIR_XLS_TRANSFORMS_XLS_LOWER_H_
#define GDM_HW_MLIR_XLS_TRANSFORMS_XLS_LOWER_H_

#include "llvm/include/llvm/Support/CommandLine.h"
#include "mlir/include/mlir/Pass/PassManager.h"
#include "mlir/include/mlir/Pass/PassOptions.h"

namespace mlir::xls {

struct XlsLowerPassPipelineOptions
    : public PassPipelineOptions<XlsLowerPassPipelineOptions> {
  // Whether to instantiate eprocs after proc elaboration.
  // If true, xls.instantiate_eproc ops are turned into "real" eprocs and the
  // output is suitable for translation to XLS.
  // If false, xls.instantiate_eproc ops are left as-is and the output needs
  // postprocessing before sending to XLS as eproc channels are not yet
  // instantiated / linked.
  PassOptions::Option<bool> instantiate_eprocs{
      *this, "instantiate-eprocs",
      llvm::cl::desc(
          "If true, xls.instantiate_eproc ops are turned into 'real' eprocs "
          "and the output is suitable for translation to XLS"),
      llvm::cl::init(true)};
};

// A Pass pipeline that lowers to a form that can be translated to XLS.
void XlsLowerPassPipeline(OpPassManager& pm,
                          const XlsLowerPassPipelineOptions& options = {});

// Registers a pass pipeline that lowers to a form that can be translated to
// XLS.
void RegisterXlsLowerPassPipeline();

}  // namespace mlir::xls

#endif  // GDM_HW_MLIR_XLS_TRANSFORMS_XLS_LOWER_H_
