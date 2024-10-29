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

#ifndef XLS_CONTRIB_MLIR_TOOLS_XLS_TRANSLATE_XLS_STITCH_H_
#define XLS_CONTRIB_MLIR_TOOLS_XLS_TRANSLATE_XLS_STITCH_H_

#include <string>

#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
class Operation;
}  // namespace mlir

namespace llvm {
class raw_ostream;
}  // namespace llvm

namespace mlir::xls {

struct XlsStitchOptions {
  // The name of the clock signal.
  std::string clock_signal_name = "clk";

  // The name of the reset signal.
  std::string reset_signal_name = "rst";

  // The suffix of a channels's data port.
  std::string data_port_suffix = "";

  // The suffix of a channel's ready port.
  std::string ready_port_suffix = "_rdy";

  // The suffix of a channel's valid port.
  std::string valid_port_suffix = "_vld";
};

// Generates a a top Verilog module to stitch together the given module. The
// stitching is driven by the `xls.instantiate_eproc` ops in the module.
LogicalResult XlsStitch(ModuleOp op, llvm::raw_ostream& output,
                        XlsStitchOptions options = {});

}  // namespace mlir::xls

#endif  // XLS_CONTRIB_MLIR_TOOLS_XLS_TRANSLATE_XLS_STITCH_H_
