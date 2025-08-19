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

#ifndef XLS_CODEGEN_CODEGEN_RESULT_H_
#define XLS_CODEGEN_CODEGEN_RESULT_H_

#include <string>

#include "xls/codegen/codegen_residual_data.pb.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/verilog_line_map.pb.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/passes/pass_metrics.pb.h"

namespace xls::verilog {

// Data structure gathering together all the artifacts created by codegen.
struct CodegenResult {
  std::string verilog_text;
  VerilogLineMap verilog_line_map;
  ModuleSignature signature;
  XlsMetricsProto block_metrics;
  CodegenResidualData residual_data;
  PassPipelineMetricsProto pass_pipeline_metrics;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_RESULT_H_
