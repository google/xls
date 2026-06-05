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

// Create a standard pipeline of passes. This pipeline should
// be used in the main driver as well as in testing.

#include "xls/codegen_v_1_5/block_conversion_pass_pipeline.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/codegen_pass_registry.h"
#include "xls/common/status/status_macros.h"

namespace xls::codegen {

absl::StatusOr<std::unique_ptr<BlockConversionPass>>
CreateBlockConversionPassPipeline() {
  XLS_ASSIGN_OR_RETURN(auto* generator,
                       GetCodegenPassRegistry().Generator("default_pipeline"));
  return generator->Generate();
}

}  // namespace xls::codegen
