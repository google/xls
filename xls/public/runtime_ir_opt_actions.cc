// Copyright 2021 The XLS Authors
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

#include "xls/public/runtime_ir_opt_actions.h"

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/tools/opt.h"

namespace xls {

absl::StatusOr<std::string> OptimizeIr(std::string_view ir,
                                       std::string_view top) {
  const tools::OptOptions options = {
      .opt_level = xls::kMaxOptLevel,
      .top = std::string(top),
  };
  return tools::OptimizeIrForTop(ir, options);
}

}  // namespace xls
