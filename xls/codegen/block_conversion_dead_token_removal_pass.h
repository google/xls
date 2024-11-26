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

#ifndef XLS_CODEGEN_BLOCK_CONVERSION_DEAD_TOKEN_REMOVAL_PASS_H_
#define XLS_CODEGEN_BLOCK_CONVERSION_DEAD_TOKEN_REMOVAL_PASS_H_

#include <string_view>

#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

// Send/receive nodes are not cloned from the proc into the block, but the
// network of tokens connecting these send/receive nodes *is* cloned. This
// pass removes the token operations.
class BlockConversionDeadTokenRemovalPass : public CodegenCompoundPass {
 public:
  static constexpr std::string_view kName =
      "block_conversion_dead_token_removal";
  explicit BlockConversionDeadTokenRemovalPass();
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_BLOCK_CONVERSION_DEAD_TOKEN_REMOVAL_PASS_H
