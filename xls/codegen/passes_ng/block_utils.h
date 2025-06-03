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

#ifndef XLS_CODEGEN_PASSES_NG_BLOCK_UTILS_H_
#define XLS_CODEGEN_PASSES_NG_BLOCK_UTILS_H_

// Common utility functions for modifying and block ir used by codegen ng
// passes.

#include <string_view>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"

namespace xls::verilog {

// Inserts a buffer after the given source node and before all it uses.  Returns
// the buffer node..
//
// Original:           After
//         src               buf      src
//   x------<|------   x------<|------<|----------
//       |                 |
//   x---+             x---+
//
inline absl::StatusOr<Node*> CreateBufferAfter(std::string_view buf_name,
                                               Node* ABSL_NONNULL src,
                                               SourceInfo loc,
                                               Block* ABSL_NONNULL block) {
  XLS_RET_CHECK(src != nullptr);
  XLS_RET_CHECK(block != nullptr);
  XLS_ASSIGN_OR_RETURN(Node * buf, block->MakeNodeWithName<UnOp>(
                                       loc, src, Op::kIdentity, buf_name));
  XLS_RETURN_IF_ERROR(src->ReplaceUsesWith(buf));
  return buf;
}

// Inserts a buffer before the given source op.  The source op must have only
// one operand.  Returns the buffer node.
//
// Original:           After
//         src               buf      src
//   x------|>------   x------|>------|>----------
//
inline absl::StatusOr<Node*> CreateBufferBefore(std::string_view buf_name,
                                                Node* ABSL_NONNULL src,
                                                SourceInfo loc,
                                                Block* ABSL_NONNULL block) {
  XLS_RET_CHECK(src != nullptr);
  XLS_RET_CHECK(block != nullptr);
  if (src->operand_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("CreateBufferBefore's source node %s must have only "
                        "one operand, got %d",
                        src->GetNameView(), src->operand_count()));
  }

  XLS_ASSIGN_OR_RETURN(Node * buf,
                       block->MakeNodeWithName<UnOp>(loc, src->operand(0),
                                                     Op::kIdentity, buf_name));
  XLS_RETURN_IF_ERROR(src->ReplaceOperandNumber(0, buf));

  return buf;
}

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_BLOCK_UTILS_H_
