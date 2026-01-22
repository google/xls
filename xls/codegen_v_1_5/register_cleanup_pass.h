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

#ifndef XLS_CODEGEN_V_1_5_REGISTER_CLEANUP_PASS_H_
#define XLS_CODEGEN_V_1_5_REGISTER_CLEANUP_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"

namespace xls::codegen {

// A pass which cleans up registers and register writes in the block.
//
// Specific cleanup actions performed:
// 1. Removes load-enable signals from register writes if the load-enable is
//    statically determined to be always one (true).
// 2. Removes register writes if the load-enable is statically determined to
//    be always zero (false).
// 3. Replaces registers which are never written to, or only ever have values
//    identical to their reset value written to them, with literals.
// 4. Removes registers which do not transitively drive any output port.
//
// This pass is intended to run late in the codegen pipeline after flow control
// and pipeline registers have been inserted.
class RegisterCleanupPass : public BlockConversionPass {
 public:
  RegisterCleanupPass()
      : BlockConversionPass("register_cleanup",
                            "Remove dead registers & unused load-enable bits") {
  }

 protected:
  absl::StatusOr<bool> RemoveTrivialLoadEnables(
      Block* block, QueryEngine& query_engine) const;
  absl::StatusOr<bool> RemoveImpossibleWrites(Block* block,
                                              QueryEngine& query_engine) const;
  absl::StatusOr<bool> RemoveUnreadRegisters(Block* block,
                                             QueryEngine& query_engine) const;

  absl::StatusOr<bool> RunInternal(Package* package,
                                   const BlockConversionPassOptions& options,
                                   PassResults* results) const override;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_REGISTER_CLEANUP_PASS_H_
