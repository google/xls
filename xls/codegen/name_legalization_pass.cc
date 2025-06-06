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

#include "xls/codegen/name_legalization_pass.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/vast/verilog_keywords.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

// Add a new copy of register `old_reg` to get a new name. Replace all uses of
// the old register with the new one and remove the old register.
// This effectively renames the `old_reg`.
absl::Status RenameRegister(Block* block, Register* old_reg) {
  std::string_view old_name = old_reg->name();
  XLS_ASSIGN_OR_RETURN(Register * new_reg,
                       block->AddRegister(old_reg->name(), old_reg->type(),
                                          old_reg->reset_value()));
  XLS_RET_CHECK_NE(old_name, new_reg->name());

  XLS_ASSIGN_OR_RETURN(RegisterRead * old_read,
                       block->GetRegisterRead(old_reg));
  XLS_ASSIGN_OR_RETURN(RegisterWrite * old_write,
                       block->GetUniqueRegisterWrite(old_reg));

  XLS_RETURN_IF_ERROR(
      old_read->ReplaceUsesWithNew<RegisterRead>(new_reg).status());
  XLS_RETURN_IF_ERROR(block->RemoveNode(old_read));
  XLS_RETURN_IF_ERROR(old_write
                          ->ReplaceUsesWithNew<RegisterWrite>(
                              old_write->data(), old_write->load_enable(),
                              old_write->reset(), new_reg)
                          .status());
  XLS_RETURN_IF_ERROR(block->RemoveNode(old_write));

  return block->RemoveRegister(old_reg);
}

absl::StatusOr<bool> LegalizeNames(Block* block, bool use_system_verilog) {
  const absl::flat_hash_set<std::string>& sv_keywords = SystemVerilogKeywords();
  const absl::flat_hash_set<std::string>& v_keywords = VerilogKeywords();
  const absl::flat_hash_set<std::string>& keywords =
      use_system_verilog ? sv_keywords : v_keywords;

  if (keywords.contains(block->name())) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module name `", block->name(), "` is a keyword."));
  }
  for (const Block::Port& port : block->GetPorts()) {
    std::string name = Block::PortName(port);
    if (keywords.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Port `", name, "` is a keyword."));
    }
  }

  bool changed = false;
  std::vector<Register*> registers(block->GetRegisters().begin(),
                                   block->GetRegisters().end());
  for (Register* reg : registers) {
    if (!keywords.contains(reg->name())) {
      continue;
    }
    XLS_RETURN_IF_ERROR(RenameRegister(block, reg));
  }

  for (Node* node : block->nodes()) {
    std::string old_name = node->GetName();
    if (!keywords.contains(old_name)) {
      continue;
    }
    // SetName() chooses a new name with a suffix as it doesn't check the node's
    // current name.
    node->SetName(old_name);
    XLS_RET_CHECK_NE(node->GetName(), old_name);
    // Make sure the new name is not a keyword. The renaming policy should not
    // allow this to happen, but it's good to check.
    XLS_RET_CHECK(!keywords.contains(node->GetName()));
    changed = true;
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> NameLegalizationPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    XLS_ASSIGN_OR_RETURN(
        bool block_changed,
        LegalizeNames(block.get(),
                      options.codegen_options.use_system_verilog()));
    changed = changed || block_changed;
  }
  return changed;
}

}  // namespace xls::verilog
