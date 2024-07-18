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

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"

namespace xls::verilog {
namespace {

const absl::flat_hash_set<std::string>& VerilogKeywords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kKeywords({
      "always",
      "and",
      "assign",
      "automatic",
      "begin",
      "buf",
      "bufif0",
      "bufif1",
      "case",
      "casex",
      "casez",
      "cell",
      "cmos",
      "config",
      "deassign",
      "default",
      "defparam",
      "design",
      "disable",
      "edge",
      "else",
      "end",
      "endcase",
      "endconfig",
      "endfunction",
      "endgenerate",
      "endmodule",
      "endprimitive",
      "endspecify",
      "endtable",
      "endtask",
      "event",
      "for",
      "force",
      "forever",
      "fork",
      "function",
      "generate",
      "genvar",
      "highz0",
      "highz1",
      "if",
      "ifnone",
      "incdir",
      "include",
      "initial",
      "inout",
      "input",
      "instance",
      "integer",
      "join",
      "large",
      "liblist",
      "library",
      "localparam",
      "macromodule",
      "medium",
      "module",
      "nand",
      "negedge",
      "nmos",
      "nor",
      "noshowcancelled",
      "not",
      "notif0",
      "notif1",
      "or",
      "output",
      "parameter",
      "pmos",
      "posedge",
      "primitive",
      "pull0",
      "pull1",
      "pulldown",
      "pullup",
      "pulsestyle_onevent",
      "pulsestyle_ondetect",
      "rcmos",
      "real",
      "realtime",
      "reg",
      "release",
      "repeat",
      "rnmos",
      "rpmos",
      "rtran",
      "rtranif0",
      "rtranif1",
      "scalared",
      "showcancelled",
      "signed",
      "small",
      "specify",
      "specparam",
      "strong0",
      "strong1",
      "supply0",
      "supply1",
      "table",
      "task",
      "time",
      "tran",
      "tranif0",
      "tranif1",
      "tri",
      "tri0",
      "tri1",
      "triand",
      "trior",
      "trireg",
      "unsigned",
      "use",
      "uwire",
      "vectored",
      "wait",
      "wand",
      "weak0",
      "weak1",
      "while",
      "wire",
      "wor",
      "xnor",
      "xor",
  });
  return *kKeywords;
}

const absl::flat_hash_set<std::string>& SystemVerilogKeywords() {
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> kKeywords(
      [] {
        absl::flat_hash_set<std::string> keywords{
            // 1800-2005
            "alias",
            "always_comb",
            "always_ff",
            "always_latch",
            "assert",
            "assume",
            "before",
            "bind",
            "bins",
            "binsof",
            "bit",
            "break",
            "byte",
            "chandle",
            "class",
            "clocking",
            "const",
            "constraint",
            "context",
            "continue",
            "cover",
            "covergroup",
            "coverpoint",
            "cross",
            "dist",
            "do",
            "endclass",
            "endclocking",
            "endgroup",
            "endinterface",
            "endpackage",
            "endprogram",
            "endproperty",
            "endsequence",
            "enum",
            "expect",
            "export",
            "extends",
            "extern",
            "final",
            "first_match",
            "foreach",
            "forkjoin",
            "iff",
            "ignore_bins",
            "illegal_bins",
            "import",
            "inside",
            "int",
            "interface",
            "intersect",
            "join_any",
            "join_none",
            "local",
            "logic",
            "longint",
            "matches",
            "modport",
            "new",
            "null",
            "package",
            "packed",
            "priority",
            "program",
            "property",
            "protected",
            "pure",
            "rand",
            "randc",
            "randcase",
            "randsequence",
            "ref",
            "return",
            "sequence",
            "shortint",
            "shortreal",
            "solve",
            "static",
            "string",
            "struct",
            "super",
            "tagged",
            "this",
            "throughout",
            "timeprecision",
            "timeunit",
            "type",
            "typedef",
            "union",
            "unique",
            "var",
            "virtual",
            "void",
            "wait_order",
            "wildcard",
            "with",
            "within",

            // 1800-2009
            "accept_on",
            "checker",
            "endchecker",
            "eventually",
            "global",
            "implies",
            "let",
            "nexttime",
            "reject_on",
            "restrict",
            "s_always",
            "s_eventually",
            "s_nexttime",
            "s_until",
            "s_until_with",
            "strong",
            "sync_accept_on",
            "sync_reject_on",
            "unique0",
            "until",
            "until_with",
            "untyped",
            "weak",

            // 1800-2012
            "implements",
            "interconnect",
            "nettype",
            "soft",
        };
        // SystemVerilog keywords are a superset of Verilog keywords.
        keywords.insert(VerilogKeywords().begin(), VerilogKeywords().end());
        return keywords;
      }());
  return *kKeywords;
}

// Add a new copy of register `old_reg` to get a new name. Replace all uses of
// the old register with the new one and remove the old register.
// This effectively renames the `old_reg`.
absl::Status RenameRegister(Block* block, Register* old_reg) {
  std::string_view old_name = old_reg->name();
  XLS_ASSIGN_OR_RETURN(
      Register * new_reg,
      block->AddRegister(old_reg->name(), old_reg->type(), old_reg->reset()));
  XLS_RET_CHECK_NE(old_name, new_reg->name());

  std::vector<Node*> to_remove;

  for (Node* node : block->nodes()) {
    if (node->Is<RegisterRead>() &&
        node->As<RegisterRead>()->GetRegister() == old_reg) {
      RegisterRead* old_read = node->As<RegisterRead>();
      XLS_RETURN_IF_ERROR(
          old_read->ReplaceUsesWithNew<RegisterRead>(new_reg).status());
      to_remove.push_back(old_read);
    }
    if (node->Is<RegisterWrite>() &&
        node->As<RegisterWrite>()->GetRegister() == old_reg) {
      RegisterWrite* old_write = node->As<RegisterWrite>();
      XLS_RETURN_IF_ERROR(old_write
                              ->ReplaceUsesWithNew<RegisterWrite>(
                                  old_write->data(), old_write->load_enable(),
                                  old_write->reset(), new_reg)
                              .status());
      to_remove.push_back(old_write);
    }
  }

  for (Node* node : to_remove) {
    XLS_RETURN_IF_ERROR(block->RemoveNode(node));
  }

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
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : unit->package->blocks()) {
    XLS_ASSIGN_OR_RETURN(
        bool block_changed,
        LegalizeNames(block.get(),
                      options.codegen_options.use_system_verilog()));
    changed = changed || block_changed;
  }
  return changed;
}

}  // namespace xls::verilog
