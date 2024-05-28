// Copyright 2020 The XLS Authors
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

#include "xls/solvers/z3_lec.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/netlist/netlist.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_netlist_translator.h"
#include "xls/solvers/z3_utils.h"
#include "external/z3/src/api/z3_api.h"

namespace xls {
namespace solvers {
namespace z3 {

using netlist::rtl::Module;
using netlist::rtl::Netlist;
using netlist::rtl::NetRef;

namespace {

// Returns true if we're checking a single pipeline stage.
bool CheckingSingleStage(std::optional<PipelineSchedule> schedule,
                         int32_t stage) {
  return schedule && stage != -1;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Lec>> Lec::Create(const LecParams& params) {
  auto lec = absl::WrapUnique<Lec>(new Lec(params.ir_function, params.netlist,
                                           params.netlist_module_name,
                                           std::nullopt, 0));
  XLS_RETURN_IF_ERROR(lec->Init());
  return lec;
}

// This is just sugar; we could add schedule & stage to LecParams, but I like
// the more-explicit invocation style here.
absl::StatusOr<std::unique_ptr<Lec>> Lec::CreateForStage(
    const LecParams& params, const PipelineSchedule& schedule, int stage) {
  auto lec = absl::WrapUnique<Lec>(new Lec(params.ir_function, params.netlist,
                                           params.netlist_module_name, schedule,
                                           stage));
  XLS_RETURN_IF_ERROR(lec->Init());
  return lec;
}

Lec::Lec(Function* ir_function, Netlist* netlist,
         const std::string& netlist_module_name,
         std::optional<PipelineSchedule> schedule, int stage)
    : ir_function_(ir_function),
      netlist_(netlist),
      netlist_module_name_(netlist_module_name),
      schedule_(schedule),
      stage_(stage) {}

Lec::~Lec() {
  if (model_) {
    Z3_model_dec_ref(ctx(), model_.value());
  }
  if (solver_) {
    Z3_solver_dec_ref(ctx(), solver_.value());
  }
}

absl::Status Lec::Init() {
  XLS_ASSIGN_OR_RETURN(module_, netlist_->GetModule(netlist_module_name_));
  XLS_RETURN_IF_ERROR(CreateIrTranslator());
  XLS_RETURN_IF_ERROR(CreateNetlistTranslator());

  XLS_RETURN_IF_ERROR(CollectIrInputs());
  if (VLOG_IS_ON(2)) {
    for (const auto& pair : input_mapping_) {
      LOG(INFO) << "Stage input [IR] node: " << pair.first;
    }
  }
  XLS_RETURN_IF_ERROR(BindNetlistInputs());

  CollectIrOutputNodes();
  // "Filler" value for unused output bits (those not present in the netlist).
  // Helpful for reading result output.
  Z3_ast x = Z3_mk_const(ctx(), Z3_mk_string_symbol(ctx(), "X"),
                         Z3_mk_bv_sort(ctx(), 1));
  std::vector<Z3_ast> eq_nodes;
  for (const Node* node : ir_output_nodes_) {
    // Extract the individual bits out of each IR output node, and match those
    // up the corresponding netlist bits. The netlist outputs do not contain
    // references to bits that are actually unused, instead having nullptr in
    // those indices.
    VLOG(3) << "Stage output [IR] node: " << node->GetName()
            << " : width: " << node->GetType()->GetFlatBitCount();
    std::vector<Z3_ast> ir_bits = ir_translator_->FlattenValue(
        node->GetType(), ir_translator_->GetTranslation(node),
        /*little_endian=*/true);
    XLS_ASSIGN_OR_RETURN(std::vector<Z3_ast> netlist_bits,
                         GetNetlistZ3ForIr(node));
    XLS_RET_CHECK(ir_bits.size() == netlist_bits.size());

    for (int i = 0; i < ir_bits.size(); i++) {
      if (netlist_bits[i] == nullptr) {
        VLOG(3) << "  Skipping " << node->GetName() << " IR output bit " << i;
        ir_outputs_.push_back(x);
        netlist_outputs_.push_back(x);
      } else {
        ir_outputs_.push_back(ir_bits[i]);
        netlist_outputs_.push_back(netlist_bits[i]);
        eq_nodes.push_back(Z3_mk_eq(ctx(), ir_bits[i], netlist_bits[i]));
      }
    }
  }

  Z3_ast eval_node = Z3_mk_and(ctx(), eq_nodes.size(), eq_nodes.data());
  eval_node = Z3_mk_not(ctx(), eval_node);
  solver_ = CreateSolver(ctx(), std::thread::hardware_concurrency());
  Z3_solver_assert(ctx(), solver_.value(), eval_node);

  return absl::OkStatus();
}

absl::Status Lec::CollectIrInputs() {
  if (CheckingSingleStage(schedule_, stage_) && stage_ != 0) {
    // If we're evaluating a single stage (aside from the first), then we need
    // to create "fake" Z3 nodes for the stage inputs.
    absl::flat_hash_set<const Node*> stage_inputs;
    for (const Node* node : schedule_->nodes_in_cycle(stage_)) {
      for (Node* operand : node->operands()) {
        if (schedule_->cycle(operand) != stage_) {
          stage_inputs.insert(operand);
        }
      }
    }

    // Create new Z3 [free] constants to replace those out-of-stage inputs.
    // Plop the inputs into a vector & sort it for deterministic iteration
    // order.
    for (const Node* stage_input : SetToSortedVector(stage_inputs)) {
      Z3_ast new_input = Z3_mk_const(
          ctx(), Z3_mk_string_symbol(ctx(), stage_input->GetName().c_str()),
          TypeToSort(ctx(), *stage_input->GetType()));
      // Update the translator to use this new input for later references.
      input_mapping_[stage_input] = new_input;
    }
    XLS_RETURN_IF_ERROR(ir_translator_->Retranslate(input_mapping_));
  } else {
    // Otherwise, just collect the function inputs.
    for (const Param* param : ir_function_->params()) {
      input_mapping_[param] = ir_translator_->GetTranslation(param);
    }
  }
  return absl::OkStatus();
}

void Lec::CollectIrOutputNodes() {
  // Easy case first! If we're working on the whole function, then we just need
  // the output node & its corresponding wires.
  if (!CheckingSingleStage(schedule_, stage_)) {
    ir_output_nodes_ = {ir_function_->return_value()};
    return;
  }

  // Collect all stage outputs - those nodes using a node within this stage
  // but that aren't present in this cycle. Use a set to uniqify the output
  // nodes.
  absl::flat_hash_set<const Node*> stage_outputs;
  for (const Node* node : schedule_->nodes_in_cycle(stage_)) {
    for (const Node* user : node->users()) {
      if (schedule_->cycle(user) != stage_) {
        stage_outputs.insert(node);
      }
    }
  }

  // Ensure a deterministic output order.
  ir_output_nodes_ = SetToSortedVector(stage_outputs);
}

absl::StatusOr<std::vector<NetRef>> Lec::GetIrNetrefs(const Node* node) {
  std::vector<NetRef> refs;
  // Increment down (instead of up) to keep the std::reverse symmetry in
  // UnflattenNetlistOutputs() (WRT FlattenNetlistInputs()).
  for (int i = node->GetType()->GetFlatBitCount() - 1; i >= 0; i--) {
    // "bit_index" identifies individual wires part of a multi-bit IR value.
    std::optional<int> bit_index;
    if (node->GetType()->GetFlatBitCount() > 1) {
      bit_index = i;
    }

    std::string name = NodeToNetlistName(node, bit_index);

    auto status_or_cell = module_->ResolveCell(name);
    if (!status_or_cell.ok()) {
      // There are netrefs that aren't present as registers but _are_ present as
      // combinational wires (those w/a "comb" suffix) - these are values used
      // within a stage, but not as input or output. Knowing their values is
      // useful for debugging, so we make an attempt to find them here.
      name = NodeToNetlistName(node, bit_index, /*is_cell=*/false);
      auto status_or_ref = module_->ResolveNet(name);
      if (status_or_ref.ok()) {
        refs.push_back(status_or_ref.value());
      } else {
        // This line is assuming that the "unused" cell only has a single
        // output. In cases we've tested so far, this has always been the case,
        // but we should keep it in mind if we encounter problems in the future.
        refs.push_back(nullptr);
      }
    } else {
      for (const auto& output : status_or_cell.value()->outputs()) {
        refs.push_back(output.netref);
      }
    }
  }

  return refs;
}

absl::Status Lec::AddConstraints(Function* constraints) {
  XLS_RET_CHECK(!CheckingSingleStage(schedule_, stage_) || stage_ == 0)
      << "Constraints cannot be specified with per-stage LEC, "
      << "except on the first stage.";
  XLS_RET_CHECK(constraints->params().size() == ir_function_->params().size());

  std::vector<Z3_ast> params;
  for (int i = 0; i < ir_function_->params().size(); i++) {
    XLS_RET_CHECK(constraints->param(i)->GetType()->IsEqualTo(
        ir_function_->param(i)->GetType()));
    params.push_back(ir_translator_->GetTranslation(ir_function_->param(i)));
  }

  XLS_ASSIGN_OR_RETURN(
      auto constraint_translator,
      IrTranslator::CreateAndTranslate(ctx(), constraints, params));
  Z3_ast eq_node = Z3_mk_eq(ctx(), constraint_translator->GetReturnNode(),
                            Z3_mk_int(ctx(), 1, Z3_mk_bv_sort(ctx(), 1)));
  Z3_solver_assert(ctx(), solver_.value(), eq_node);
  return absl::OkStatus();
}

bool Lec::Run() {
  LOG(INFO) << "Beginning execution";
  satisfiable_ = Z3_solver_check(ctx(), solver_.value()) == Z3_L_TRUE;
  if (satisfiable_) {
    model_ = Z3_solver_get_model(ctx(), solver_.value());
    Z3_model_inc_ref(ctx(), model_.value());
  }
  return !satisfiable_;
}

std::string Lec::ResultToString() {
  std::vector<std::string> output;
  output.push_back(SolverResultToString(ctx(), solver_.value(),
                                        satisfiable_ ? Z3_L_TRUE : Z3_L_FALSE,
                                        /*hexify=*/true));
  if (satisfiable_) {
    for (const Node* node : ir_output_nodes_) {
      std::pair<std::string, std::string> outputs = GetComparisonStrings(node);
      std::string ir_string = outputs.first;
      std::string nl_string = outputs.second;

      // Only for printing model outputs do we mask "don't cares" in IR bits.
      std::vector<Z3_ast> nl_bits = GetNetlistZ3ForIr(node).value();
      MarkDontCareBits(nl_bits, ir_string);

      BitsRope ir_rope(node->BitCountOrDie());
      BitsRope nl_rope(node->BitCountOrDie());
      for (int i = ir_string.size() - 1; i >= 2; i--) {
        if (ir_string[i] == '_') {
          continue;
        }
        ir_rope.push_back(ir_string[i] == '1');
        nl_rope.push_back(nl_string[i] == '1');
      }

      output.push_back(
          absl::StrCat("\nOutput IR node ", node->ToString(), ":"));
      output.push_back(absl::StrCat(
          "  IR: ", ir_string, " (",
          BitsToString(ir_rope.Build(), FormatPreference::kHex), ")"));
      output.push_back(absl::StrCat(
          "  NL: ", outputs.second, " (",
          BitsToString(nl_rope.Build(), FormatPreference::kHex), ")"));
    }
  }

  return absl::StrJoin(output, "\n");
}

absl::Status Lec::CreateIrTranslator() {
  XLS_ASSIGN_OR_RETURN(ir_translator_,
                       IrTranslator::CreateAndTranslate(ir_function_));
  return absl::OkStatus();
}

absl::Status Lec::BindNetlistInputs() {
  absl::flat_hash_map<std::string, Z3_ast> nl_inputs = FlattenNetlistInputs();
  return netlist_translator_->Retranslate(nl_inputs);
}

absl::flat_hash_map<std::string, Z3_ast> Lec::FlattenNetlistInputs() {
  absl::flat_hash_map<std::string, Z3_ast> netlist_inputs;
  for (const auto& pair : input_mapping_) {
    // We need to reverse the entire bits vector, per item 1 in the header
    // description, and we need to pass true as little_endian to FlattenValue
    // per item 2.
    const Node* node = pair.first;
    Z3_ast translation = pair.second;
    std::vector<Z3_ast> bits = ir_translator_->FlattenValue(
        node->GetType(), translation, /*little_endian=*/true);
    std::reverse(bits.begin(), bits.end());
    for (int i = 0; i < bits.size(); i++) {
      // We have a flat IR node that's our input; we need to find the matching
      // cells and use their outputs.
      std::string name;
      if (bits.size() == 1) {
        name = NodeToNetlistName(node, std::nullopt);
      } else {
        name = NodeToNetlistName(node, i);
      }

      // Get the cell...
      auto status_or_cell = module_->ResolveCell(name);
      if (!status_or_cell.ok()) {
        VLOG(3) << "Could not resolve input cell: " << name << "; skipping";
        LOG(INFO) << "Could not resolve input cell: " << name << "; skipping";
        continue;
      }

      // Then plop its output in.
      for (const auto& output : status_or_cell.value()->outputs()) {
        netlist_inputs[output.netref->name()] = bits[i];
      }
    }
  }

  return netlist_inputs;
}

absl::StatusOr<std::vector<Z3_ast>> Lec::GetNetlistZ3ForIr(const Node* node) {
  std::vector<Z3_ast> netlist_output;

  XLS_ASSIGN_OR_RETURN(std::vector<NetRef> netrefs, GetIrNetrefs(node));
  netlist_output.reserve(netrefs.size());
  for (const auto& netref : netrefs) {
    if (netref == nullptr) {
      netlist_output.push_back(nullptr);
    } else if (netref->name() == "output_valid") {
      // Drop output wires not part of the original signature.
      // TODO(rspringer): These special wires aren't necessarily fixed - they're
      // specified by codegen, and could change in the future. These need to be
      // properly handled (i.e., not hardcoded).
      continue;
    } else {
      XLS_ASSIGN_OR_RETURN(Z3_ast z3_output,
                           netlist_translator_->GetTranslation(netref));
      netlist_output.push_back(z3_output);
    }
  }
  return netlist_output;
}

absl::Status Lec::CreateNetlistTranslator() {
  absl::flat_hash_map<std::string, const Module*> module_refs;
  for (const std::unique_ptr<Module>& module : netlist_->modules()) {
    if (module->name() != netlist_module_name_) {
      module_refs[module->name()] = module.get();
    }
  }

  XLS_ASSIGN_OR_RETURN(netlist_translator_,
                       NetlistTranslator::CreateAndTranslate(
                           ir_translator_->ctx(), module_, module_refs));

  return absl::OkStatus();
}

void Lec::DumpIrTree() {
  std::deque<const Node*> to_process;
  absl::flat_hash_set<const Node*> seen;
  for (const Node* node : ir_output_nodes_) {
    to_process.push_back(node);
    seen.insert(node);
  }

  for (const auto& pair : input_mapping_) {
    to_process.push_back(pair.first);
    seen.insert(pair.first);
  }

  while (!to_process.empty()) {
    const Node* node = to_process.front();
    to_process.pop_front();

    std::cout << "Node: " << node->ToString() << '\n';
    std::pair<std::string, std::string> outputs = GetComparisonStrings(node);
    std::cout << "  IR: " << outputs.first << '\n';
    std::cout << "  NL: " << outputs.second << '\n' << '\n';

    for (const Node* operand : node->operands()) {
      if (!seen.contains(operand) && schedule_->cycle(operand) == stage_) {
        to_process.push_back(operand);
        seen.insert(operand);
      }
    }
  }
}

std::pair<std::string, std::string> Lec::GetComparisonStrings(
    const Node* node) {
  std::vector<Z3_ast> ir_bits;
  if (input_mapping_.contains(node)) {
    ir_bits =
        ir_translator_->FlattenValue(node->GetType(), input_mapping_[node]);
  } else {
    ir_bits = ir_translator_->FlattenValue(
        node->GetType(), ir_translator_->GetTranslation(node));
  }
  std::reverse(ir_bits.begin(), ir_bits.end());

  auto status_or_nl_bits = GetNetlistZ3ForIr(node);
  if (!status_or_nl_bits.ok()) {
    VLOG(2) << "Node " << node->GetName() << " not present in netlist.";
    return std::make_pair("", "");
  }
  std::vector<Z3_ast> nl_bits = status_or_nl_bits.value();
  std::string ir_string = BitVectorToString(ctx(), ir_bits, model_.value());
  std::string nl_string = BitVectorToString(ctx(), nl_bits, model_.value());
  MarkDontCareBits(nl_bits, nl_string);
  return std::make_pair(ir_string, nl_string);
}

void Lec::MarkDontCareBits(const std::vector<Z3_ast>& nl_bits,
                           std::string& nl_string) {
  int bit_count = nl_bits.size();
  int skip_pos = bit_count % 4;
  int output_pos = 2;
  for (int i = 0; i < bit_count; i++) {
    if (nl_bits[i] == nullptr) {
      nl_string[output_pos] = 'X';
    }

    output_pos++;
    skip_pos--;
    if (skip_pos == 0) {
      output_pos++;
      skip_pos = 4;
    }
  }
}

// Bit 1 of the 3-bit IR node foo.123 in stage 3 is present as p3_foo_123_1_.
std::string Lec::NodeToNetlistName(const Node* node,
                                   std::optional<int> bit_index,
                                   bool is_cell) {
  std::string name = verilog::SanitizeIdentifier(node->GetName());
  for (char& c : name) {
    if (c == '.') {
      c = '_';
    }
  }

  bool is_input = node->Is<Param>();
  int stage = schedule_ ? schedule_.value().cycle(node) + 1 : 0;
  if (schedule_ && is_input) {
    name = absl::StrCat("p0_", name, is_cell ? "_reg" : "_comb");
  } else {
    name = absl::StrCat("p", stage, "_", name, is_cell ? "_reg" : "_comb");
  }

  // Each IR gate can only have one [multi-bit] output.
  if (bit_index) {
    absl::StrAppend(&name, "_", bit_index.value(), "_");
  }

  return name;
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls
