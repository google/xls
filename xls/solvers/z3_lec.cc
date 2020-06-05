// Copyright 2020 Google LLC
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

#include "absl/base/internal/sysinfo.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"

namespace xls {
namespace solvers {
namespace z3 {

using netlist::rtl::Cell;
using netlist::rtl::Module;
using netlist::rtl::Netlist;
using netlist::rtl::NetRef;

xabsl::StatusOr<std::unique_ptr<Lec>> Lec::Create(const LecParams& params) {
  auto lec = absl::WrapUnique<Lec>(
      new Lec(params.ir_package, params.ir_function, params.netlist,
              params.netlist_module_name, absl::nullopt, 0));
  XLS_RETURN_IF_ERROR(lec->Init(params.high_cells));
  return lec;
}

// This is just sugar; we could add schedule & stage to LecParams, but I like
// the more-explicit invocation style here.
xabsl::StatusOr<std::unique_ptr<Lec>> Lec::CreateForStage(
    const LecParams& params, const PipelineSchedule& schedule, int stage) {
  auto lec = absl::WrapUnique<Lec>(
      new Lec(params.ir_package, params.ir_function, params.netlist,
              params.netlist_module_name, schedule, stage));
  XLS_RETURN_IF_ERROR(lec->Init(params.high_cells));
  return lec;
}

Lec::Lec(Package* ir_package, Function* ir_function, Netlist* netlist,
         const std::string& netlist_module_name,
         absl::optional<PipelineSchedule> schedule, int stage)
    : ir_package_(ir_package),
      ir_function_(ir_function),
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

absl::Status Lec::Init(const absl::flat_hash_set<std::string>& high_cells) {
  XLS_ASSIGN_OR_RETURN(module_, netlist_->GetModule(netlist_module_name_));
  XLS_RETURN_IF_ERROR(CreateIrTranslator());
  XLS_RETURN_IF_ERROR(CreateNetlistTranslator(high_cells));

  std::vector<const Node*> ir_inputs = GetIrInputs();
  XLS_RETURN_IF_ERROR(BindNetlistInputs(absl::MakeSpan(ir_inputs)));

  std::vector<const Node*> ir_output_nodes = GetIrOutputs();
  ir_outputs_.reserve(ir_output_nodes.size());
  for (const Node* node : ir_output_nodes) {
    ir_outputs_.push_back(ir_translator_->GetTranslation(node));
  }

  XLS_ASSIGN_OR_RETURN(netlist_outputs_,
                       GetNetlistOutputs(absl::MakeSpan(ir_output_nodes)));

  std::vector<Z3_ast> eq_nodes;
  eq_nodes.reserve(ir_outputs_.size());
  for (int i = 0; i < ir_outputs_.size(); i++) {
    eq_nodes.push_back(Z3_mk_eq(ctx(), ir_outputs_[i], netlist_outputs_[i]));
  }
  Z3_ast eval_node = Z3_mk_and(ctx(), ir_outputs_.size(), eq_nodes.data());
  eval_node = Z3_mk_not(ctx(), eval_node);
  solver_ = CreateSolver(ctx(), absl::base_internal::NumCPUs());
  Z3_solver_assert(ctx(), solver_.value(), eval_node);

  return absl::OkStatus();
}

std::vector<const Node*> Lec::GetIrInputs() {
  std::vector<const Node*> ir_inputs;
  if (schedule_ && (stage_ != 0)) {
    // If we're evaluating a single stage, then we need to create "fake"
    // Z3 nodes for the stage inputs.
    absl::flat_hash_set<Node*> stage_inputs;
    for (const Node* node : schedule_->nodes_in_cycle(stage_)) {
      for (Node* operand : node->operands()) {
        if (schedule_->cycle(operand) != stage_) {
          stage_inputs.insert(operand);
        }
      }
    }

    // Create new Z3 [free] constants to replace those out-of-stage inputs.
    for (Node* stage_input : stage_inputs) {
      Z3_ast foo = Z3_mk_const(
          ctx(), Z3_mk_string_symbol(ctx(), stage_input->GetName().c_str()),
          TypeToSort(ctx(), *stage_input->GetType()));
      // Update the translator to use this new input for later references.
      ir_translator_->SetTranslation(stage_input, foo);
      ir_inputs.push_back(stage_input);
    }
  } else {
    // Otherwise, just collect the function inputs.
    for (const Param* param : ir_function_->params()) {
      ir_inputs.push_back(param);
    }
  }
  return ir_inputs;
}

std::vector<const Node*> Lec::GetIrOutputs() {
  // Easy case first! If we're working on the whole function, then we just need
  // the output node & its corresponding wires.
  if (!schedule_) {
    return {ir_function_->return_value()};
  }

  // First, collect all stage inputs - those input nodes not present in this
  // cycle - along with all stage outputs.
  absl::flat_hash_set<const Node*> stage_outputs;
  for (const Node* node : schedule_->nodes_in_cycle(stage_)) {
    for (const Node* user : node->users()) {
      if (schedule_->cycle(user) != stage_) {
        stage_outputs.insert(node);
      }
    }
  }

  std::vector<const Node*> ir_output_nodes;
  ir_output_nodes.reserve(stage_outputs.size());
  for (const Node* node : stage_outputs) {
    ir_output_nodes.push_back(node);
  }

  return ir_output_nodes;
}

xabsl::StatusOr<std::vector<NetRef>> Lec::GetIrNetrefs(const Node* node) {
  std::vector<NetRef> refs;
  // Increment down (instead of up) to keep the std::reverse symmetry in
  // UnflattenNetlistOutputs() (WRT FlattenNetlistInputs()).
  for (int i = node->GetType()->GetFlatBitCount() - 1; i >= 0; i--) {
    // "bit_index" identifies individual wires part of a multi-bit IR value.
    absl::optional<int> bit_index;
    if (node->GetType()->GetFlatBitCount() > 1) {
      bit_index = i;
    }
    std::string name = NodeToWireName(node, bit_index);

    XLS_ASSIGN_OR_RETURN(const Cell* cell, module_->ResolveCell(name));
    for (const auto& output : cell->outputs()) {
      refs.push_back(output.netref);
    }
  }

  return refs;
}

absl::Status Lec::AddConstraints(Function* constraints) {
  XLS_RET_CHECK(!schedule_)
      << "Constraints cannot be specified with per-stage LEC.";
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
  satisfiable_ = Z3_solver_check(ctx(), solver_.value()) == Z3_L_TRUE;
  if (satisfiable_) {
    model_ = Z3_solver_get_model(ctx(), solver_.value());
    Z3_model_inc_ref(ctx(), model_.value());
  }
  return !satisfiable_;
}

xabsl::StatusOr<std::string> Lec::ResultToString() {
  std::vector<std::string> output;
  output.push_back(SolverResultToString(ctx(), solver_.value(),
                                        satisfiable_ ? Z3_L_TRUE : Z3_L_FALSE,
                                        /*hexify=*/true));
  if (satisfiable_) {
    output.push_back("IR result:");
    for (Z3_ast node : ir_outputs_) {
      output.push_back(QueryNode(ctx(), model_.value(), node));
    }

    output.push_back("Netlist result:");
    for (Z3_ast node : netlist_outputs_) {
      output.push_back(QueryNode(ctx(), model_.value(), node));
    }
  }

  return absl::StrJoin(output, "\n");
}

absl::Status Lec::CreateIrTranslator() {
  XLS_ASSIGN_OR_RETURN(ir_translator_,
                       IrTranslator::CreateAndTranslate(ir_function_));
  return absl::OkStatus();
}

absl::Status Lec::BindNetlistInputs(absl::Span<const Node*> ir_inputs) {
  absl::flat_hash_map<std::string, Z3_ast> inputs =
      FlattenNetlistInputs(ir_inputs);
  for (auto& input : inputs) {
    // Skip synthesized inputs.
    if (input.first == "clk" || input.first == "input_valid") {
      continue;
    }
    XLS_RETURN_IF_ERROR(
        netlist_translator_->RebindInputNet(input.first, input.second));
  }

  return absl::OkStatus();
}

absl::flat_hash_map<std::string, Z3_ast> Lec::FlattenNetlistInputs(
    absl::Span<const Node*> ir_inputs) {
  absl::flat_hash_map<std::string, Z3_ast> netlist_inputs;
  for (const Node* node : ir_inputs) {
    // We need to reverse the entire bits vector, per item 1 in the header
    // description, and we need to pass true as little_endian to FlattenValue
    // per item 2.
    Z3_ast translation = ir_translator_->GetTranslation(node);
    std::vector<Z3_ast> bits = ir_translator_->FlattenValue(
        node->GetType(), translation, /*little_endian=*/true);
    std::reverse(bits.begin(), bits.end());
    if (bits.size() > 1) {
      for (int i = 0; i < bits.size(); i++) {
        std::string name = NodeToWireName(node, i);
        netlist_inputs[name] = bits[i];
      }
    } else {
      std::string name = NodeToWireName(node, std::nullopt);
      netlist_inputs[name] = bits[0];
    }
  }

  return netlist_inputs;
}

xabsl::StatusOr<std::vector<Z3_ast>> Lec::GetNetlistOutputs(
    absl::Span<const Node*> ir_outputs) {
  std::vector<Z3_ast> unflattened_outputs;

  for (const Node* node : ir_outputs) {
    XLS_ASSIGN_OR_RETURN(std::vector<NetRef> netrefs, GetIrNetrefs(node));

    std::vector<Z3_ast> z3_outputs;
    z3_outputs.reserve(netrefs.size());
    for (const auto& netref : netrefs) {
      // Drop output wires not part of the original signature.
      if (netref->name() == "output_valid") {
        continue;
      }

      XLS_ASSIGN_OR_RETURN(Z3_ast z3_output,
                           netlist_translator_->GetTranslation(netref));
      z3_outputs.push_back(z3_output);
    }
    std::reverse(z3_outputs.begin(), z3_outputs.end());

    // Specify little endian here as with FlattenValue() above.
    Z3_ast unflattened_output = ir_translator_->UnflattenZ3Ast(
        node->GetType(), absl::MakeSpan(z3_outputs),
        /*little_endian=*/true);
    unflattened_outputs.push_back(unflattened_output);
  }
  return unflattened_outputs;
}

absl::Status Lec::CreateNetlistTranslator(
    const absl::flat_hash_set<std::string>& high_cells) {
  absl::flat_hash_map<std::string, const Module*> module_refs;
  for (const std::unique_ptr<Module>& module : netlist_->modules()) {
    if (module->name() != netlist_module_name_) {
      module_refs[module->name()] = module.get();
    }
  }

  XLS_ASSIGN_OR_RETURN(
      netlist_translator_,
      NetlistTranslator::CreateAndTranslate(ir_translator_->ctx(), module_,
                                            module_refs, high_cells));

  return absl::OkStatus();
}

void Lec::DumpIrTree() {
  std::deque<const Node*> to_process;
  to_process.push_back(ir_translator_->xls_function()->return_value());

  absl::flat_hash_set<const Node*> seen;
  while (!to_process.empty()) {
    const Node* node = to_process.front();
    to_process.pop_front();
    Z3_ast translation = ir_translator_->GetTranslation(node);
    std::cout << "IR: " << node->ToString() << std::endl;
    std::cout << "Z3: " << QueryNode(ctx(), model_.value(), translation)
              << std::endl
              << std::endl;
    seen.insert(node);
    for (const Node* operand : node->operands()) {
      if (!seen.contains(operand)) {
        to_process.push_back(operand);
      }
    }
  }
}

// Bit 1 of the 3-bit IR node foo.123 in stage 3 is present as p3_foo_123_1_.
std::string Lec::NodeToWireName(const Node* node,
                                absl::optional<int> bit_index) {
  std::string name = node->GetName();
  for (char& c : name) {
    if (c == '.') {
      c = '_';
    }
  }

  int stage = -1;
  if (schedule_) {
    stage = stage_;
  }

  // If we're doing staged blah blah blah
  name = absl::StrCat("p", stage + 1, "_", name);

  // Each IR gate can only have one [multi-bit] output.
  if (bit_index) {
    absl::StrAppend(&name, "_", bit_index.value(), "_");
  }

  return name;
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls
