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

xabsl::StatusOr<std::unique_ptr<Lec>> Lec::Create(LecParams params) {
  auto lec = absl::WrapUnique<Lec>(new Lec(params.ir_package,
                                           params.ir_function, params.netlist,
                                           params.netlist_module_name));
  XLS_RETURN_IF_ERROR(lec->Init(params.high_cells));
  return lec;
}

Lec::Lec(Package* ir_package, Function* ir_function,
         netlist::rtl::Netlist* netlist, const std::string& netlist_module_name)
    : ir_package_(ir_package),
      ir_function_(ir_function),
      netlist_(netlist),
      netlist_module_name_(netlist_module_name) {}

Lec::~Lec() {
  if (model_) {
    Z3_model_dec_ref(ctx(), model_.value());
  }
  if (solver_) {
    Z3_solver_dec_ref(ctx(), solver_.value());
  }
}

absl::Status Lec::Init(const absl::flat_hash_set<std::string>& high_cells) {
  XLS_RETURN_IF_ERROR(CreateIrTranslator());
  XLS_RETURN_IF_ERROR(CreateNetlistTranslator(high_cells));
  XLS_ASSIGN_OR_RETURN(Z3_ast netlist_output, UnflattenNetlistOutputs());
  Z3_ast eq_node =
      Z3_mk_eq(ctx(), ir_translator_->GetReturnNode(), netlist_output);
  eq_node = Z3_mk_not(ctx(), eq_node);
  solver_ = solvers::z3::CreateSolver(ctx(), absl::base_internal::NumCPUs());
  Z3_solver_assert(ctx(), solver_.value(), eq_node);

  return absl::OkStatus();
}

absl::Status Lec::AddConstraints(Function* constraints) {
  XLS_RET_CHECK(constraints->params().size() == ir_function_->params().size());

  std::vector<Z3_ast> params;
  for (int i = 0; i < ir_function_->params().size(); i++) {
    XLS_RET_CHECK(constraints->param(i)->GetType()->IsEqualTo(
        ir_function_->param(i)->GetType()));
    params.push_back(ir_translator_->GetTranslation(ir_function_->param(i)));
  }

  XLS_ASSIGN_OR_RETURN(auto constraint_translator,
                       solvers::z3::IrTranslator::CreateAndTranslate(
                           ctx(), constraints, params));
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
  output.push_back(solvers::z3::SolverResultToString(
      ctx(), solver_.value(), satisfiable_ ? Z3_L_TRUE : Z3_L_FALSE,
      /*hexify=*/true));
  if (satisfiable_) {
    output.push_back(
        absl::StrCat("IR result: ",
                     solvers::z3::QueryNode(ctx(), model_.value(),
                                            ir_translator_->GetReturnNode())));
    // TODO(rspringer): This smells of bad factoring, but cleaning it up
    // (eliminating the dependency on IrTranslator::UnflattenZ3Ast()) is a big
    // enough effort that it should be its own change.
    XLS_ASSIGN_OR_RETURN(Z3_ast netlist_output, UnflattenNetlistOutputs());
    output.push_back(absl::StrCat(
        "Netlist result: %s",
        solvers::z3::QueryNode(ctx(), model_.value(), netlist_output)));
  }

  return absl::StrJoin(output, "\n");
}

absl::Status Lec::CreateIrTranslator() {
  XLS_ASSIGN_OR_RETURN(
      ir_translator_,
      solvers::z3::IrTranslator::CreateAndTranslate(ir_function_));
  return absl::OkStatus();
}

absl::flat_hash_map<std::string, Z3_ast> Lec::FlattenNetlistInputs() {
  absl::flat_hash_map<std::string, Z3_ast> inputs;
  for (const Param* param : ir_translator_->xls_function()->params()) {
    // We need to reverse the entire bits vector, per item 1 in the header
    // description, and we need to pass true as little_endian to FlattenValue
    // per item 2.
    std::vector<Z3_ast> bits = ir_translator_->FlattenValue(
        param->GetType(), ir_translator_->GetTranslation(param),
        /*little_endian=*/true);
    std::reverse(bits.begin(), bits.end());
    if (bits.size() > 1) {
      for (int i = 0; i < bits.size(); i++) {
        // Param names are formatted by the parser as
        // <param_name>[<bit_index>] or <param_name> (for single-bit)
        std::string name = absl::StrCat(param->name(), "_", i, "_");
        inputs[name] = bits[i];
      }
    } else {
      inputs[param->name()] = bits[0];
    }
  }

  return inputs;
}

xabsl::StatusOr<Z3_ast> Lec::UnflattenNetlistOutputs() {
  XLS_ASSIGN_OR_RETURN(const netlist::rtl::Module* module,
                       netlist_->GetModule(netlist_module_name_));
  std::vector<Z3_ast> z3_outputs;
  z3_outputs.reserve(module->outputs().size());
  for (const auto& output : module->outputs()) {
    // Drop output wires not part of the original signature.
    if (output->name() == "output_valid") {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Z3_ast z3_output,
                         netlist_translator_->GetTranslation(output));
    z3_outputs.push_back(z3_output);
  }
  std::reverse(z3_outputs.begin(), z3_outputs.end());

  // Specify little endian here as with FlattenValue() above.
  return ir_translator_->UnflattenZ3Ast(ir_function_->GetType()->return_type(),
                                        absl::MakeSpan(z3_outputs),
                                        /*little_endian=*/true);
}

absl::Status Lec::CreateNetlistTranslator(
    const absl::flat_hash_set<std::string>& high_cells) {
  XLS_ASSIGN_OR_RETURN(const netlist::rtl::Module* module,
                       netlist_->GetModule(netlist_module_name_));
  absl::flat_hash_map<std::string, const netlist::rtl::Module*> module_refs;
  for (const std::unique_ptr<netlist::rtl::Module>& module :
       netlist_->modules()) {
    if (module->name() != netlist_module_name_) {
      module_refs[module->name()] = module.get();
    }
  }

  absl::flat_hash_map<std::string, Z3_ast> inputs = FlattenNetlistInputs();

  XLS_ASSIGN_OR_RETURN(
      netlist_translator_,
      NetlistTranslator::CreateAndTranslate(ir_translator_->ctx(), module,
                                            module_refs, inputs, high_cells));

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
    std::cout << "Z3: "
              << solvers::z3::QueryNode(ctx(), model_.value(), translation)
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

}  // namespace z3
}  // namespace solvers
}  // namespace xls
