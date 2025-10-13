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

#include "xls/dslx/ir_convert/get_conversion_records.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/ir_convert/conversion_record.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/public/status_macros.h"

namespace xls::dslx {

namespace {

// Makes the conversion record from the pieces.
absl::StatusOr<ConversionRecord> MakeConversionRecord(
    Function* f, Module* m, TypeInfo* type_info, const ParametricEnv& bindings,
    std::optional<ProcId> proc_id, const Invocation* invocation, bool is_top,
    std::unique_ptr<ConversionRecord> config_record = nullptr) {
  return ConversionRecord::Make(f, invocation, m, type_info, bindings, proc_id,
                                is_top, std::move(config_record));
}

// An AstNodeVisitor that creates ConversionRecords from appropriate AstNodes
// like Function, QuickCheck, Proc, etc.
class ConversionRecordVisitor : public AstNodeVisitorWithDefault {
 public:
  ConversionRecordVisitor(Module* module, TypeInfo* type_info,
                          bool include_tests, ProcIdFactory proc_id_factory,
                          AstNode* top)
      : module_(module),
        type_info_(type_info),
        include_tests_(include_tests),
        proc_id_factory_(proc_id_factory),
        top_(top) {}

  absl::StatusOr<ConversionRecord> InvocationToConversionRecord(
      const Function* f, const Invocation* invocation,
      TypeInfo* instantiation_type_info, ParametricEnv callee_bindings,
      ParametricEnv caller_bindings, bool is_top,
      std::optional<ProcId> proc_id) {
    if (invocation != nullptr) {
      VLOG(5) << "Processing invocation " << invocation->ToString();
    } else {
      VLOG(5) << "Processing fn " << f->ToString();
    }
    // Note, it's possible there is no config invocation if it's a
    // top proc or some other reason.
    std::unique_ptr<ConversionRecord> config_record;
    if (f->tag() == FunctionTag::kProcNext) {
      // If this is a proc next function, find the corresponding config
      // invocation (spawn) with the same parametrics and put it in the
      // conversion record.
      const Function& config_fn = f->proc().value()->config();
      std::vector<InvocationCalleeData> all_callee_data =
          type_info_->GetUniqueInvocationCalleeData(&config_fn);
      for (InvocationCalleeData& config_callee_data : all_callee_data) {
        if (config_callee_data.callee_bindings == callee_bindings) {
          VLOG(5) << "Found config for next: "
                  << config_callee_data.invocation->ToString();
          const Invocation* config_invocation = config_callee_data.invocation;
          std::optional<TypeInfo*> config_type_info =
              type_info_->GetInvocationTypeInfo(config_invocation,
                                                caller_bindings);
          XLS_RET_CHECK(config_type_info.has_value())
              << "Could not find instantiation for `"
              << config_invocation->ToString()
              << "` via bindings: " << caller_bindings;
          XLS_ASSIGN_OR_RETURN(
              ConversionRecord cr,
              MakeConversionRecord(const_cast<Function*>(f), module_,
                                   *config_type_info, callee_bindings, proc_id,
                                   config_invocation,
                                   // config functions can never be 'top'
                                   /*is_top=*/false));
          config_record = std::make_unique<ConversionRecord>(std::move(cr));
          break;
        }
      }
    }

    XLS_ASSIGN_OR_RETURN(
        ConversionRecord cr,
        MakeConversionRecord(const_cast<Function*>(f), module_,
                             instantiation_type_info, callee_bindings, proc_id,
                             invocation, is_top, std::move(config_record)));
    return cr;
  }

  // TODO: davidplass - whenever a parametric proc spawns another parametric
  // proc, for every unique invocation of the parent proc we have to add the
  // child proc to the list as well, recursively.
  absl::Status HandleSpawn(const Spawn* spawn) override {
    Invocation* invocation = spawn->config();
    auto root_invocation_data = type_info_->GetRootInvocationData(invocation);
    XLS_RET_CHECK(root_invocation_data.has_value());
    const InvocationData* invocation_data = *root_invocation_data;
    const Function* config_fn = invocation_data->callee();
    XLS_RET_CHECK(config_fn->proc().has_value());
    Proc* proc = config_fn->proc().value();
    const Function* next_fn = &proc->next();

    std::optional<ProcId> proc_id = proc_id_factory_.CreateProcId(
        /*parent=*/std::nullopt, proc,
        /*count_as_new_instance=*/false);

    std::vector<InvocationCalleeData> calls =
        type_info_->GetUniqueInvocationCalleeData(next_fn);
    // Look at these calls and find the one with the
    // (caller) parametric env that matches the invocation_datum.
    for (auto& callee_data : calls) {
      for (auto& [caller_bindings, invocation_datum] :
           invocation_data->env_to_callee_data()) {
        if (callee_data.caller_bindings == caller_bindings) {
          XLS_ASSIGN_OR_RETURN(
              ConversionRecord cr,
              InvocationToConversionRecord(
                  next_fn, callee_data.invocation,
                  callee_data.derived_type_info,
                  invocation_datum.callee_bindings,
                  invocation_datum.caller_bindings,
                  // Since this proc is being spawned, it's certainly not top.
                  /* is_top= */ false, proc_id));
          records_.push_back(std::move(cr));
          break;
        }
      }
    }
    return absl::OkStatus();
  }

  absl::Status AddFunction(const Function* f) {
    std::optional<ProcId> proc_id;
    if (f->proc().has_value()) {
      proc_id = proc_id_factory_.CreateProcId(
          /*parent=*/std::nullopt, f->proc().value(),
          /*count_as_new_instance=*/false);
    }
    std::vector<InvocationCalleeData> calls =
        type_info_->GetUniqueInvocationCalleeData(f);
    if (f->IsParametric() && calls.empty()) {
      VLOG(5) << "No calls to parametric proc " << f->name_def()->ToString();
      return DefaultHandler(f);
    }
    for (auto& callee_data : calls) {
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord cr,
          InvocationToConversionRecord(
              f, callee_data.invocation, callee_data.derived_type_info,
              callee_data.callee_bindings, callee_data.caller_bindings,
              // Parametric functions can never be top.
              /* is_top= */ !f->IsParametric() && f == top_, proc_id));
      records_.push_back(std::move(cr));
    }
    if (calls.empty()) {
      // We can still convert this function even though it's never been called.
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord cr,
          InvocationToConversionRecord(f, /* invocation= */ nullptr, type_info_,
                                       /* callee_bindings= */ ParametricEnv(),
                                       /* caller_bindings= */ ParametricEnv(),
                                       /* is_top= */ f == top_, proc_id));
      records_.push_back(std::move(cr));
    }
    return DefaultHandler(f);
  }

  absl::Status HandleFunction(const Function* f) override {
    if (f->tag() == FunctionTag::kProcInit ||
        f->tag() == FunctionTag::kProcConfig ||
        f->tag() == FunctionTag::kProcNext) {
      // TODO: https://github.com/google/xls/issues/1029 - remove module-level
      // proc functions.
      return absl::OkStatus();
    }

    return AddFunction(f);
  }

  absl::Status HandleTestFunction(const TestFunction* tf) override {
    if (!include_tests_) {
      return absl::OkStatus();
    }
    return DefaultHandler(tf);
  }

  absl::Status HandleQuickCheck(const QuickCheck* qc) override {
    Function* f = qc->fn();
    XLS_RET_CHECK(!f->IsParametric()) << f->ToString();
    return DefaultHandler(qc);
  }

  absl::Status HandleProc(const Proc* p) override {
    // Process config so it can use the spawns to identify the dependent procs
    // to convert.
    XLS_RETURN_IF_ERROR(DefaultHandler(&p->config()));

    if (top_ == &p->next() || !p->IsParametric()) {
      // "top" procs won't have spawns referencing them so they won't
      // otherwise be added to the list, so we have to manually do it here.

      // Similarly, if a proc is not parametric, while it might not have any
      // spawns, we still want to convert it.
      return AddFunction(&p->next());
    }
    return absl::OkStatus();
  }

  absl::Status HandleTestProc(const TestProc* tp) override {
    if (!include_tests_) {
      return absl::OkStatus();
    }
    return DefaultHandler(tp);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (auto child : node->GetChildren(false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  std::vector<ConversionRecord> records() { return std::move(records_); }

 private:
  Module* const module_;
  TypeInfo* const type_info_;
  const bool include_tests_;
  ProcIdFactory proc_id_factory_;
  AstNode* top_;

  std::vector<ConversionRecord> records_;
};

}  // namespace

// This function removes duplicate conversion records from a list.
// The input list is modified.
void RemoveFunctionDuplicates(std::vector<ConversionRecord>& ready) {
  for (auto iter_func = ready.begin(); iter_func != ready.end(); iter_func++) {
    const ConversionRecord& function_cr = *iter_func;
    for (auto iter_subject = iter_func + 1; iter_subject != ready.end();) {
      const ConversionRecord& subject_cr = *iter_subject;

      if (function_cr.f() == subject_cr.f()) {
        bool either_is_parametric =
            function_cr.f()->IsParametric() || subject_cr.f()->IsParametric();
        // If neither are parametric, then function identity comparison is
        // a sufficient test to eliminate detected duplicates.
        if (!either_is_parametric) {
          iter_subject = ready.erase(iter_subject);
          continue;
        }

        // If the functions are the same and they have the same parametric
        // environment, eliminate any duplicates.
        bool both_are_parametric =
            function_cr.f()->IsParametric() && subject_cr.f()->IsParametric();
        if (both_are_parametric &&
            function_cr.parametric_env() == subject_cr.parametric_env()) {
          iter_subject = ready.erase(iter_subject);
          continue;
        }
      }
      iter_subject++;
    }
  }
}

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecords(
    Module* module, TypeInfo* type_info, bool include_tests) {
  ProcIdFactory proc_id_factory;
  // TODO: https://github.com/google/xls/issues/2078 - properly set
  // top instead of setting to nullptr.
  ConversionRecordVisitor visitor(module, type_info, include_tests,
                                  proc_id_factory, /*top=*/nullptr);
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));

  std::vector<ConversionRecord> records = visitor.records();
  RemoveFunctionDuplicates(records);
  return records;
}

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecordsForEntry(
    std::variant<Proc*, Function*> entry, TypeInfo* type_info) {
  ProcIdFactory proc_id_factory;
  if (std::holds_alternative<Function*>(entry)) {
    Function* f = std::get<Function*>(entry);
    Module* m = f->owner();
    // We are only ever called for tests, so we set include_tests to
    // true, and make sure that this function is top.
    ConversionRecordVisitor visitor(m, type_info, /*include_tests=*/true,
                                    proc_id_factory, f);
    XLS_RETURN_IF_ERROR(m->Accept(&visitor));

    std::vector<ConversionRecord> records = visitor.records();
    RemoveFunctionDuplicates(records);
    return records;
  }

  Proc* p = std::get<Proc*>(entry);
  Module* m = p->owner();
  XLS_ASSIGN_OR_RETURN(TypeInfo * new_ti,
                       type_info->GetTopLevelProcTypeInfo(p));
  // We are only ever called for tests, so we set include_tests to true,
  // and make sure that this proc's next function is top.
  ConversionRecordVisitor visitor(m, new_ti, /*include_tests=*/true,
                                  proc_id_factory, &p->next());
  XLS_RETURN_IF_ERROR(m->Accept(&visitor));

  std::vector<ConversionRecord> records = visitor.records();
  RemoveFunctionDuplicates(records);
  return records;
}
}  // namespace xls::dslx
