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

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
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
                          AstNode* top,
                          std::optional<ResolvedProcAlias> resolved_proc_alias)
      : module_(module),
        type_info_(type_info),
        include_tests_(include_tests),
        proc_id_factory_(proc_id_factory),
        top_(top),
        resolved_proc_alias_(resolved_proc_alias) {}

  absl::StatusOr<ConversionRecord> InvocationToConversionRecord(
      const Function* f, const Invocation* invocation,
      TypeInfo* instantiation_type_info, ParametricEnv callee_bindings,
      ParametricEnv caller_bindings, bool is_top,
      std::optional<ProcId> proc_id) {
    if (invocation != nullptr) {
      VLOG(5) << "InvocationToConversionRecord processing invocation "
              << invocation->ToString();
    } else {
      VLOG(5) << "InvocationToConversionRecord processing fn " << f->ToString();
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
              MakeConversionRecord(const_cast<Function*>(&config_fn),
                                   f->owner(), *config_type_info,
                                   callee_bindings, proc_id, config_invocation,
                                   // config functions can never be 'top'
                                   /*is_top=*/false));
          config_record = std::make_unique<ConversionRecord>(std::move(cr));
          break;
        }
      }
    }

    XLS_ASSIGN_OR_RETURN(
        ConversionRecord cr,
        MakeConversionRecord(const_cast<Function*>(f), f->owner(),
                             instantiation_type_info, callee_bindings, proc_id,
                             invocation, is_top, std::move(config_record)));
    return cr;
  }

  absl::Status AddFunction(const Function* f) {
    VLOG(5) << "AddFunction " << f->ToString();
    std::optional<ProcId> proc_id;
    if (f->proc().has_value()) {
      proc_id = proc_id_factory_.CreateProcId(
          /*parent=*/std::nullopt, f->proc().value(),
          /*count_as_new_instance=*/false);
    }
    // Process the child nodes first, so that function invocations or proc
    // spawns that _we_ make are added to the list _before us_. This only
    // matters to invocations to functions outside our module; functions inside
    // our module should have already been added to the list.
    XLS_RETURN_IF_ERROR(DefaultHandler(f));

    std::vector<InvocationCalleeData> calls =
        type_info_->GetUniqueInvocationCalleeData(f);
    if (f->IsParametric() && calls.empty()) {
      VLOG(5) << "No calls to parametric function "
              << f->name_def()->ToString();
      return absl::OkStatus();
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
      // Make sure we are using the right type info for imported functions.
      TypeInfo* invocation_ti =
          f->owner() == module_ ? type_info_
                                : *type_info_->GetImportedTypeInfo(f->owner());
      XLS_ASSIGN_OR_RETURN(ConversionRecord cr,
                           InvocationToConversionRecord(
                               f, /* invocation= */ nullptr, invocation_ti,
                               /* callee_bindings= */ ParametricEnv(),
                               /* caller_bindings= */ ParametricEnv(),
                               /* is_top= */ f == top_, proc_id));
      records_.push_back(std::move(cr));
    }
    return DefaultHandler(f);
  }

  absl::Status HandleFunction(const Function* f) override {
    VLOG(5) << "HandleFunction " << f->ToString();
    if (f->tag() == FunctionTag::kProcInit ||
        f->tag() == FunctionTag::kProcConfig ||
        f->tag() == FunctionTag::kProcNext) {
      // TODO: https://github.com/google/xls/issues/1029 - remove module-level
      // proc functions.
      return absl::OkStatus();
    }

    return AddFunction(f);
  }

  absl::Status HandleInvocation(const Invocation* invocation) override {
    VLOG(5) << "HandleInvocation " << invocation->ToString();
    auto root_invocation_data = type_info_->GetRootInvocationData(invocation);
    XLS_RET_CHECK(root_invocation_data.has_value());
    const InvocationData* invocation_data = *root_invocation_data;
    const Function* f = invocation_data->callee();
    if (f == nullptr || IsBuiltin(f)) {
      return DefaultHandler(invocation);
    }

    if (f->owner() == module_) {
      // Since this function is inside this module, we will convert this
      // function, so there's no need to do any more processing here.
      return absl::OkStatus();
    }
    return HandleFunction(f);
  }

  absl::Status HandleSpawn(const Spawn* spawn) override {
    VLOG(5) << "HandleSpawn " << spawn->ToString();
    Invocation* invocation = spawn->config();

    auto root_invocation_data = type_info_->GetRootInvocationData(invocation);
    XLS_RET_CHECK(root_invocation_data.has_value());

    const InvocationData* invocation_data = *root_invocation_data;
    const Function* config_fn = invocation_data->callee();
    if (config_fn->owner() == module_) {
      // Since this proc is inside this module, We will convert this proc, so
      // there's no need to do any more processing here.
      return absl::OkStatus();
    }
    XLS_RET_CHECK(config_fn->proc().has_value());
    Proc* proc = config_fn->proc().value();
    return HandleProc(proc);
  }

  absl::Status HandleTestFunction(const TestFunction* tf) override {
    if (!include_tests_) {
      VLOG(5) << "include_tests_ is false; skipping HandleTestFunction "
              << tf->ToString();
      return absl::OkStatus();
    }
    VLOG(5) << "HandleTestFunction " << tf->ToString();
    return DefaultHandler(tf);
  }

  absl::Status HandleQuickCheck(const QuickCheck* qc) override {
    VLOG(5) << "HandleQuickCheck " << qc->ToString();
    Function* f = qc->fn();
    XLS_RET_CHECK(!f->IsParametric()) << f->ToString();
    return DefaultHandler(qc);
  }

  absl::Status HandleProc(const Proc* p) override {
    VLOG(5) << "HandleProc " << p->ToString();
    const Function* next_fn = &p->next();

    if (top_ == next_fn && resolved_proc_alias_.has_value()) {
      ProcId proc_id = proc_id_factory_.CreateProcId(
          /*parent=*/std::nullopt, const_cast<Proc*>(p),
          /*count_as_new_instance=*/false);
      proc_id.alias_name = resolved_proc_alias_->name;
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord config_record,
          MakeConversionRecord(
              const_cast<Function*>(&p->config()), top_->owner(),
              resolved_proc_alias_->config_type_info, resolved_proc_alias_->env,
              proc_id, /*invocation=*/nullptr,
              /*is_top=*/false));
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord next_record,
          MakeConversionRecord(
              const_cast<Function*>(&p->next()), top_->owner(),
              resolved_proc_alias_->next_type_info, resolved_proc_alias_->env,
              proc_id, /*invocation=*/nullptr,
              /*is_top=*/true,
              std::make_unique<ConversionRecord>(std::move(config_record))));
      records_.push_back(std::move(next_record));
      return absl::OkStatus();
    }

    // This is required in order to process cross-module spawns; otherwise it
    // will never add procs from imported modules to the list of functions to
    // convert.
    XLS_RETURN_IF_ERROR(DefaultHandler(&p->config()));
    if (p->IsParametric()) {
      std::optional<ProcId> proc_id = proc_id_factory_.CreateProcId(
          /*parent=*/std::nullopt, const_cast<Proc*>(p),
          /*count_as_new_instance=*/false);

      std::vector<InvocationCalleeData> next_calls =
          type_info_->GetUniqueInvocationCalleeData(next_fn);
      for (auto& callee_data : next_calls) {
        XLS_ASSIGN_OR_RETURN(
            ConversionRecord cr,
            InvocationToConversionRecord(
                next_fn, callee_data.invocation, callee_data.derived_type_info,
                callee_data.callee_bindings, callee_data.caller_bindings,
                // Since this proc is being spawned, it's certainly not top.
                /* is_top= */ false, proc_id));
        records_.push_back(std::move(cr));
      }
    }
    if (top_ == next_fn || !p->IsParametric()) {
      // "top" procs won't have spawns referencing them so they won't
      // otherwise be added to the list, so we have to manually do it here.

      // Similarly, if a proc is not parametric, while it might not have any
      // spawns, we still want to convert it.
      return AddFunction(next_fn);
    }
    return DefaultHandler(p);
  }

  absl::Status HandleTestProc(const TestProc* tp) override {
    if (!include_tests_) {
      VLOG(5) << "include_tests_ is false; skipping HandleTestProc "
              << tp->ToString();
      return absl::OkStatus();
    }
    VLOG(5) << "HandleTestProc " << tp->ToString();
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

  // The proc alias that was used to specify the top proc, if any.
  std::optional<ResolvedProcAlias> resolved_proc_alias_;

  std::vector<ConversionRecord> records_;
};

}  // namespace

// Filters duplicate conversion records from the given vector and returns a new
// vector without duplicates.
std::vector<ConversionRecord> RemoveFunctionDuplicates(
    std::vector<ConversionRecord>& ready) {
  absl::flat_hash_set<std::pair<Function*, ParametricEnv>> records;
  std::vector<ConversionRecord> result;
  for (auto& record : ready) {
    if (records.emplace(record.f(), record.parametric_env()).second) {
      result.push_back(std::move(record));
    }
  }
  return result;
}

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecords(
    Module* module, TypeInfo* type_info, bool include_tests) {
  ProcIdFactory proc_id_factory;
  // TODO: https://github.com/google/xls/issues/2078 - properly set
  // top instead of setting to nullptr.
  ConversionRecordVisitor visitor(module, type_info, include_tests,
                                  proc_id_factory, /*top=*/nullptr,
                                  /*resolved_proc_alias=*/std::nullopt);
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));

  std::vector<ConversionRecord> records = visitor.records();
  return RemoveFunctionDuplicates(records);
}

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecordsForEntry(
    std::variant<Proc*, Function*> entry, TypeInfo* type_info,
    std::optional<ResolvedProcAlias> resolved_proc_alias) {
  ProcIdFactory proc_id_factory;
  if (std::holds_alternative<Function*>(entry)) {
    XLS_RET_CHECK(!resolved_proc_alias.has_value());
    Function* f = std::get<Function*>(entry);
    Module* m = f->owner();
    // We are only ever called for tests, so we set include_tests to
    // true, and make sure that this function is top.
    ConversionRecordVisitor visitor(m, type_info, /*include_tests=*/true,
                                    proc_id_factory, f,
                                    /*resolved_proc_alias=*/std::nullopt);
    XLS_RETURN_IF_ERROR(m->Accept(&visitor));

    std::vector<ConversionRecord> records = visitor.records();
    return RemoveFunctionDuplicates(records);
  }

  Proc* p = std::get<Proc*>(entry);
  Module* m = p->owner();
  XLS_ASSIGN_OR_RETURN(TypeInfo * new_ti,
                       type_info->GetTopLevelProcTypeInfo(p));
  // We are only ever called for tests, so we set include_tests to true,
  // and make sure that this proc's next function is top.
  ConversionRecordVisitor visitor(m, new_ti, /*include_tests=*/true,
                                  proc_id_factory, &p->next(),
                                  resolved_proc_alias);
  XLS_RETURN_IF_ERROR(m->Accept(&visitor));

  std::vector<ConversionRecord> records = visitor.records();
  return RemoveFunctionDuplicates(records);
}
}  // namespace xls::dslx
