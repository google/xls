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

#include <algorithm>
#include <ios>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtin_stubs_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/conversion_record.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/public/status_macros.h"

namespace xls::dslx {

namespace {

// Makes the conversion record from the pieces.
absl::StatusOr<ConversionRecord> MakeConversionRecord(
    Function* f, Module* m, TypeInfo* type_info, const ParametricEnv& bindings,
    std::optional<ProcId> proc_id, bool is_top,
    std::unique_ptr<ConversionRecord> config_record = nullptr,
    std::optional<InterpValue> init_value = std::nullopt) {
  return ConversionRecord::Make(f, m, type_info, bindings, proc_id, is_top,
                                std::move(config_record),
                                std::move(init_value));
}

// An AstNodeVisitor that creates ConversionRecords from appropriate AstNodes
// like Function, QuickCheck, Proc, etc.
class ConversionRecordVisitor : public AstNodeVisitorWithDefault {
 public:
  ConversionRecordVisitor(Module* module, TypeInfo* type_info,
                          bool include_tests, ProcIdFactory proc_id_factory,
                          AstNode* top,
                          std::optional<ResolvedProcAlias> resolved_proc_alias,
                          std::vector<ConversionRecord>& records)
      : module_(module),
        type_info_(type_info),
        include_tests_(include_tests),
        proc_id_factory_(proc_id_factory),
        top_(top),
        resolved_proc_alias_(resolved_proc_alias),
        records_(records) {}

  absl::StatusOr<ConversionRecord> SpawnDataToConversionRecord(
      const SpawnData& spawn, ProcId proc_id) {
    VLOG(5) << "Making conversion record for SpawnData with proc: "
            << spawn.proc->identifier() << "; env: " << spawn.env.ToString()
            << "; init: " << spawn.init_value.ToHumanString()
            << "; config TI: " << std::hex << spawn.config_type_info
            << "; next TI: " << spawn.next_type_info;

    XLS_ASSIGN_OR_RETURN(
        ConversionRecord config_record,
        MakeConversionRecord(&spawn.proc->config(), spawn.proc->owner(),
                             spawn.config_type_info, spawn.env, proc_id,
                             /*is_top=*/false));
    XLS_ASSIGN_OR_RETURN(
        ConversionRecord next_record,
        MakeConversionRecord(
            &spawn.proc->next(), spawn.proc->owner(), spawn.next_type_info,
            spawn.env, proc_id,
            /*is_top=*/false,
            std::make_unique<ConversionRecord>(std::move(config_record)),
            spawn.init_value));
    return next_record;
  }

  TypeInfo* GetTypeInfo(const AstNode* node) {
    return node->owner() == module_
               ? type_info_
               : *type_info_->GetImportedTypeInfo(node->owner());
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

    std::vector<InvocationCalleeData> calls =
        type_info_->GetUniqueInvocationCalleeData(f);
    if (f->IsParametric() && calls.empty()) {
      VLOG(5) << "No calls to parametric function " << f->name_def()->ToString()
              << "; not traversing for dependencies.";
      return absl::OkStatus();
    }
    // Process the child nodes first, so that function invocations or proc
    // spawns that _we_ make are added to the list _before us_. This only
    // matters to invocations to functions outside our module; functions inside
    // our module should have already been added to the list.
    XLS_RETURN_IF_ERROR(DefaultHandler(f));

    if (f->IsParametric() && !include_tests_) {
      XLS_RETURN_IF_ERROR(CheckIfCalledOnlyFromTestCode(
          type_info_, calls, /*is_proc=*/false, f->identifier()));
    }
    if (calls.empty()) {
      // We can still convert this function even though it's never been called.
      // Make sure we are using the right type info for imported functions.
      TypeInfo* invocation_ti = GetTypeInfo(f);
      VLOG(5) << "Processing fn " << f->ToString();
      XLS_ASSIGN_OR_RETURN(ConversionRecord cr,
                           MakeConversionRecord(const_cast<Function*>(f),
                                                f->owner(), invocation_ti,
                                                /*bindings=*/ParametricEnv(),
                                                /*proc_id=*/std::nullopt,
                                                /*is_top=*/f == top_));
      records_.push_back(std::move(cr));
    }
    return absl::OkStatus();
  }

  absl::Status HandleInvocation(const Invocation* invocation) override {
    VLOG(5) << "HandleInvocation " << invocation->ToString();
    TypeInfo* invocation_owner_ti = GetTypeInfo(invocation);
    auto invocation_data = invocation_owner_ti->GetInvocationData(invocation);
    XLS_RET_CHECK(invocation_data.has_value())
        << " no root invocation data for " << invocation->ToString();
    const Function* f = (*invocation_data)->callee();
    if (f == nullptr || IsBuiltin(f)) {
      return DefaultHandler(invocation);
    }

    if (f->owner() != module_) {
      // Function is outside this module; get additional conversion records from
      // its invocation and add to our list of records.
      ConversionRecordVisitor visitor(module_, invocation_owner_ti,
                                      include_tests_, proc_id_factory_, top_,
                                      resolved_proc_alias_, records_);
      XLS_RETURN_IF_ERROR(f->Accept(&visitor));
    } else {
      XLS_RETURN_IF_ERROR(f->Accept(this));
    }
    std::vector<InvocationCalleeData> calls;
    for (const auto& [_, callee_data] :
         (*invocation_data)->env_to_callee_data()) {
      calls.push_back(callee_data);
    }
    // Sort the calls by callee bindings so that the order is deterministic.
    // This is primarily for testing purposes. If tests move away from
    // golden-file comparison, the sort can probably be removed.
    std::sort(calls.begin(), calls.end(),
              [](const InvocationCalleeData& a, const InvocationCalleeData& b) {
                return a.callee_bindings.ToString() <
                       b.callee_bindings.ToString();
              });

    for (const auto& callee_data : calls) {
      VLOG(5) << "Processing invocation " << callee_data.invocation->ToString();
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord cr,
          MakeConversionRecord(const_cast<Function*>(f), f->owner(),
                               callee_data.derived_type_info,
                               callee_data.callee_bindings,
                               /*proc_id=*/std::nullopt,
                               // Parametric functions can never be top.
                               /*is_top=*/!f->IsParametric() && f == top_));
      records_.push_back(std::move(cr));
    }
    // Process the children, specifically, to find invocations in parameters.
    return DefaultHandler(invocation);
  }

  absl::Status HandleSpawn(const Spawn* spawn) override {
    VLOG(5) << "HandleSpawn " << spawn->ToString();
    Invocation* invocation = spawn->config();

    TypeInfo* spawn_owner_ti = GetTypeInfo(spawn);
    auto invocation_data = spawn_owner_ti->GetInvocationData(invocation);
    XLS_RET_CHECK(invocation_data.has_value())
        << " no invocation data for " << invocation->ToString();

    const Function* config_fn = (*invocation_data)->callee();
    XLS_RET_CHECK(config_fn->proc().has_value());
    Proc* proc = config_fn->proc().value();
    if (config_fn->owner() == module_) {
      return proc->Accept(this);
    }
    // Proc is outside this module; get additional conversion records from
    // its spawning and add to our list of records.
    ConversionRecordVisitor visitor(module_, spawn_owner_ti, include_tests_,
                                    proc_id_factory_, top_,
                                    resolved_proc_alias_, records_);
    XLS_RETURN_IF_ERROR(proc->Accept(&visitor));
    return absl::OkStatus();
  }

  absl::Status HandleUnrollFor(const UnrollFor* unroll_for) override {
    std::vector<Expr*> unrolled = type_info_->GetAllUnrolledLoops(unroll_for);
    for (const auto& expr : unrolled) {
      XLS_RETURN_IF_ERROR(expr->Accept(this));
    }
    return absl::OkStatus();
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
    // Handle any calls inside function bodies.
    XLS_RETURN_IF_ERROR(DefaultHandler(next_fn));
    // This is required in order to process cross-module spawns; otherwise it
    // will never add procs from imported modules to the list of functions to
    // convert.
    XLS_RETURN_IF_ERROR(DefaultHandler(&p->config()));

    ProcId proc_id = proc_id_factory_.CreateProcId(
        /*parent=*/std::nullopt, const_cast<Proc*>(p),
        /*count_as_new_instance=*/false);
    if (top_ == next_fn && resolved_proc_alias_.has_value()) {
      proc_id.alias_name = resolved_proc_alias_->name;
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord config_record,
          MakeConversionRecord(
              const_cast<Function*>(&p->config()), top_->owner(),
              resolved_proc_alias_->config_type_info, resolved_proc_alias_->env,
              proc_id, /*is_top=*/false));
      // TODO: Set up the initial value
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord next_record,
          MakeConversionRecord(
              const_cast<Function*>(&p->next()), top_->owner(),
              resolved_proc_alias_->next_type_info, resolved_proc_alias_->env,
              proc_id, /*is_top=*/true,
              std::make_unique<ConversionRecord>(std::move(config_record))));
      records_.push_back(std::move(next_record));
      return absl::OkStatus();
    }

    TypeInfo* proc_owner_ti = GetTypeInfo(p);
    XLS_ASSIGN_OR_RETURN(std::vector<SpawnData> spawn_data,
                         proc_owner_ti->GetUniqueSpawns(p));
    if (p->IsParametric() && spawn_data.empty()) {
      VLOG(5) << "No calls to parametric proc " << p->name_def()->ToString();
      return absl::OkStatus();
    }
    for (const SpawnData& spawn : spawn_data) {
      if (!include_tests_ && spawn.test) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Parametric proc `%s` is only called from test code, but "
            "test conversion is disabled.",
            p->identifier()));
      }

      XLS_ASSIGN_OR_RETURN(ConversionRecord cr,
                           SpawnDataToConversionRecord(spawn, proc_id));
      records_.push_back(std::move(cr));
    }
    if (spawn_data.empty() && (top_ == next_fn || !p->IsParametric())) {
      // "top" procs won't have spawns referencing them so they won't
      // otherwise be added to the list, so we have to manually do it here.

      // Similarly, if a proc is not parametric, while it might not have any
      // spawns, we still want to convert it.

      // Get the initial value for this proc; since there might not be a spawn,
      // there isn't SpawnData for it.
      Expr* init_body = p->init().body();

      XLS_ASSIGN_OR_RETURN(InterpValue initial_value,
                           proc_owner_ti->GetConstExpr(init_body));

      VLOG(5) << "HandleProc: initial element "
              << initial_value.ToHumanString();
      if (top_ == nullptr) {
        // Pick this proc/function as top if there isn't one already.
        top_ = const_cast<Function*>(next_fn);
      }
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord cr,
          MakeConversionRecord(const_cast<Function*>(next_fn), p->owner(),
                               proc_owner_ti,
                               /*bindings=*/ParametricEnv(), proc_id,
                               /*is_top=*/top_ == next_fn,
                               /*config_record=*/nullptr, initial_value));
      records_.push_back(std::move(cr));
      return absl::OkStatus();
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

#define OK_HANDLER(TYPE) \
  absl::Status Handle##TYPE(const TYPE*) override { return absl::OkStatus(); }

  // These may have (built-in) function invocations in parametric expressions,
  // but they shouldn't be added to the list of functions to convert, since
  // they are required to be constexprs. So instead of calling DefaultHandler,
  // which would process the invocations, just return OK.

  // keep-sorted start
  OK_HANDLER(ConstAssert)
  OK_HANDLER(EnumDef)
  OK_HANDLER(ParametricBinding)
  OK_HANDLER(ProcDef)
  OK_HANDLER(StructDef)
  OK_HANDLER(TypeAlias)
  // keep-sorted end
#undef DEFAULT_HANDLE

  absl::Status DefaultHandler(const AstNode* node) override {
    for (auto child : node->GetChildren(false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status CheckIfCalledOnlyFromTestCode(
      TypeInfo* type_info, const std::vector<InvocationCalleeData>& calls,
      bool is_proc, std::string_view identifier) {
    bool called_from_outside_test = false;
    for (auto& callee_data : calls) {
      std::optional<const InvocationData*> invocation_data =
          type_info->GetInvocationData(callee_data.invocation);
      XLS_RET_CHECK(invocation_data.has_value())
          << " no root invocation data for "
          << callee_data.invocation->ToString();
      if (!IsTestFn((*invocation_data)->caller())) {
        called_from_outside_test = true;
        break;
      }
    }
    if (!called_from_outside_test) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Parametric %s `%s` is only called from test code, but "
          "test conversion is disabled.",
          is_proc ? "proc" : "function", identifier));
    }
    return absl::OkStatus();
  }
  Module* const module_;
  TypeInfo* const type_info_;
  const bool include_tests_;
  ProcIdFactory proc_id_factory_;
  AstNode* top_;

  // The proc alias that was used to specify the top proc, if any.
  std::optional<ResolvedProcAlias> resolved_proc_alias_;

  std::vector<ConversionRecord>& records_;
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
  std::vector<ConversionRecord> records;
  ConversionRecordVisitor visitor(
      module, type_info, include_tests, proc_id_factory, /*top=*/nullptr,
      /*resolved_proc_alias=*/std::nullopt, records);
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));

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
    std::vector<ConversionRecord> records;
    // We are only ever called for tests, so we set include_tests to
    // true, and make sure that this function is top.
    ConversionRecordVisitor visitor(
        m, type_info, /*include_tests=*/true, proc_id_factory, f,
        /*resolved_proc_alias=*/std::nullopt, records);
    XLS_RETURN_IF_ERROR(f->Accept(&visitor));

    return RemoveFunctionDuplicates(records);
  }

  Proc* p = std::get<Proc*>(entry);
  Module* m = p->owner();
  XLS_ASSIGN_OR_RETURN(TypeInfo * new_ti,
                       type_info->GetTopLevelProcTypeInfo(p));
  std::vector<ConversionRecord> records;
  // We are only ever called for tests, so we set include_tests to true,
  // and make sure that this proc's next function is top.
  ConversionRecordVisitor visitor(m, new_ti, /*include_tests=*/true,
                                  proc_id_factory, &p->next(),
                                  resolved_proc_alias, records);
  XLS_RETURN_IF_ERROR(p->Accept(&visitor));

  return RemoveFunctionDuplicates(records);
}
}  // namespace xls::dslx
