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

#include <optional>
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
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/public/status_macros.h"

namespace xls::dslx {

namespace {

// Makes the conversion record from the pieces.
absl::StatusOr<ConversionRecord> MakeConversionRecord(
    Function* f, Module* m, TypeInfo* type_info, const ParametricEnv& bindings,
    std::optional<ProcId> proc_id, const Invocation* invocation, bool is_top) {
  return ConversionRecord::Make(f, invocation, m, type_info, bindings, proc_id,
                                is_top);
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

  absl::Status HandleFunction(const Function* f) override {
    if (f->tag() == FunctionTag::kProcInit ||
        f->tag() == FunctionTag::kProcConfig) {
      // Don't include init or config, since they will always be converted while
      // converting 'next'.
      return absl::OkStatus();
    }

    std::optional<ProcId> proc_id;
    if (f->proc().has_value()) {
      proc_id = proc_id_factory_.CreateProcId(
          /*parent=*/std::nullopt, f->proc().value(),
          /*count_as_new_instance=*/false);
    }
    if (f->IsParametric()) {
      // We want one ConversionRecord per *unique* parametric binding of
      // this function.
      XLS_RET_CHECK(!type_info_->GetUniqueInvocationCalleeData(f).empty())
          << "Cannot lower a parametric proc without an invocation";
      for (auto& callee_data : type_info_->GetUniqueInvocationCalleeData(f)) {
        const Invocation* invocation = callee_data.invocation;
        // TODO: davidplass - change this to gather invocations from *spawns*
        // instead of *functions*. This will allow function_converter to emit
        // multiple IR procs with the same parametrics but different config
        // parameters. Then, HandleFunction would only have to handle procs
        // that are not spawned explicitly, like test or top procs.
        if (f->tag() == FunctionTag::kProcNext) {
          // If this is a proc next function, find the corresponding config
          // invocation (spawn) with the same parametrics and put it in the
          // conversion record.
          invocation = nullptr;
          const Function& config_fn = f->proc().value()->config();
          for (InvocationCalleeData& config_callee_data :
               type_info_->GetUniqueInvocationCalleeData(&config_fn)) {
            if (config_callee_data.callee_bindings ==
                callee_data.callee_bindings) {
              invocation = config_callee_data.invocation;
              break;
            }
          }
          // Note, it's possible there is no config invocation if it's a
          // top proc or some other reason.
        }
        XLS_ASSIGN_OR_RETURN(
            ConversionRecord cr,
            MakeConversionRecord(const_cast<Function*>(f), module_,
                                 callee_data.derived_type_info,
                                 callee_data.callee_bindings, proc_id,
                                 invocation,
                                 // parametric functions can never be 'top'
                                 /*is_top=*/false));
        records_.push_back(cr);
      }
      return DefaultHandler(f);
    }

    const Invocation* invocation = nullptr;
    // If this is a proc next function, find the corresponding config
    // invocation (spawn) and put it in the conversion record. Take the first
    // one because there is no way to disambiguate them at this point.
    if (f->tag() == FunctionTag::kProcNext) {
      Function& config_fn = f->proc().value()->config();
      std::vector<InvocationCalleeData> all_callee_data =
          type_info_->GetUniqueInvocationCalleeData(&config_fn);
      if (!all_callee_data.empty()) {
        if (all_callee_data.size() > 1) {
          // Warning
          VLOG(2) << "More than 1 non-parametric invocation of "
                  << config_fn.identifier();
        }
        invocation = all_callee_data[0].invocation;
      }
    }
    XLS_ASSIGN_OR_RETURN(
        ConversionRecord cr,
        MakeConversionRecord(const_cast<Function*>(f), module_, type_info_,
                             ParametricEnv(), proc_id, invocation, f == top_));
    records_.push_back(cr);
    return DefaultHandler(f);
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
    // Do not process children, because we'll process the `next` function
    // as a top-level function in the module.
    // TODO: https://github.com/google/xls/issues/1029 - remove module-level
    // proc functions.
    return absl::OkStatus();
  }

  absl::Status HandleTestProc(const TestProc* tp) override {
    // Do not process children, because we'll process the next function
    // as a top-level function in the module.
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (auto child : node->GetChildren(false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  std::vector<ConversionRecord>& records() { return records_; }

 private:
  Module* const module_;
  TypeInfo* const type_info_;
  const bool include_tests_;
  ProcIdFactory proc_id_factory_;
  AstNode* top_;

  std::vector<ConversionRecord> records_;
};

}  // namespace

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecords(
    Module* module, TypeInfo* type_info, bool include_tests) {
  ProcIdFactory proc_id_factory;
  // TODO: https://github.com/google/xls/issues/2078 - properly set
  // top instead of setting to nullptr.
  ConversionRecordVisitor visitor(module, type_info, include_tests,
                                  proc_id_factory, /*top=*/nullptr);
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));

  return visitor.records();
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
    return visitor.records();
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
  return visitor.records();
}
}  // namespace xls::dslx
