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
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/public/status_macros.h"

namespace xls::dslx {

namespace {

// Makes the conversion record from the pieces.
absl::StatusOr<ConversionRecord> MakeConversionRecord(
    Function* f, Module* m, TypeInfo* type_info, const ParametricEnv& bindings,
    bool is_top = false) {
  return ConversionRecord::Make(
      f, /*invocation=*/nullptr, m, type_info, bindings,
      /*orig_callees=*/{}, /*proc_id=*/std::nullopt, is_top);
}

// An AstNodeVisitor that creates ConversionRecords from appropriate AstNodes
// like Function, QuickCheck, Proc, etc.
class ConversionRecordVisitor : public AstNodeVisitorWithDefault {
 public:
  ConversionRecordVisitor(Module* module, TypeInfo* type_info,
                          bool include_tests)
      : module_(module), type_info_(type_info), include_tests_(include_tests) {}

  absl::Status HandleFunction(const Function* f) override {
    if (f->tag() == FunctionTag::kProcInit ||
        f->tag() == FunctionTag::kProcConfig) {
      // Don't include init or config, since they will always be converted while
      // converting 'next'.
      return absl::OkStatus();
    }

    if (f->IsParametric()) {
      // We want one ConversionRecord per *unique* parametric binding of
      // this function.
      for (auto& callee_data : type_info_->GetUniqueInvocationCalleeData(f)) {
        XLS_ASSIGN_OR_RETURN(
            ConversionRecord cr,
            MakeConversionRecord(const_cast<Function*>(f), module_,
                                 callee_data.derived_type_info,
                                 callee_data.callee_bindings));
        records_.push_back(cr);
      }
      return DefaultHandler(f);
    }
    XLS_ASSIGN_OR_RETURN(ConversionRecord cr,
                         MakeConversionRecord(const_cast<Function*>(f), module_,
                                              type_info_, ParametricEnv()));
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

  std::vector<ConversionRecord> records() { return records_; }

 private:
  Module* module_;
  TypeInfo* type_info_;
  bool include_tests_;

  std::vector<ConversionRecord> records_;
};

}  // namespace

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecords(
    Module* module, TypeInfo* type_info, bool include_tests) {
  ConversionRecordVisitor visitor(module, type_info, include_tests);
  XLS_RETURN_IF_ERROR(module->Accept(&visitor));

  return visitor.records();
}

}  // namespace xls::dslx
