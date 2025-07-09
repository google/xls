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
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
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
  return ConversionRecord::Make(f, /*invocation=*/nullptr, m, type_info,
                                bindings,
                                /*orig_callees=*/{}, std::nullopt, is_top);
}
}  // namespace

absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecords(
    Module* module, TypeInfo* type_info, bool include_tests) {
  std::vector<ConversionRecord> records;
  auto handle_function = [&](Function* f) -> absl::Status {
    if (!f->IsParametric()) {
      XLS_ASSIGN_OR_RETURN(
          ConversionRecord cr,
          MakeConversionRecord(f, module, type_info, ParametricEnv()));
      records.push_back(cr);
    } else {
      // We want one ConversionRecord per *unique* parametric binding of
      // this function.
      std::vector<ParametricEnv> unique_bindings;
      for (auto& invocation : type_info->GetInvocationData(f)) {
        for (auto& callee_data : invocation.env_to_callee_data()) {
          ParametricEnv bindings = callee_data.second.callee_bindings;
          if (std::find(unique_bindings.begin(), unique_bindings.end(),
                        bindings) == unique_bindings.end()) {
            // Only add the record if we haven't processed this binding yet.
            XLS_ASSIGN_OR_RETURN(
                ConversionRecord cr,
                MakeConversionRecord(f, module, type_info, bindings));
            records.push_back(cr);
            unique_bindings.push_back(bindings);
          }
        }
      }
    }

    return absl::OkStatus();
  };

  for (ModuleMember member : module->top()) {
    absl::Status status = absl::visit(
        Visitor{
            [&](Function* f) -> absl::Status {
              if (f->tag() != FunctionTag::kNormal) {
                // Skip proc functions because they'll be handled elsewhere.
                return absl::OkStatus();
              }

              return handle_function(f);
            },
            [&](QuickCheck* quickcheck) -> absl::Status {
              Function* function = quickcheck->fn();
              XLS_RET_CHECK(!function->IsParametric()) << function->ToString();
              XLS_ASSIGN_OR_RETURN(
                  ConversionRecord cr,
                  MakeConversionRecord(function, module, type_info,
                                       ParametricEnv()));

              records.push_back(cr);
              return absl::OkStatus();
            },
            [&](Proc* p) { return handle_function(&p->next()); },
            [&](TestFunction* test) {
              if (!include_tests) {
                return absl::OkStatus();
              }
              return handle_function(&test->fn());
            },
            [](auto*) { return absl::OkStatus(); },
        },
        member);
    XLS_RETURN_IF_ERROR(status);
  }

  return records;
}

}  // namespace xls::dslx
