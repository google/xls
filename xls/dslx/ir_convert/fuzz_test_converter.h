// Copyright 2026 The XLS Authors
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

#ifndef XLS_DSLX_IR_CONVERT_FUZZ_TEST_CONVERTER_H_
#define XLS_DSLX_IR_CONVERT_FUZZ_TEST_CONVERTER_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/attribute_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {

class FuzzTestConverter {
 public:
  FuzzTestConverter(Module* module, TypeInfo* type_info,
                    ImportData* import_data)
      : module_(module),
        current_type_info_(type_info),
        import_data_(import_data) {}

  absl::StatusOr<std::optional<AttributeData>> LowerFuzzTestDomains(
      const Function* node);

 private:
  absl::Status LowerDomainExpr(const Expr* expr,
                               PackageInterfaceProto::FuzzTestDomain& proto);
  absl::Status LowerRangeExpr(const Range* range_node,
                              PackageInterfaceProto::FuzzTestDomain& proto);

 private:
  const Module* module_;
  TypeInfo* current_type_info_;
  ImportData* import_data_;
};
}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_FUZZTEST_CONVERTER_H_
