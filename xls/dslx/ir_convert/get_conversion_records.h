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

#ifndef XLS_DSLX_IR_CONVERT_GET_CONVERSION_RECORDS_H_
#define XLS_DSLX_IR_CONVERT_GET_CONVERSION_RECORDS_H_

#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/ir_convert/conversion_record.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Returns order for functions to be converted to IR.
//
// Args:
//  module: Module to convert the (non-parametric) functions for.
//  type_info: Mapping from node to type.
//  include_tests: should test functions be included.
absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecords(
    Module* module, TypeInfo* type_info, bool include_tests);

// Returns order for functions/procs to be converted to IR. For testing only.
// Args:
//  entry: Proc or Function to start from (the top)
//  type_info: Mapping from node to type.
absl::StatusOr<std::vector<ConversionRecord>> GetConversionRecordsForEntry(
    std::variant<Proc*, Function*> entry, TypeInfo* type_info);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_GET_CONVERSION_RECORDS_H_
