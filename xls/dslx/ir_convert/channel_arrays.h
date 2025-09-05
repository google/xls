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

#ifndef XLS_DSLX_IR_CONVERT_CHANNEL_ARRAYS_H_
#define XLS_DSLX_IR_CONVERT_CHANNEL_ARRAYS_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Helper class for managing channel arrays.
class ChannelArrays {
 public:
  ChannelArrays(ImportData* const import_data, TypeInfo* const type_info,
                const ParametricEnv& bindings)
      : import_data_(import_data), type_info_(type_info), bindings_(bindings) {}

  // Creates all the suffixes for elements in a channel array with the given
  // `dims`.
  absl::StatusOr<std::vector<std::string>> CreateAllArrayElementSuffixes(
      const std::vector<Expr*>& dims) const;

 private:
  ImportData* const import_data_;
  TypeInfo* const type_info_;
  const ParametricEnv& bindings_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_CHANNEL_ARRAYS_H_
