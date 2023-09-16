// Copyright 2023 The XLS Authors
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

#ifndef XLS_IR_FOREIGN_FUNCTION_H_
#define XLS_IR_FOREIGN_FUNCTION_H_

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/value.h"

namespace xls {
absl::StatusOr<ForeignFunctionData> ForeignFunctionDataCreateFromTemplate(
    std::string_view annotation);

// Helper to substitute some ffi template parameters with constant values.
// Many constant values might only be available in earlier stages of processing;
// this helps to update the template and replace template values already known.
//
// This helper allows partially fill the template with these values, while the
// remaining template parameters are left as-is.
// Parameter is an optional ForeignFunctionData; if nullopt, no replacmenets
// will be done.
// (Used in the FunctionConverter)
class FfiPartialValueSubstituteHelper {
 public:
  explicit FfiPartialValueSubstituteHelper(
      std::optional<ForeignFunctionData> ffi)
      : ffi_(std::move(ffi)) {}

  // Set a named value. If the name is referenced in the ForeingFunctionData
  // template, it will be replaced with the value.
  void SetNamedValue(std::string_view name, const Value& value);

  // Get updated template with all the substituion values replaced.
  // If the originally provided std::optional was empty, returns nullopt.
  std::optional<ForeignFunctionData> GetUpdatedFfiData() const;

 private:
  const std::optional<ForeignFunctionData> ffi_;
  absl::flat_hash_map<std::string, std::string> substitutions_;
};
}  // namespace xls

#endif  // XLS_IR_FOREIGN_FUNCTION_H_
