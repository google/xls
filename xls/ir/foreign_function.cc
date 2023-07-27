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

#include "xls/ir/foreign_function.h"

#include "absl/status/statusor.h"
#include "xls/ir/code_template.h"
#include "xls/ir/foreign_function_data.pb.h"

namespace xls {

absl::StatusOr<ForeignFunctionData> ForeignFunctionDataCreateFromTemplate(
    std::string_view annotation) {
  absl::StatusOr<CodeTemplate> parse_result = CodeTemplate::Create(annotation);
  if (!parse_result.ok()) {
    return parse_result.status();
  }
  // We just pass the template along as string, but we validated it worked
  ForeignFunctionData result;
  result.set_code_template(annotation);
  return result;
}

}  // namespace xls
