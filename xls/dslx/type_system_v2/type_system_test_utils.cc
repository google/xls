// Copyright 2024 The XLS Authors
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

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

absl::StatusOr<std::string> TypeInfoToString(const TypeInfo& ti,
                                             const FileTable& file_table) {
  if (ti.dict().empty()) {
    return "";
  }
  std::vector<std::string> strings;
  for (const auto& [node, type] : ti.dict()) {
    Span span = node->GetSpan().has_value() ? *node->GetSpan() : Span::Fake();
    strings.push_back(absl::Substitute("span: $0, node: `$1`, type: $2",
                                       span.ToString(file_table),
                                       node->ToString(), type->ToString()));
  }
  absl::c_sort(strings);
  return strings.size() == 1
             ? strings[0]
             : absl::Substitute("\n$0\n", absl::StrJoin(strings, "\n"));
}

absl::StatusOr<std::string> TypeInfoToString(const TypecheckedModule& module) {
  return TypeInfoToString(*module.type_info, *module.module->file_table());
}

}  // namespace xls::dslx
