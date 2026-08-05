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

#include "xls/ir/struct.h"

namespace xls {

void StructDef::AddMember(const Type* type, std::string name) {
  members_.emplace_back(Member{type, std::move(name)});
}

std::string StructDef::DumpIr() const {
  std::string out;
  absl::StrAppend(&out, "struct ", name_, " {\n");
  for (const auto& [type, name] : members_) {
    absl::StrAppend(&out, "  ", name, ": ", type->ToString(), "\n");
  }
  absl::StrAppend(&out, "}");
  return out;
}

}  // namespace xls
