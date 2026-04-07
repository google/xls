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

#include "xls/dev_tools/link_to_source.h"

#include <optional>
#include <string>

#include "absl/strings/str_format.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xls {

std::optional<std::string> LinkToSource(const SourceLocation& loc,
                                        const Package* package) {
  auto it = package->fileno_to_name().find(loc.fileno());
  if (it == package->fileno_to_name().end()) {
    return std::nullopt;
  }
  return absl::StrFormat("file://%s", it->second);
}

}  // namespace xls
