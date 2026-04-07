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

#ifndef XLS_DEV_TOOLS_LINK_TO_SOURCE_H_
#define XLS_DEV_TOOLS_LINK_TO_SOURCE_H_

#include <optional>
#include <string>

#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xls {

// Get a link to the source code associated with the given source location.
//
// TODO(allight): Let the link format be configurable with a flag.
std::optional<std::string> LinkToSource(const SourceLocation& loc,
                                        const Package* package);

}  // namespace xls

#endif  // XLS_DEV_TOOLS_LINK_TO_SOURCE_H_
