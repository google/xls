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

#ifndef XLS_TOOLS_OPT_FLAGS_H_
#define XLS_TOOLS_OPT_FLAGS_H_

#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/tools/opt_flags.pb.h"

namespace xls {

absl::StatusOr<OptFlagsProto> GetOptFlags(
    std::optional<std::string_view> ir_path);

}  // namespace xls

#endif  // XLS_TOOLS_OPT_FLAGS_H_
