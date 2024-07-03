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

#ifndef XLS_IR_CLONE_PACKAGE_H_
#define XLS_IR_CLONE_PACKAGE_H_

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"

namespace xls {

// Create a clone of the given package.
//
// Id numbers are not necessarily going to be identical, ordering of lists of
// channels/fucntions/procs/blocks etc might not be identical.
absl::StatusOr<std::unique_ptr<Package>> ClonePackage(
    Package* p, std::optional<std::string_view> name = std::nullopt);

}  // namespace xls

#endif  // XLS_IR_CLONE_PACKAGE_H_
