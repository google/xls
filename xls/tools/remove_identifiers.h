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

#ifndef XLS_TOOLS_REMOVE_IDENTIFIERS_H_
#define XLS_TOOLS_REMOVE_IDENTIFIERS_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
namespace xls {

struct StripOptions {
  std::string new_package_name;
  bool strip_location_info = true;
  bool strip_node_names = true;
  bool strip_function_names = true;
  bool strip_chan_names = true;
  bool strip_reg_names = true;
};
// Strip the requested data from the package and return a new one.
absl::StatusOr<std::unique_ptr<Package>> StripPackage(
    Package* source, const StripOptions& options);

}  // namespace xls

#endif  // XLS_TOOLS_REMOVE_IDENTIFIERS_H_
