// Copyright 2022 The XLS Authors
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

#include "xls/public/ir_parser.h"

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/package.h"

namespace xls {

absl::StatusOr<std::unique_ptr<Package>> ParsePackage(
    std::string_view input_string, std::optional<std::string_view> filename) {
  return Parser::ParsePackage(input_string, filename);
}

absl::StatusOr<Function*> ParseFunctionIntoPackage(
    std::string_view function_string, Package* package) {
  return Parser::ParseFunction(function_string, package);
}

}  // namespace xls
