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

#ifndef XLS_PUBLIC_IR_PARSER_H_
#define XLS_PUBLIC_IR_PARSER_H_

#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/public/ir.h"

namespace xls {

// Parses a given input string as an XLS IR "package".
//
// **Background:** packages are the building blocks that XLS deals with, similar
// to a "module" in other programming environments, but "module" means other
// things for hardware design, so the name "package" was chosen to avoid
// ambiguity. A hardware block can be contained entirely in a single package, or
// can be built out of a set of packages with a defined entry point (usually
// referred to as "top").
//
// Args:
//  input_string: The contents of the IR package. This is "IR text", described
//    in https://google.github.io/xls/ir_semantics/
//  filename: The filename for these IR contents, e.g. "/path/to/my.ir" -- this
//    is used for positional error messages that occur during IR text parsing.
//
// Returns:
//  The built package object, or an error.
absl::StatusOr<std::unique_ptr<Package>> ParsePackage(
    std::string_view input_string, std::optional<std::string_view> filename);

// Parses a function into an existing package.
absl::StatusOr<Function*> ParseFunctionIntoPackage(
    std::string_view function_string, Package* package);

}  // namespace xls

#endif  // XLS_PUBLIC_IR_PARSER_H_
