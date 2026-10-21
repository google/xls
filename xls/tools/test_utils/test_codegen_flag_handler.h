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

#ifndef XLS_TOOLS_TEST_UTILS_TEST_CODEGEN_FLAG_HANDLER_H_
#define XLS_TOOLS_TEST_UTILS_TEST_CODEGEN_FLAG_HANDLER_H_

#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/declare.h"
#include "xls/codegen/codegen_options.h"

ABSL_DECLARE_FLAG(std::string, test_codegen_flag);

namespace xls {

struct TestCodegenOptionExtension : public verilog::CodegenOptionExtension {
  static constexpr std::string_view kExtensionName = "test_codegen_extension";

  explicit TestCodegenOptionExtension(std::string value)
      : value(std::move(value)) {}

  std::string_view extension_name() const override { return kExtensionName; }

  std::string value;
};

}  // namespace xls

#endif  // XLS_TOOLS_TEST_UTILS_TEST_CODEGEN_FLAG_HANDLER_H_
