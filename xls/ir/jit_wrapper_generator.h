// Copyright 2020 Google LLC
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
#ifndef THIRD_PARTY_XLS_IR_JIT_WRAPPER_GENERATOR_H_
#define THIRD_PARTY_XLS_IR_JIT_WRAPPER_GENERATOR_H_

#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "xls/ir/function.h"

namespace xls {

// Holds the results of generating a JIT wrapper (see below).
struct GeneratedJitWrapper {
  std::string header;
  std::string source;
};

// Generates a header and source file for a class that "wraps" JIT creation and
// invocation for the given function.
// Args:
//   function: The function for which to generate the wrapper.
//   class_name: The name to give to the generated class.
//   header_path: Path to the eventual location of the class header.
GeneratedJitWrapper GenerateJitWrapper(
    const Function& function, const std::string& class_name,
    const std::filesystem::path& header_path);

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_JIT_WRAPPER_GENERATOR_H_
