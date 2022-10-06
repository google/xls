// Copyright 2020 The XLS Authors
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
#ifndef XLS_JIT_JIT_WRAPPER_GENERATOR_H_
#define XLS_JIT_JIT_WRAPPER_GENERATOR_H_

#include <filesystem>
#include <string>

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
//   wrapper_namespace: C++ namespace to put the wrapper in.
//   header_path: Path to the eventual location of the class header.
// TODO(rspringer): 2020-08-19 Add support for non-opt IR.
GeneratedJitWrapper GenerateJitWrapper(
    const Function& function, std::string_view class_name,
    std::string_view wrapper_namespace,
    const std::filesystem::path& header_path,
    const std::filesystem::path& genfiles_path);

}  // namespace xls

#endif  // XLS_JIT_JIT_WRAPPER_GENERATOR_H_
