// Copyright 2021 The XLS Authors
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

#include "xls/codegen/codegen_pass.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace xls::verilog {

std::string CodegenPassUnit::DumpIr() const {
  // Dump the Package and metadata. The metadata is commented out ('//') so the
  // output is parsable.
  std::string out =
      absl::StrFormat("// Generating code for proc: %s\n\n", block->name());
  absl::StrAppend(&out, package->DumpIr());
  if (signature.has_value()) {
    absl::StrAppend(&out, "\n\n// Module signature:\n");
    for (auto line : absl::StrSplit(signature->ToString(), '\n')) {
      absl::StrAppend(&out, "// ", line, "\n");
    }
  }
  return out;
}

}  // namespace xls::verilog
