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

#ifndef XLS_CODEGEN_LINT_ANNOTATE_H_
#define XLS_CODEGEN_LINT_ANNOTATE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "xls/codegen/vast.h"

namespace xls::verilog {

enum class Lint {
  kSignedType,
  kMultiply,  // XLS emits multiply operations.
};

// Note: acts as a shim for different lint environments.
class ScopedLintDisable {
 public:
  ScopedLintDisable(ModuleSection* section, std::vector<Lint> flags)
      : section_(section), flags_(std::move(flags)) {
    for (Lint flag : flags_) {
      ToggleCheck(flag, /*on=*/false);
    }
  }

  ~ScopedLintDisable() {
    for (auto it = flags_.rbegin(); it != flags_.rend(); ++it) {
      ToggleCheck(*it, /*on=*/true);
    }
  }

 private:
  void ToggleCheck(Lint flag, bool on);

  ModuleSection* section_;
  const std::vector<Lint> flags_;
};

inline std::string LintToString(Lint flag) {
  switch (flag) {
    case Lint::kSignedType:
      return "SIGNED_TYPE";
    case Lint::kMultiply:
      return "MULTIPLY";
  }
  return absl::StrFormat("<invalid Lint(%d)>", static_cast<int>(flag));
}

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_LINT_ANNOTATE_H_
