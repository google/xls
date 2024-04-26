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

#include "xls/codegen/lint_annotate.h"

#include "absl/strings/str_format.h"
#include "xls/codegen/vast.h"
#include "xls/ir/source_location.h"

namespace xls::verilog {

void ScopedLintDisable::ToggleCheck(Lint flag, bool on) {
  section_->Add<Comment>(
      SourceInfo(),
      absl::StrFormat("lint_%s %s", on ? "on" : "off", LintToString(flag)));
}

}  // namespace xls::verilog
