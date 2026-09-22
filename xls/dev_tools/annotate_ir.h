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

#ifndef XLS_DEV_TOOLS_ANNOTATE_IR_H_
#define XLS_DEV_TOOLS_ANNOTATE_IR_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/package.h"

namespace xls {

enum class AnnotatorKind {
  kNone,
  kVisibility,
  kDelay,
  kCriticalPath,
};

// Parses an annotator name string, e.g. "none", "delay", into an AnnotatorKind.
absl::StatusOr<AnnotatorKind> ParseAnnotatorKind(
    std::string_view annotator_str);

// Dumps a FunctionBase with the given IrAnnotator applied such that the
// resulting text remains valid, parseable XLS IR:
// - Param annotations are `// <annotation>` comments above the the function.
// - Non-param node annotations are trailing inline comments `// <annotation>`.
std::string AnnotateFunctionBaseIr(FunctionBase* fb,
                                   const IrAnnotator& annotator);

// Dumps a Package with the given IrAnnotator applied to all FunctionBases in
// the package, producing valid, parseable XLS IR.
std::string AnnotateIr(Package* package, const IrAnnotator& annotator);

// Dumps a Package using a configured AnnotatorKind producing parse-able XLS IR.
absl::StatusOr<std::string> AnnotateIr(Package* package, AnnotatorKind kind,
                                       std::string_view delay_model = "unit");

}  // namespace xls

#endif  // XLS_DEV_TOOLS_ANNOTATE_IR_H_
