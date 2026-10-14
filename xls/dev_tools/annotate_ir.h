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

#include <array>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/package.h"

namespace xls {

enum class AnnotatorKind {
  kNone,
  kDelay,
  kCriticalPath,
};

inline constexpr std::array<AnnotatorKind, 4> kAllAnnotatorKinds = {
    AnnotatorKind::kNone,
    AnnotatorKind::kDelay,
    AnnotatorKind::kCriticalPath,
};

template <typename Sink>
void AbslStringify(Sink& sink, AnnotatorKind kind) {
  static absl::NoDestructor<
      absl::flat_hash_map<AnnotatorKind, std::string_view>>
      annotators({
          {AnnotatorKind::kNone, "none"},
          {AnnotatorKind::kDelay, "delay"},
          {AnnotatorKind::kCriticalPath, "critical_path"},
      });
  if (auto it = annotators->find(kind); it != annotators->end()) {
    absl::Format(&sink, "%s", it->second);
    return;
  }
  absl::Format(&sink, "<unknown AnnotatorKind %d>", static_cast<int>(kind));
}

// Dumps a Package using a configured AnnotatorKind, wrapping that annotator in
// a comment formatting annotator to produce parse-able XLS IR, and a package
// annotator to route to function-specific annotator instances.
absl::StatusOr<std::string> AnnotateIr(AnnotatorKind kind, Package* package,
                                       std::string_view delay_model = "unit");

// Dumps a Package using a custom annotator, wrapping that annotator in a
// comment formatting annotator to produce parse-able XLS IR.
absl::StatusOr<std::string> AnnotateIr(const IrAnnotator& annotator,
                                       Package* package);

}  // namespace xls

#endif  // XLS_DEV_TOOLS_ANNOTATE_IR_H_
