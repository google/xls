// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_FMT_AST_FMT_H_
#define XLS_DSLX_FMT_AST_FMT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

// TODO: davidplass - Move this class to its own file after all the
// Expr-descendants are migrated to it.
class Formatter {
 public:
  Formatter(const Comments& comments, DocArena& arena)
      : comments_(comments), arena_(arena) {}

  // keep-sorted start
  // Each `Format` method creates a pretty-printable document from the given AST
  // node `n`.
  DocRef Format(const Function& n);
  DocRef Format(const Expr& n);
  DocRef Format(const Let& n, bool trailing_semi);
  DocRef Format(const Module& n);
  // If `trailing_semi` is true, then a trailing semicolon will also be emitted.
  DocRef Format(const Statement& n, bool trailing_semi);
  DocRef Format(const VerbatimNode& n);
  // keep-sorted end

 private:
  // keep-sorted start
  DocRef Format(const ConstAssert& n);
  DocRef Format(const ConstantDef& n);
  DocRef Format(const EnumDef& n);
  DocRef Format(const EnumMember& n);
  DocRef Format(const Impl& n);
  DocRef Format(const ImplMember& n);
  DocRef Format(const Import& n);
  DocRef Format(const ModuleMember& n);
  DocRef Format(const ParametricBinding& n);
  DocRef Format(const ParametricBinding* n);
  DocRef Format(const Proc& n);
  DocRef Format(const ProcDef& n);
  DocRef Format(const ProcMember& n);
  DocRef Format(const QuickCheck& n);
  DocRef Format(const StructDef& n);
  DocRef Format(const TestFunction& n);
  DocRef Format(const TestProc& n);
  DocRef Format(const TypeAlias& n);
  DocRef FormatParams(absl::Span<const Param* const> params);
  DocRef FormatStructDefBase(
      const StructDefBase& n, Keyword keyword,
      const std::optional<std::string>& extern_type_name);
  // keep-sorted end

  const Comments& comments_;
  DocArena& arena_;
};

inline constexpr int64_t kDslxDefaultTextWidth = 100;

// Auto-formatting entry points.
//
// Performs a reflow-capable formatting of module `m` with standard line width,
// but with the ability to disable formatting for specific ranges of text.
absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    int64_t text_width = kDslxDefaultTextWidth);

// Performs a reflow-capable formatting of module `m` with standard line width,
// for the actual `content` but with the ability to disable formatting for
// specific ranges of text.
absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    std::string contents,
                                    int64_t text_width = kDslxDefaultTextWidth);

// If we fail the postcondition we return back the data we used to detect that
// the postcondition was violated.
struct AutoFmtPostconditionViolation {
  std::string original_transformed;
  std::string autofmt_transformed;
};

// Checks whether the auto-formatting process looks "opportunistically sound" --
// that is, this will not hold true for all examples, but it'll hold true for a
// bunch of them, and so can be a useful debugging tool.
//
// It's difficult to come up with a /simple/ postcondition for the
// auto-formatter because it does some cleanup transformations based on the
// grammar, and we want this to be a simple linear / regexp style check on the
// flattened text, so we can't account for all the transforms that the
// autoformatter may perform. Still, it's useful in testing or debugging
// scenarios where we know none of those constructs / situations are present.
std::optional<AutoFmtPostconditionViolation>
ObeysAutoFmtOpportunisticPostcondition(std::string_view original,
                                       std::string_view autofmt);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FMT_AST_FMT_H_
