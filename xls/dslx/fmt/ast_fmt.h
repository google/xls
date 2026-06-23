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
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/ir/channel.h"

namespace xls::dslx {

// TODO: davidplass - Move this class to its own file after all the
// Expr-descendants are migrated to it.
class Formatter {
  friend class FmtExprVisitor;

 public:
  Formatter(Comments& comments, DocArena& arena)
      : comments_(comments), arena_(arena) {}

  virtual ~Formatter() = default;

  Comments& comments() { return comments_; }
  const Comments& comments() const { return comments_; }
  DocArena& arena() { return arena_; }
  const DocArena& arena() const { return arena_; }

  // Each `FormatXXX` method creates a pretty-printable document from the given
  // AST node `n`.

  // keep-sorted start
  virtual DocRef FormatBlockedExprLeader(const Expr& e);
  virtual DocRef FormatExpr(const Expr& n, bool suppress_parens = false);
  virtual DocRef FormatFunction(const Function& n, bool is_test = false);
  virtual DocRef FormatLet(const Let& n, bool trailing_semi);
  virtual DocRef FormatStatement(const Statement& n, bool trailing_semi);
  virtual DocRef FormatTypeAnnotation(const TypeAnnotation& n);
  virtual DocRef FormatVerbatimNode(const VerbatimNode& n);
  virtual absl::StatusOr<DocRef> FormatModule(const Module& n);
  // keep-sorted end

 protected:
  enum class Joiner : uint8_t {
    kCommaSpace,
    kCommaBreak1,

    kCommaHardlineTrailingCommaAlways,

    // Separates via a comma and break1, but groups the element with its
    // delimiter. This is useful when we're packing member elements that we want
    // to be reflowed across lines.
    //
    // Note that, in this mode, if we span multiple lines, we'll put a trailing
    // comma as well.
    kCommaBreak1AsGroupTrailingCommaOnBreak,
    kCommaBreak1AsGroupTrailingCommaAlways,
    kCommaBreak1AsGroupNoTrailingComma,

    kSpaceBarBreak,
    kHardLine,
  };

  std::optional<DocRef> FormatCommentsBetween(
      std::optional<Pos> start, const Pos& limit,
      std::optional<Span>* last_comment_span = nullptr);

  template <typename T>
  DocRef FormatImplOrTrait(const T& n, Keyword keyword,
                           DocRef name_or_struct_ref);

  // keep-sorted start
  virtual DocRef FormatAllOnesMacro(const AllOnesMacro& n);
  virtual DocRef FormatArray(const Array& n);
  virtual DocRef FormatArrayTypeAnnotation(const ArrayTypeAnnotation& n);
  virtual DocRef FormatAttr(const Attr& n);
  virtual DocRef FormatAttribute(const Attribute& n);
  virtual DocRef FormatBinop(const Binop& n);
  virtual DocRef FormatBlock(const StatementBlock& n, bool add_curls = true,
                             bool force_multiline = false);
  virtual DocRef FormatBuiltinTypeAnnotation(const BuiltinTypeAnnotation& n);
  virtual DocRef FormatCast(const Cast& n);
  virtual DocRef FormatChannelConfig(const ChannelConfig& n);
  virtual DocRef FormatChannelDecl(const ChannelDecl& n);
  virtual DocRef FormatChannelTypeAnnotation(const ChannelTypeAnnotation& n);
  virtual DocRef FormatColonRef(const ColonRef& n);
  virtual DocRef FormatConditional(const Conditional& n);
  virtual DocRef FormatConstAssert(const ConstAssert& n);
  virtual DocRef FormatConstFor(const ConstFor& n);
  virtual DocRef FormatConstantDef(const ConstantDef& n);
  virtual DocRef FormatEnumDef(const EnumDef& n);
  virtual DocRef FormatEnumMember(const EnumMember& n);
  virtual DocRef FormatFor(const For& n);
  virtual DocRef FormatFormatMacro(const FormatMacro& n);
  virtual DocRef FormatFunctionRef(const FunctionRef& n);
  virtual DocRef FormatImpl(const Impl& n);
  virtual DocRef FormatImplMember(const ImplMember& n);
  virtual DocRef FormatImport(const Import& n);
  virtual DocRef FormatIndex(const Index& n);
  virtual DocRef FormatIndexRhs(const IndexRhs& n);
  virtual DocRef FormatInvocation(const Invocation& n);
  virtual DocRef FormatLambda(const Lambda& n);
  virtual DocRef FormatMatch(const Match& n);
  virtual DocRef FormatModuleMember(const ModuleMember& n);
  virtual DocRef FormatNameDef(const NameDef& n);
  virtual DocRef FormatNameDefTree(const NameDefTree& n);
  virtual DocRef FormatNameDefTreeLeaf(const NameDefTree::Leaf& n);
  virtual DocRef FormatNameRef(const NameRef& n);
  virtual DocRef FormatNumber(const Number& n);
  virtual DocRef FormatParametricBinding(const ParametricBinding& n);
  virtual DocRef FormatParametricBindingPtr(const ParametricBinding* n);
  virtual DocRef FormatParams(absl::Span<const Param* const> params);
  virtual DocRef FormatProc(const Proc& n, bool is_test = false);
  virtual DocRef FormatProcAlias(const ProcAlias& n);
  virtual DocRef FormatProcDef(const ProcDef& n);
  virtual DocRef FormatProcMember(const ProcMember& n);
  virtual DocRef FormatQuickCheck(const QuickCheck& n);
  virtual DocRef FormatRange(const Range& n);
  virtual DocRef FormatRestOfTuple(const RestOfTuple& n);
  virtual DocRef FormatSlice(const Slice& n);
  virtual DocRef FormatSpawn(const Spawn& n);
  virtual DocRef FormatSplatStructInstance(const SplatStructInstance& n);
  virtual DocRef FormatStatementBlock(const StatementBlock& n);
  virtual DocRef FormatString(const String& n);
  virtual DocRef FormatStructDef(const StructDef& n);
  virtual DocRef FormatStructDefBase(
      const StructDefBase& n, Keyword keyword,
      const std::optional<std::string>& extern_type_name);
  virtual DocRef FormatStructInstance(const StructInstance& n);
  virtual DocRef FormatSumDef(const SumDef& n);
  virtual DocRef FormatSumInstance(const SumInstance& n);
  virtual DocRef FormatSumVariantPayloadPattern(
      const SumVariantPayloadPattern& n);
  virtual DocRef FormatTestFunction(const TestFunction& n);
  virtual DocRef FormatTestProc(const TestProc& n);
  virtual DocRef FormatTrait(const Trait& n);
  virtual DocRef FormatTupleIndex(const TupleIndex& n);
  virtual DocRef FormatTupleTypeAnnotation(const TupleTypeAnnotation& n);
  virtual DocRef FormatTypeAlias(const TypeAlias& n);
  virtual DocRef FormatTypeRef(const TypeRef& n);
  virtual DocRef FormatTypeRefTypeAnnotation(const TypeRefTypeAnnotation& n);
  virtual DocRef FormatTypeVariableTypeAnnotation(
      const TypeVariableTypeAnnotation& n);
  virtual DocRef FormatUnop(const Unop& n);
  virtual DocRef FormatUse(const Use& n);
  virtual DocRef FormatWidthSlice(const WidthSlice& n);
  virtual DocRef FormatWildcardPattern(const WildcardPattern& n);
  virtual DocRef FormatXlsTuple(const XlsTuple& n);
  virtual DocRef FormatZeroMacro(const ZeroMacro& n);
  // keep-sorted end

  DocRef Format(const Expr* n);
  DocRef Format(const TypeAnnotation* n);
  DocRef Format(const NameDefTree* n);
  DocRef FormatBreakBody(const Array& n);
  DocRef FormatBreakRest(const StructInstance& n);
  Pos FormatCollectInlineComments(const Pos& prev_limit,
                                  const Pos& last_entity_pos,
                                  std::vector<DocRef>& pieces,
                                  std::optional<Span> last_comment_span);
  std::optional<DocRef> FormatCommentsNested(const Pos start, const Pos limit);
  DocRef FormatConditionalMultiline(const Conditional& n);
  std::optional<DocRef> FormatExplicitParametrics(
      absl::Span<const ExprOrType> parametrics);
  DocRef FormatExprOrType(const ExprOrType& n);
  DocRef FormatFlatBody(const Array& n);
  DocRef FormatFlatRest(const StructInstance& n);
  DocRef FormatForLoopBase(Keyword keyword, const ForLoopBase& n,
                           bool is_const_for);
  DocRef FormatForLoopBaseLeader(Keyword keyword, DocRef names_ref,
                                 const ForLoopBase& n, bool is_const_for);
  DocRef FormatJoinWithAttr(std::optional<DocRef> attr, DocRef rest);
  DocRef FormatJoinWithAttrs(absl::Span<const DocRef> attrs, DocRef rest);
  DocRef FormatMakeArrayLeader(const Array& n);
  DocRef FormatMakeConditionalTest(const Conditional& n);
  DocRef FormatMatchArm(const MatchArm& n);
  DocRef FormatParametricArg(const ExprOrType& n);
  DocRef FormatSingleStatementBlockInline(const StatementBlock& n,
                                          bool add_curls);
  DocRef FormatStructLeader(const TypeAnnotation* struct_ref);
  void FormatStructMembers(const StructDefBase& n, std::vector<DocRef>& pieces);
  DocRef FormatStructMembersBreak(
      Span struct_span,
      absl::Span<const std::pair<std::string, Expr*>> members);
  DocRef FormatStructMembersFlat(
      absl::Span<const std::pair<std::string, Expr*>> members);
  bool FormatAppendSumCommentsBetween(const Pos& start_pos,
                                      const Pos& limit_pos,
                                      std::vector<DocRef>& pieces);
  void FormatSumStructMembers(absl::Span<StructMemberNode* const> members,
                              const Span& body_span,
                              std::vector<DocRef>& pieces,
                              bool place_internal_comments = false);
  void FormatSumTuplePayloadMembers(const SumVariant& variant,
                                    const Span& payload_span,
                                    std::vector<DocRef>& pieces);
  DocRef FormatTuple(const XlsTuple& n);
  DocRef FormatTupleWithoutComments(const XlsTuple& n);
  template <typename T>
  DocRef FormatJoin(absl::Span<const T> items, Joiner joiner,
                    const std::function<DocRef(const T&)>& fmt);

  virtual bool IsBlockedExprNoLeader(const Expr& e);
  virtual bool IsBlockedExprWithLeader(const Expr& e);

  Comments& comments_;
  DocArena& arena_;
};

inline constexpr int64_t kDslxDefaultTextWidth = 100;

// Auto-formatting entry points.

// Performs a reflow-capable formatting of module `m` with standard line width,
// but with the ability to disable formatting for specific ranges of text.
absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    int64_t text_width = kDslxDefaultTextWidth);

// Variant which takes a `Formatter`, allowing the caller to customize the
// behavior.
absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Formatter& formatter,
                                    int64_t text_width = kDslxDefaultTextWidth);

// Performs a reflow-capable formatting of module `m` with standard line width,
// for the actual `content` but with the ability to disable formatting for
// specific ranges of text.
absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Comments& comments,
                                    std::string contents,
                                    int64_t text_width = kDslxDefaultTextWidth);

// Variant which takes a `Formatter`, allowing the caller to customize the
// behavior.
absl::StatusOr<std::string> AutoFmt(VirtualizableFilesystem& vfs,
                                    const Module& m, Formatter& formatter,
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
