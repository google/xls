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

#include "xls/contrib/xlscc/cc_parser.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/ASTConsumer.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclBase.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/Diagnostic.h"
#include "clang/include/clang/Basic/DiagnosticIDs.h"
#include "clang/include/clang/Basic/DiagnosticLex.h"
#include "clang/include/clang/Basic/DiagnosticSema.h"
#include "clang/include/clang/Basic/FileSystemOptions.h"
#include "clang/include/clang/Basic/IdentifierTable.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/ParsedAttrInfo.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Basic/TokenKinds.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendAction.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Lex/PPCallbacks.h"
#include "clang/include/clang/Lex/Pragma.h"
#include "clang/include/clang/Lex/Preprocessor.h"
#include "clang/include/clang/Lex/Token.h"
#include "clang/include/clang/Sema/ParsedAttr.h"
#include "clang/include/clang/Sema/Sema.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "llvm/include/llvm/ADT/APInt.h"
#include "llvm/include/llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/VirtualFileSystem.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/file/filesystem.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/ir/channel.h"
#include "xls/ir/fileno.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xlscc {
namespace {

void GenerateAnnotation(clang::Preprocessor& PP, std::string_view name,
                        const clang::Token& after,
                        const absl::Span<const clang::Token>& arguments) {
  const int64_t num_tokens =
      5 + (arguments.empty() ? 0 : (arguments.size() + 2));

  std::unique_ptr<clang::Token[]> tokens =
      std::make_unique<clang::Token[]>(num_tokens);
  int64_t token_index = 0;
  auto add_token = [&](clang::tok::TokenKind kind, unsigned length,
                       clang::SourceLocation location,
                       std::string_view identifier = "") {
    CHECK_LT(token_index, num_tokens);
    tokens[token_index].setKind(kind);
    tokens[token_index].setLength(length);
    tokens[token_index].setLocation(location);
    if (!identifier.empty()) {
      tokens[token_index].setIdentifierInfo(PP.getIdentifierInfo(identifier));
    }
    token_index++;
  };
  auto insert_token = [&](clang::Token token) {
    CHECK_LT(token_index, num_tokens);
    tokens[token_index++] = std::move(token);
  };

  add_token(clang::tok::l_square, /*length=*/0, after.getLocation());
  add_token(clang::tok::l_square, /*length=*/0, after.getLocation());

  add_token(clang::tok::identifier, after.getLength(), after.getLocation(),
            /*identifier=*/name);

  if (!arguments.empty()) {
    add_token(clang::tok::l_paren, /*length=*/0, after.getEndLoc());
    for (const clang::Token& token : arguments) {
      insert_token(token);
    }
    add_token(clang::tok::r_paren, /*length=*/0, arguments.back().getEndLoc());
  }

  add_token(clang::tok::r_square, /*length=*/0, after.getEndLoc());
  add_token(clang::tok::r_square, /*length=*/0, after.getEndLoc());

  CHECK_EQ(token_index, num_tokens);
  PP.EnterTokenStream(std::move(tokens), num_tokens,
                      /*DisableMacroExpansion=*/false,
                      /*IsReinject=*/false);
}

std::optional<std::vector<clang::Token>> LexPragmaArgs(
    clang::Preprocessor& PP, clang::Token& firstToken,
    int64_t num_token_arguments, std::string_view pragma_name) {
  const clang::PresumedLoc presumed_loc =
      PP.getSourceManager().getPresumedLoc(firstToken.getLocation());

  std::vector<clang::Token> toks;
  PP.LexTokensUntilEOF(&toks);

  if (toks.size() != num_token_arguments) {
    PP.Diag(firstToken.getLocation(), clang::diag::err_pragma_message)
        << absl::StrFormat("#pragma %s must have %i arguments. At %s:%i",
                           pragma_name, num_token_arguments,
                           presumed_loc.getFilename(), presumed_loc.getLine());
    return std::nullopt;
  }

  return toks;
}

}  // namespace

class LibToolVisitor : public clang::RecursiveASTVisitor<LibToolVisitor> {
 public:
  explicit LibToolVisitor(clang::CompilerInstance& CI, CCParser& parser)
      : ast_context_(&(CI.getASTContext())), parser_(parser) {}
  virtual ~LibToolVisitor() = default;
  virtual bool VisitVarDecl(clang::VarDecl* decl) {
    return parser_.LibToolVisitVarDecl(decl);
    return true;
  }
  virtual bool VisitFunctionDecl(clang::FunctionDecl* func) {
    return parser_.LibToolVisitFunction(func);
  }

 private:
  clang::ASTContext* ast_context_;
  CCParser& parser_;
};

struct XlsTopAttrInfo : public clang::ParsedAttrInfo {
  XlsTopAttrInfo() {
    // GNU-style __attribute__(("hls_top")) and C++/C23-style [[hls_top]] and
    // [[xlscc::hls_top]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_top"},
        {clang::ParsedAttr::AS_C23, "hls_top"},
        {clang::ParsedAttr::AS_CXX11, "hls_top"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_top"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    // This attribute appertains to functions only.
    if (!isa<clang::FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    // Attach an annotate attribute to the Decl.
    D->addAttr(clang::AnnotateAttr::Create(S.Context, "hls_top", nullptr, 0,
                                           Attr.getRange()));
    return AttributeApplied;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsTopAttrInfo> hls_top(
    "hls_top", "Marks a function as the top function of an HLS block.");

struct XlsBlockAttrInfo : public clang::ParsedAttrInfo {
  XlsBlockAttrInfo() {
    // GNU-style __attribute__(("hls_block")) and C++/C23-style [[hls_block]]
    // and [[xlscc::hls_block]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_block"},
        {clang::ParsedAttr::AS_C23, "hls_block"},
        {clang::ParsedAttr::AS_CXX11, "hls_block"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_block"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    // This attribute appertains to functions only.
    if (!isa<clang::FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    // Attach an annotate attribute to the Decl.
    D->addAttr(clang::AnnotateAttr::Create(S.Context, "hls_block", nullptr, 0,
                                           Attr.getRange()));
    return AttributeApplied;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsBlockAttrInfo> hls_block(
    "hls_block",
    "Marks a function its own HLS block, below the top in the hierarchy.");

struct XlsAllowDefaultPadAttrInfo : public clang::ParsedAttrInfo {
  XlsAllowDefaultPadAttrInfo() {
    // GNU-style __attribute__(("hls_array_allow_default_pad")) and
    // C++/C23-style [[hls_array_allow_default_pad]] and
    // [[xlscc::hls_array_allow_default_pad]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_array_allow_default_pad"},
        {clang::ParsedAttr::AS_C23, "hls_array_allow_default_pad"},
        {clang::ParsedAttr::AS_CXX11, "hls_array_allow_default_pad"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_array_allow_default_pad"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    // This attribute appertains to variables only.
    if (!isa<clang::VarDecl>(D) && !isa<clang::FieldDecl>(D)) {
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "variables";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    // Attach an annotate attribute to the Decl.
    D->addAttr(clang::AnnotateAttr::Create(
        S.Context, "hls_array_allow_default_pad", nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsAllowDefaultPadAttrInfo>
    hls_array_allow_default_pad("hls_array_allow_default_pad",
                                "Marks a variable declaration as allowing "
                                "default zero padding to full size.");

struct XlsSyntheticIntAttrInfo : public clang::ParsedAttrInfo {
  XlsSyntheticIntAttrInfo() {
    // GNU-style __attribute__(("hls_synthetic_int")) and
    // C++/C23-style [[hls_synthetic_int]] and
    // [[xlscc::hls_synthetic_int]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_synthetic_int"},
        {clang::ParsedAttr::AS_C23, "hls_synthetic_int"},
        {clang::ParsedAttr::AS_CXX11, "hls_synthetic_int"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_synthetic_int"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    // This attribute appertains to structs only.
    if (!clang::isa<clang::RecordDecl>(D)) {
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "structs/classes";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    // Attach an annotate attribute to the Decl.
    D->addAttr(clang::AnnotateAttr::Create(S.Context, "hls_synthetic_int",
                                           nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsSyntheticIntAttrInfo>
    hls_synthetic_int(
        "hls_synthetic_int",
        "Marks a struct or class declaration as implementing an integer type.");

struct XlsNoTupleAttrInfo : public clang::ParsedAttrInfo {
  XlsNoTupleAttrInfo() {
    // GNU-style __attribute__(("hls_synthetic_int")) and
    // C++/C23-style [[hls_synthetic_int]] and
    // [[xlscc::hls_synthetic_int]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_no_tuple"},
        {clang::ParsedAttr::AS_C23, "hls_no_tuple"},
        {clang::ParsedAttr::AS_CXX11, "hls_no_tuple"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_no_tuple"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    // This attribute appertains to structs only.
    if (!clang::isa<clang::RecordDecl>(D)) {
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "structs/classes";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    // Attach an annotate attribute to the Decl.
    D->addAttr(clang::AnnotateAttr::Create(S.Context, "hls_no_tuple", nullptr,
                                           0, Attr.getRange()));
    return AttributeApplied;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsNoTupleAttrInfo> hls_no_tuple(
    "hls_no_tuple",
    "Marks a struct or class as represented as the type of its first, and "
    "only, member.");

struct XlsPipelineInitIntervalAttrInfo : public clang::ParsedAttrInfo {
  XlsPipelineInitIntervalAttrInfo() {
    // GNU-style __attribute__(("hls_pipeline_init_interval")) and C++/C23-style
    // [[hls_pipeline_init_interval]] and [[xlscc::hls_pipeline_init_interval]]
    // supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_pipeline_init_interval"},
        {clang::ParsedAttr::AS_C23, "hls_pipeline_init_interval"},
        {clang::ParsedAttr::AS_CXX11, "hls_pipeline_init_interval"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_pipeline_init_interval"}};
    Spellings = S;
    NumArgs = 1;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    if (isa<clang::LabelDecl>(D)) {
      // Allow application to labels; enables pass-through to the following
      // statement.
      return true;
    }
    S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
        << Attr << Attr.isRegularKeywordAttribute() << "loop statements";
    return false;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    if (!isa<clang::LabelDecl>(D)) {
      return NotHandled;
    }

    clang::Attr* Result;
    if (!CreateAttr(S, Attr, Result)) {
      return AttributeNotApplied;
    }
    D->addAttr(Result);
    return AttributeApplied;
  }

  bool diagAppertainsToStmt(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Stmt* St) const override {
    if (!isa<clang::ForStmt>(St) && !isa<clang::WhileStmt>(St) &&
        !isa<clang::DoStmt>(St)) {
      // This attribute appertains to loop statements only.
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "loop statements";
      return false;
    }
    return true;
  }

  AttrHandling handleStmtAttribute(clang::Sema& S, clang::Stmt* St,
                                   const clang::ParsedAttr& Attr,
                                   class clang::Attr*& Result) const override {
    return CreateAttr(S, Attr, Result) ? AttributeApplied : AttributeNotApplied;
  }

 private:
  bool CreateAttr(clang::Sema& S, const clang::ParsedAttr& Attr,
                  clang::Attr*& Result) const {
    CHECK_EQ(Attr.getNumArgs(), 1);

    auto invalid_argument = [&]() {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "the argument to the 'hls_pipeline_init_interval' attribute must be "
          "an integer >= 1");
      S.Diag(Attr.getLoc(), ID);
      return false;
    };

    clang::Expr* args[1];
    if (!Attr.isArgExpr(0)) {
      return invalid_argument();
    }
    args[0] = Attr.getArgAsExpr(0);
    if (!args[0]->isIntegerConstantExpr(S.Context) ||
        !args[0]->EvaluateKnownConstInt(S.Context).isStrictlyPositive()) {
      return invalid_argument();
    }
    Result =
        clang::AnnotateAttr::Create(S.Context, "hls_pipeline_init_interval",
                                    args, /*ArgsSize=*/1, Attr.getRange());
    return true;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsPipelineInitIntervalAttrInfo>
    hls_pipeline_init_interval("hls_pipeline_init_interval",
                               "Marks a loop to be pipelined, and how.");

struct XlsUnrollAttrInfo : public clang::ParsedAttrInfo {
  XlsUnrollAttrInfo() {
    // GNU-style __attribute__(("hls_unroll")) and C++/C23-style [[hls_unroll]]
    // and [[xlscc::hls_unroll]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_unroll"},
        {clang::ParsedAttr::AS_C23, "hls_unroll"},
        {clang::ParsedAttr::AS_CXX11, "hls_unroll"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_unroll"}};
    Spellings = S;
    OptArgs = 1;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    if (isa<clang::LabelDecl>(D)) {
      // Allow application to labels; enables pass-through to the following
      // statement.
      return true;
    }
    S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
        << Attr << Attr.isRegularKeywordAttribute() << "loop statements";
    return false;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    if (!isa<clang::LabelDecl>(D)) {
      return NotHandled;
    }

    clang::Attr* Result;
    if (!CreateAttr(S, Attr, Result)) {
      return AttributeNotApplied;
    }
    D->addAttr(Result);
    return AttributeApplied;
  }

  bool diagAppertainsToStmt(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Stmt* St) const override {
    if (!isa<clang::ForStmt>(St) && !isa<clang::WhileStmt>(St) &&
        !isa<clang::DoStmt>(St)) {
      // This attribute appertains to loop statements only.
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "loop statements";
      return false;
    }
    return true;
  }

  AttrHandling handleStmtAttribute(clang::Sema& S, clang::Stmt* St,
                                   const clang::ParsedAttr& Attr,
                                   clang::Attr*& Result) const override {
    return CreateAttr(S, Attr, Result) ? AttributeApplied : AttributeNotApplied;
  }

 private:
  bool CreateAttr(clang::Sema& S, const clang::ParsedAttr& Attr,
                  clang::Attr*& Result) const {
    if (Attr.getNumArgs() > 1) {
      S.Diag(Attr.getLoc(), clang::diag::err_attribute_too_many_arguments)
          << Attr << 1;
      return false;
    }

    clang::Expr* args[1];
    int64_t num_args = Attr.getNumArgs();
    if (Attr.getNumArgs() > 0) {
      if (Attr.isArgExpr(0)) {
        args[0] = Attr.getArgAsExpr(0);
      } else {
        std::string_view arg_name = Attr.getArgAsIdent(0)->Ident->getName();
        if (absl::EqualsIgnoreCase(arg_name, "no")) {
          const clang::PresumedLoc presumed_loc =
              S.getSourceManager().getPresumedLoc(Attr.getLoc());
          LOG(WARNING) << "Ignoring [[xlscc::hls_unroll(no)]] (at "
                       << presumed_loc.getFilename() << ":"
                       << presumed_loc.getLine()
                       << "). Attribute is not needed and has no effect.";
          return true;
        } else if (absl::EqualsIgnoreCase(arg_name, "yes")) {
          // Ignore the argument; the default is unbounded.
          num_args = 0;
        } else {
          unsigned ID = S.getDiagnostics().getCustomDiagID(
              clang::DiagnosticsEngine::Error,
              "%0 attribute argument must be 'yes', 'no', or an integer.");
          S.Diag(Attr.getLoc(), ID) << Attr;
          return false;
        }
      }
    }
    Result = clang::AnnotateAttr::Create(S.Context, "hls_unroll", args,
                                         num_args, Attr.getRange());
    return true;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsUnrollAttrInfo> hls_unroll(
    "hls_unroll", "Marks a loop to be unrolled, and how.");

struct XlsChannelStrictnessAttrInfo : public clang::ParsedAttrInfo {
  XlsChannelStrictnessAttrInfo() {
    // GNU-style __attribute__(("hls_channel_strictness")) and C++/C23-style
    // [[hls_channel_strictness]] and [[xlscc::hls_channel_strictness]]
    // supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "hls_channel_strictness"},
        {clang::ParsedAttr::AS_C23, "hls_channel_strictness"},
        {clang::ParsedAttr::AS_CXX11, "hls_channel_strictness"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::hls_channel_strictness"}};
    Spellings = S;
    NumArgs = 1;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    // This attribute appertains to channel declarations only.
    if (!isa<clang::VarDecl>(D) && !isa<clang::FieldDecl>(D)) {
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "channel declarations";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    // Attach an annotate attribute to the Decl.
    if (Attr.getNumArgs() != 1) {
      S.Diag(Attr.getLoc(), clang::diag::err_attribute_wrong_number_arguments)
          << Attr << 1;
      return AttributeNotApplied;
    }

    clang::Expr* args[1];
    std::string strictness_str;
    clang::SourceLocation arg_loc;
    if (Attr.isArgIdent(0)) {
      strictness_str = Attr.getArgAsIdent(0)->Ident->getName();
      arg_loc = Attr.getArgAsIdent(0)->Loc;
    } else {
      args[0] = Attr.getArgAsExpr(0);
      arg_loc = args[0]->getExprLoc();
      if (clang::StringLiteral* literal =
              dyn_cast<clang::StringLiteral>(args[0]);
          literal != nullptr && !literal->isUnevaluated() &&
          !literal->isOrdinary()) {
        strictness_str = literal->getString();
      } else if (std::optional<std::string> maybe_strictness =
                     args[0]->tryEvaluateString(S.Context);
                 maybe_strictness.has_value()) {
        strictness_str = *maybe_strictness;
      } else if (clang::Expr::EvalResult result;
                 args[0]->EvaluateAsInt(result, S.Context) &&
                 result.Val.getInt().isRepresentableByInt64()) {
        // Accept values that evaluate to a constant integer, since they might
        // be xls::ChannelStrictness enum values - but translate them to a
        // string (we'll translate it back later) to give us a chance to report
        // that the value is invalid.
        int64_t strictness_val = result.Val.getInt().getExtValue();
        if (strictness_val >
            static_cast<int64_t>(
                std::numeric_limits<xls::ChannelStrictness>::max())) {
          strictness_str = "unknown";
        } else {
          strictness_str = xls::ChannelStrictnessToString(
              static_cast<xls::ChannelStrictness>(strictness_val));
        }
      } else {
        S.Diag(Attr.getLoc(), clang::diag::err_attribute_argument_type)
            << Attr << clang::AANT_ArgumentString;
        return AttributeNotApplied;
      }
    }

    absl::StatusOr<xls::ChannelStrictness> strictness =
        xls::ChannelStrictnessFromString(strictness_str);
    if (!strictness.ok()) {
      unsigned ID = S.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "%0 attribute argument not a recognized channel strictness: %1");
      S.Diag(Attr.getLoc(), ID) << Attr << strictness.status().message();
    }

    unsigned strictness_size = 8 * sizeof(xls::ChannelStrictness);
    clang::QualType literal_type =
        S.Context.getIntTypeForBitwidth(strictness_size,
                                        /*Signed=*/false);
    clang::Expr* strictness_expr = clang::IntegerLiteral::Create(
        S.Context,
        llvm::APInt(strictness_size, static_cast<uint64_t>(*strictness)),
        literal_type, arg_loc);
    D->addAttr(clang::AnnotateAttr::Create(S.Context, "hls_channel_strictness",
                                           &strictness_expr, /*ArgsSize=*/1,
                                           Attr.getRange()));
    return AttributeApplied;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsChannelStrictnessAttrInfo>
    hls_channel_strictness("hls_channel_strictness",
                           "Specifies the strictness of the defined channel.");

struct XlsAsapAttrInfo : public clang::ParsedAttrInfo {
  XlsAsapAttrInfo() {
    // GNU-style __attribute__(("xlscc_asap")) and C++/C23-style
    // [[xlscc_asap]] and [[xlscc::asap]] supported.
    static constexpr Spelling S[] = {
        {clang::ParsedAttr::AS_GNU, "xlscc_asap"},
        {clang::ParsedAttr::AS_C23, "xlscc_asap"},
        {clang::ParsedAttr::AS_CXX11, "xlscc_asap"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::asap"},
        {clang::ParsedAttr::AS_CXX11, "xlscc::xlscc_asap"}};
    Spellings = S;
  }

  bool diagAppertainsToDecl(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Decl* D) const override {
    if (isa<clang::LabelDecl>(D)) {
      // Allow application to labels; enables pass-through to the following
      // statement.
      return true;
    }
    S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
        << Attr << Attr.isRegularKeywordAttribute() << "loop statements";
    return false;
  }

  AttrHandling handleDeclAttribute(
      clang::Sema& S, clang::Decl* D,
      const clang::ParsedAttr& Attr) const override {
    if (!isa<clang::LabelDecl>(D)) {
      return NotHandled;
    }

    clang::Attr* Result;
    if (!CreateAttr(S, Attr, Result)) {
      return AttributeNotApplied;
    }
    D->addAttr(Result);
    return AttributeApplied;
  }

  bool diagAppertainsToStmt(clang::Sema& S, const clang::ParsedAttr& Attr,
                            const clang::Stmt* St) const override {
    if (!isa<clang::ForStmt>(St) && !isa<clang::WhileStmt>(St) &&
        !isa<clang::DoStmt>(St)) {
      // This attribute appertains to loop statements only.
      S.Diag(Attr.getLoc(), clang::diag::warn_attribute_wrong_decl_type_str)
          << Attr << Attr.isRegularKeywordAttribute() << "loop statements";
      return false;
    }
    return true;
  }

  AttrHandling handleStmtAttribute(clang::Sema& S, clang::Stmt* St,
                                   const clang::ParsedAttr& Attr,
                                   class clang::Attr*& Result) const override {
    return CreateAttr(S, Attr, Result) ? AttributeApplied : AttributeNotApplied;
  }

 private:
  bool CreateAttr(clang::Sema& S, const clang::ParsedAttr& Attr,
                  clang::Attr*& Result) const {
    Result = clang::AnnotateAttr::Create(S.Context, "xlscc_asap", nullptr,
                                         /*ArgsSize=*/0, Attr.getRange());
    return true;
  }
};
static clang::ParsedAttrInfoRegistry::Add<XlsAsapAttrInfo> xlscc_asap(
    "xlscc_asap",
    "Marks a loop to be scheduled ASAP; declares that there are no "
    "cross-iteration dependencies.");

class HlsArgsPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsArgsPragmaHandler(const std::string& pragma_name,
                                int64_t num_args)
      : clang::PragmaHandler(pragma_name), num_args_(num_args) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) final {
    std::optional<std::vector<clang::Token>> lexed_args_opt = LexPragmaArgs(
        PP, firstToken, num_args_, clang::PragmaHandler::getName());

    if (!lexed_args_opt.has_value()) {
      return;
    }

    HandlePragma(PP, Introducer, firstToken, lexed_args_opt.value());
  }

  virtual void HandlePragma(clang::Preprocessor& PP,
                            clang::PragmaIntroducer Introducer,
                            clang::Token& firstToken,
                            const std::vector<clang::Token>& toks) = 0;

 private:
  int64_t num_args_ = 0;
};

class HlsNoTuplePragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsNoTuplePragmaHandler()
      : HlsArgsPragmaHandler("hls_no_tuple", /*num_args=*/0) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    const clang::PresumedLoc presumed_loc =
        PP.getSourceManager().getPresumedLoc(Introducer.Loc);

    PP.Diag(Introducer.Loc, clang::diag::err_pragma_message) << absl::StrFormat(
        "%s cannot be used as a pragma, use an attribute instead. At "
        "%s:%i",
        clang::PragmaHandler::getName(), presumed_loc.getFilename(),
        presumed_loc.getLine());
  }
};

static clang::PragmaHandlerRegistry::Add<HlsNoTuplePragmaHandler>
    hls_no_tuple_pragma("hls_no_tuple",
                        "Pragma to mark a class or struct as represented in "
                        "the IR by its first, and only, field.");

class HlsSyntheticIntPragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsSyntheticIntPragmaHandler()
      : HlsArgsPragmaHandler("hls_synthetic_int", /*num_args=*/0) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    const clang::PresumedLoc presumed_loc =
        PP.getSourceManager().getPresumedLoc(Introducer.Loc);

    PP.Diag(Introducer.Loc, clang::diag::err_pragma_message) << absl::StrFormat(
        "%s cannot be used as a pragma, use an attribute instead. At "
        "%s:%i",
        clang::PragmaHandler::getName(), presumed_loc.getFilename(),
        presumed_loc.getLine());
  }
};

static clang::PragmaHandlerRegistry::Add<HlsSyntheticIntPragmaHandler>
    hls_synthetic_int_pragma(
        "hls_synthetic_int",
        "Pragma to mark a class or struct as representing a custom integer "
        "implementation in the generated metadata.");

class HlsTopPragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsTopPragmaHandler()
      : HlsArgsPragmaHandler("hls_top", /*num_args=*/0) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    GenerateAnnotation(PP, clang::PragmaHandler::getName(), firstToken,
                       /*arguments=*/{});
  }
};

static clang::PragmaHandlerRegistry::Add<HlsTopPragmaHandler> hls_top_pragma(
    "hls_top",
    "Pragma to mark a function or method as the entrypoint for translation.");

class HlsBlockPragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsBlockPragmaHandler()
      : HlsArgsPragmaHandler("hls_block", /*num_args=*/0) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    GenerateAnnotation(PP, clang::PragmaHandler::getName(), firstToken,
                       /*arguments=*/{});
  }
};

static clang::PragmaHandlerRegistry::Add<HlsBlockPragmaHandler>
    hls_block_pragma("hls_block",
                     "Pragma to mark a function or method as the entrypoint "
                     "for translation.");

class HlsAllowDefaultPadPragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsAllowDefaultPadPragmaHandler()
      : HlsArgsPragmaHandler("hls_array_allow_default_pad", /*num_args=*/0) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    GenerateAnnotation(PP, "hls_array_allow_default_pad", firstToken,
                       /*arguments=*/{});
  }
};

static clang::PragmaHandlerRegistry::Add<HlsAllowDefaultPadPragmaHandler>
    hls_array_allow_default_pad_pragma(
        "hls_array_allow_default_pad",
        "Pragma to mark a declaration as allowing default initialization to "
        "zero for missing elements in the initializer list.");

class HlsDesignPragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsDesignPragmaHandler()
      : HlsArgsPragmaHandler("hls_design", /*num_args=*/1) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    const clang::Token& tok = toks.at(0);

    if (tok.is(clang::tok::identifier)) {
      const clang::IdentifierInfo* identifier = tok.getIdentifierInfo();
      if (identifier->isStr("top")) {
        GenerateAnnotation(PP, "hls_top", tok,
                           /*arguments=*/{});
      } else {
        LOG(WARNING) << "Ignoring unknown #pragma hls_design: "
                     << identifier->getName().str();
      }
    }
  }
};

static clang::PragmaHandlerRegistry::Add<HlsDesignPragmaHandler>
    hls_design_pragma("hls_design",
                      "Pragma of the form hls_design [top], equivalent to "
                      "other pragmas such as hls_top");

class HlsPipelineInitIntervalPragmaHandler : public clang::PragmaHandler {
 public:
  HlsPipelineInitIntervalPragmaHandler()
      : clang::PragmaHandler("hls_pipeline_init_interval") {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    std::vector<clang::Token> toks;
    PP.LexTokensUntilEOF(&toks);
    GenerateAnnotation(PP, "hls_pipeline_init_interval", firstToken, toks);
  }
};
static clang::PragmaHandlerRegistry::Add<HlsPipelineInitIntervalPragmaHandler>
    hls_pipeline_init_interval_pragma(
        "hls_pipeline_init_interval",
        "Pragma specifying that a loop should be pipelined, and how.");

class HlsUnrollPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsUnrollPragmaHandler() : clang::PragmaHandler("hls_unroll") {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    std::vector<clang::Token> toks;
    PP.LexTokensUntilEOF(&toks);
    GenerateAnnotation(PP, "hls_unroll", firstToken, toks);
  }
};
static clang::PragmaHandlerRegistry::Add<HlsUnrollPragmaHandler>
    hls_unroll_pragma(
        "hls_unroll",
        "Pragma specifying that a loop should be unrolled, and how.");

class UnknownPragmaHandler : public clang::PragmaHandler {
 public:
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const std::string& name = firstToken.getIdentifierInfo()->getName().str();
    static const auto* non_hls_names = new absl::flat_hash_set<std::string>(
        {"top", "design", "pipeline_init_interval", "array_allow_default_pad",
         "no_tuple", "synthetic_int", "unroll"});
    if (non_hls_names->contains(name)) {
      LOG(WARNING) << "WARNING: #pragma '" << name
                   << "' requires 'hls_' prefix";
      return;
    }
    static const auto* names_upper = new absl::flat_hash_set<std::string>(
        {"TOP", "HLS_TOP", "DESIGN", "HLS_DESIGN", "PIPELINE_INIT_INTERVAL",
         "HLS_PIPELINE_INIT_INTERVAL", "ARRAY_ALLOW_DEFAULT_PAD",
         "HLS_ARRAY_ALLOW_DEFAULT_PAD", "NO_TUPLE", "HLS_NO_TUPLE",
         "SYNTHETIC_INT", "HLS_SYNTHETIC_INT", "UNROLL", "HLS_UNROLL"});
    if (names_upper->contains(name)) {
      LOG(WARNING) << "#pragma must be lowercase: " << name;
      return;
    }
  };
};

class LibToolASTConsumer : public clang::ASTConsumer {
 public:
  explicit LibToolASTConsumer(clang::CompilerInstance& CI, CCParser& parser)
      : visitor_(new LibToolVisitor(CI, parser)) {}

  void HandleTranslationUnit(clang::ASTContext& Context) override {
    visitor_->TraverseDecl(Context.getTranslationUnitDecl());
  }

 private:
  std::unique_ptr<LibToolVisitor> visitor_;
};
class LibToolFrontendAction : public clang::ASTFrontendAction {
 public:
  explicit LibToolFrontendAction(CCParser& parser) : parser_(parser) {}
  void EndSourceFileAction() override;
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& CI, clang::StringRef /*file*/) override {
    compiler_instance_ = &CI;
    return std::unique_ptr<clang::ASTConsumer>(
        new LibToolASTConsumer(CI, parser_));
  }
  void ExecuteAction() override { clang::ASTFrontendAction::ExecuteAction(); }

 private:
  CCParser& parser_;
  clang::CompilerInstance* compiler_instance_;
};
class DiagnosticInterceptor : public clang::TextDiagnosticPrinter {
 public:
  DiagnosticInterceptor(CCParser& translator, llvm::raw_ostream& os,
                        clang::DiagnosticOptions* diags,
                        bool OwnsOutputStream = false)
      : clang::TextDiagnosticPrinter(os, diags, OwnsOutputStream),
        parser_(translator) {}
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic& info) override {
    // Print the message
    clang::TextDiagnosticPrinter::HandleDiagnostic(level, info);

    if (level >= clang::DiagnosticsEngine::Level::Error) {
      llvm::SmallString<1024> str;
      info.FormatDiagnostic(str);
      parser_.libtool_visit_status_ = absl::FailedPreconditionError(str.str());
      return;
    }
  }

 private:
  CCParser& parser_;
};

CCParser::~CCParser() {
  // Allow parser and its thread to be destroyed
  if (libtool_wait_for_destruct_ != nullptr) {
    libtool_wait_for_destruct_->DecrementCount();
  }
  // Wait for it to be gone
  if (libtool_thread_ != nullptr) {
    libtool_thread_->Join();
  }
}

absl::Status CCParser::ScanFile(
    std::string_view source_filename,
    absl::Span<std::string_view> command_line_args) {
  // This function may only be called once in the lifetime of a CCParser.
  CHECK_EQ(libtool_thread_.get(), nullptr);
  CHECK_EQ(libtool_wait_for_destruct_.get(), nullptr);

  // The AST is destroyed after ToolInvocation::run() returns
  //
  // However, we want to preserve it to access it across multiple passes and
  //  various subsequent calls, such as GenerateIR().
  //
  // Therefore, ToolInvocation::Run() is executed on another thread,
  //  and the ASTFrontendAction::EndSourceFileAction() blocks it
  //  until ~CCParser(), preserving the AST.
  libtool_thread_ = std::make_unique<LibToolThread>(
      source_filename, top_class_name_, command_line_args, *this);
  libtool_wait_for_parse_ = std::make_unique<absl::BlockingCounter>(1);
  libtool_wait_for_destruct_ = std::make_unique<absl::BlockingCounter>(1);
  libtool_visit_status_ = absl::OkStatus();
  libtool_thread_->Start();
  libtool_wait_for_parse_->Wait();
  return libtool_visit_status_;
}

void CCParser::AddSourceInfoToMetadata(xlscc_metadata::MetadataOutput& output) {
  for (const auto& [path, number] : file_numbers_) {
    xlscc_metadata::SourceName* source = output.add_sources();
    source->set_path(path);
    source->set_number(number);
  }
}

void CCParser::AddSourceInfoToPackage(xls::Package& package) {
  for (const auto& [path, number] : file_numbers_) {
    package.SetFileno(xls::Fileno(number), path);
  }
}

clang::PresumedLoc CCParser::GetPresumedLoc(const clang::Stmt& stmt) {
  return sm_->getPresumedLoc(stmt.getSourceRange().getBegin());
}

xls::SourceInfo CCParser::GetLoc(const clang::Stmt& stmt) {
  return GetLoc(GetPresumedLoc(stmt));
}

clang::PresumedLoc CCParser::GetPresumedLoc(const clang::Decl& decl) {
  return sm_->getPresumedLoc(decl.getSourceRange().getBegin());
}

xls::SourceInfo CCParser::GetLoc(const clang::Decl& decl) {
  return GetLoc(GetPresumedLoc(decl));
}

xls::SourceInfo CCParser::GetLoc(const clang::Expr& expr) {
  return GetLoc(sm_->getPresumedLoc(expr.getExprLoc()));
}

xls::SourceInfo CCParser::GetLoc(const clang::PresumedLoc& loc) {
  if (!loc.isValid()) {
    return xls::SourceInfo();
  }

  std::filesystem::path path = loc.getFilename();
  absl::StatusOr<std::filesystem::path> current_path =
      xls::GetCurrentDirectory();
  if (current_path.ok()) {
    path = path.lexically_proximate(*current_path);
  }
  auto found = file_numbers_.find(path.string());

  int id = 0;

  if (found == file_numbers_.end()) {
    id = next_file_number_++;
    file_numbers_[path.string()] = id;
  } else {
    id = found->second;
  }

  return xls::SourceInfo(xls::SourceLocation(xls::Fileno(id),
                                             xls::Lineno(loc.getLine()),
                                             xls::Colno(loc.getColumn())));
}

std::string CCParser::LocString(const xls::SourceInfo& loc) {
  std::vector<std::string> strings;
  for (const xls::SourceLocation& location : loc.locations) {
    std::string found_str = "Unknown";
    for (const auto& it : file_numbers_) {
      if (it.second == static_cast<int>(location.fileno())) {
        found_str = it.first;
        break;
      }
    }
    strings.push_back(absl::StrFormat("%s:%i:%i", found_str,
                                      static_cast<int>(location.lineno()),
                                      static_cast<int>(location.fileno())));
  }
  return absl::StrFormat("[%s]", absl::StrJoin(strings, ", "));
}

bool CCParser::LibToolVisitFunction(clang::FunctionDecl* func) {
  if (libtool_visit_status_.ok()) {
    libtool_visit_status_ = VisitFunction(func);
  }
  return libtool_visit_status_.ok();
}

bool CCParser::LibToolVisitVarDecl(clang::VarDecl* func) {
  if (libtool_visit_status_.ok()) {
    libtool_visit_status_ = VisitVarDecl(func);
  }
  return libtool_visit_status_.ok();
}

// Scans for top-level function candidates
absl::Status CCParser::VisitFunction(const clang::FunctionDecl* funcdecl) {
  if (sm_ == nullptr) {
    sm_ = &funcdecl->getASTContext().getSourceManager();
  }

  const std::string fname = funcdecl->getNameAsString();

  // Top can't be a template function
  if (funcdecl->getTemplatedKind() !=
      clang::FunctionDecl::TemplatedKind::TK_NonTemplate) {
    return absl::OkStatus();
  }

  // Top can't be a forward declarations
  if (!funcdecl->doesThisDeclarationHaveABody()) {
    return absl::OkStatus();
  }

  const clang::AnnotateAttr* attr = funcdecl->getAttr<clang::AnnotateAttr>();

  std::string annotation;

  if (attr != nullptr) {
    annotation = attr->getAnnotation().str();
  }

  if ((annotation == "hls_top") || fname == top_function_name_) {
    if (top_function_ == nullptr) {
      top_function_ = funcdecl;
    } else if (fname == top_function_name_) {
      // Already found a function with the top name
      if (top_function_->getNameAsString() == top_function_name_) {
        return absl::AlreadyExistsError(absl::StrFormat(
            "Two top functions defined by name, at %s, previously at %s",
            LocString(GetLoc(*top_function_)), LocString(GetLoc(*funcdecl))));
      }
      // Name takes precedence over annotation
      top_function_ = funcdecl;
    } else if (annotation == "hls_top") {
      // If the name doesn't match the top, then it was annotation specified
      if (top_function_->getNameAsString() != top_function_name_) {
        return absl::AlreadyExistsError(absl::StrFormat(
            "Two top functions defined by annotation, at %s, previously at %s",
            LocString(GetLoc(*top_function_)), LocString(GetLoc(*funcdecl))));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status CCParser::VisitVarDecl(const clang::VarDecl* decl) {
  const std::string name = decl->getNameAsString();

  if (name == "__xlscc_on_reset") {
    CHECK(xlscc_on_reset_ == nullptr || xlscc_on_reset_ == decl);
    xlscc_on_reset_ = decl;
  }

  return absl::OkStatus();
}

absl::StatusOr<const clang::VarDecl*> CCParser::GetXlsccOnReset() const {
  if (xlscc_on_reset_ == nullptr) {
    return absl::NotFoundError(
        "__xlscc_on_reset not found. Missing /xls_builtin.h?");
  }
  return xlscc_on_reset_;
}

LibToolThread::LibToolThread(std::string_view source_filename,
                             std::string_view top_class_name,
                             absl::Span<std::string_view> command_line_args,
                             CCParser& parser)
    : source_filename_(source_filename),
      top_class_name_(top_class_name),
      command_line_args_(command_line_args),
      parser_(parser) {}

void LibToolThread::Start() {
  thread_.emplace([this] { Run(); });
}

void LibToolThread::Join() { thread_->Join(); }

void LibToolThread::Run() {
  std::vector<std::string> argv;
  argv.emplace_back("binary");
  argv.emplace_back("/xls_top.cc");
  for (const auto& view : command_line_args_) {
    argv.emplace_back(view);
  }
  // For xls_top.cc to include the source file
  argv.emplace_back("-I.");
  argv.emplace_back("-fsyntax-only");
  argv.emplace_back("-std=c++17");
  argv.emplace_back("-nostdinc");
  argv.emplace_back("-Wno-unused-label");
  argv.emplace_back("-Wno-constant-logical-operand");
  argv.emplace_back("-Wno-unused-but-set-variable");
  argv.emplace_back("-Wno-c++11-narrowing");
  argv.emplace_back("-Wno-conversion");
  argv.emplace_back("-Wno-missing-template-arg-list-after-template-kw");

  llvm::IntrusiveRefCntPtr<clang::FileManager> libtool_files;

  std::unique_ptr<LibToolFrontendAction> libtool_action(
      new LibToolFrontendAction(parser_));

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> mem_fs(
      new llvm::vfs::InMemoryFileSystem);
  mem_fs->addFile("/xls_builtin.h", 0,
                  llvm::MemoryBuffer::getMemBuffer(
                      R"(
#ifndef __XLS_BUILTIN_H
#define __XLS_BUILTIN_H
template<int N>
struct __xls_bits { };

// Should match OpType
enum __xls_channel_dir {
  __xls_channel_dir_Unknown=0,    // OpType::kNull
  __xls_channel_dir_Out=1,        // OpType::kSend
  __xls_channel_dir_In=2,         // OpType::kRecv
  __xls_channel_dir_InOut=3       // OpType::kSendRecv
};

template<typename T, __xls_channel_dir Dir=__xls_channel_dir_Unknown>
class __xls_channel {
 public:
  T read()const {
    return T();
  }
  T write(T val)const {
    return val;
  }
  void read(T& out)const {
    (void)out;
  }
  bool nb_read(T& out)const {
    (void)out;
    return true;
  }
};

template<typename T, unsigned long long Size>
class __xls_memory {
 public:
  using value_type = T;

  unsigned long long size()const {
    return Size;
  };

  T& operator[](long long int addr)const {
    static T ret;
    return ret;
  }
  void write(long long int addr, const T& value) const {
    return;
  }
  T read(long long int addr) const {
    return T();
  }
};


// Bypass no outputs error
int __xlscc_unimplemented() { return 0; }

void __xlscc_assert(const char*message, bool condition, const char*label=nullptr) { }

// See XLS IR trace op format
void __xlscc_trace(const char*fmt, ...) { }

bool __xlscc_on_reset = false;

// Returns bits for 32.32 fixed point representation
__xls_bits<64> __xlscc_fixed_32_32_bits_for_double(double input);
__xls_bits<64> __xlscc_fixed_32_32_bits_for_float(float input);

#endif//__XLS_BUILTIN_H
          )"));

  // Inject an instantiation to make Clang parse the constructor bodies
  std::string top_class_inst_injection = top_class_name_.empty()
                                             ? ""
                                             : absl::StrFormat(R"(
namespace {
// Avoid unused variable warnings
void __xlscc_top_class_instance_ref2();
void __xlscc_top_class_instance_ref() {
  %s inst;
  (void)inst;
  __xlscc_top_class_instance_ref2();
}
void __xlscc_top_class_instance_ref2() {
  __xlscc_top_class_instance_ref();
}
}  // namespace
)",
                                                               top_class_name_);

  const std::string top_src =
      absl::StrFormat(R"(
#include "/xls_builtin.h"
#include "%s"
%s
          )",
                      source_filename_, top_class_inst_injection);

  mem_fs->addFile("/xls_top.cc", 0, llvm::MemoryBuffer::getMemBuffer(top_src));

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));

  overlay_fs->pushOverlay(mem_fs);

  libtool_files =
      new clang::FileManager(clang::FileSystemOptions(), overlay_fs);

  std::unique_ptr<clang::tooling::ToolInvocation> libtool_inv(
      new clang::tooling::ToolInvocation(argv, std::move(libtool_action),
                                         libtool_files.get()));

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_opts =
      new clang::DiagnosticOptions();
  DiagnosticInterceptor diag_print(parser_, llvm::errs(), &*diag_opts);
  libtool_inv->setDiagnosticConsumer(&diag_print);

  // Errors are extracted via DiagnosticInterceptor,
  //  since we block in run() until ~CCParser()
  (void)libtool_inv->run();
}

void LibToolFrontendAction::EndSourceFileAction() {
  // ToolInvocation::run() from returning until ~CCParser()
  parser_.libtool_wait_for_parse_->DecrementCount();
  parser_.libtool_wait_for_destruct_->Wait();
}

absl::StatusOr<std::string> CCParser::GetEntryFunctionName() const {
  if (top_function_ == nullptr) {
    return absl::NotFoundError("No top function found");
  }
  return top_function_->getNameAsString();
}

absl::Status CCParser::SelectTop(std::string_view top_function_name,
                                 std::string_view top_class_name) {
  top_function_name_ = top_function_name;
  top_class_name_ = top_class_name;
  return absl::OkStatus();
}

absl::StatusOr<const clang::FunctionDecl*> CCParser::GetTopFunction() const {
  if (top_function_ == nullptr) {
    return absl::NotFoundError("No top function found");
  }
  return top_function_;
}

}  // namespace xlscc
