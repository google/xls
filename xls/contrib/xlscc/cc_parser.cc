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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <list>
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
#include "absl/strings/numbers.h"
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
#include "llvm/include/llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/VirtualFileSystem.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/ir/fileno.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "re2/re2.h"

namespace xlscc {
namespace {

void GenerateIntrinsicCall(clang::Preprocessor& PP, std::string_view name,
                           const clang::Token& after,
                           const absl::Span<const clang::Token>& arguments) {
  const int64_t sNTokens = 4 + arguments.size() * 2 - 1;
  std::unique_ptr<clang::Token[]> tokens =
      std::make_unique<clang::Token[]>(sNTokens);

  int64_t current_token = 0;

  tokens[current_token].setKind(clang::tok::identifier);
  tokens[current_token].setLength(name.size());
  tokens[current_token].setLocation(after.getLocation());
  tokens[current_token].setIdentifierInfo(PP.getIdentifierInfo(name));
  ++current_token;

  tokens[current_token].setKind(clang::tok::l_paren);
  tokens[current_token].setLength(0);
  tokens[current_token].setLocation(after.getLocation());
  ++current_token;

  for (int64_t i = 0; i < arguments.size(); ++i) {
    if (i > 0) {
      tokens[current_token].setKind(clang::tok::comma);
      tokens[current_token].setLength(0);
      tokens[current_token].setLocation(after.getLocation());
      ++current_token;
    }

    const clang::Token& argument = arguments.at(i);
    tokens[current_token] = argument;
    tokens[current_token].setLocation(after.getLocation());
    ++current_token;
  }

  tokens[current_token].setKind(clang::tok::r_paren);
  tokens[current_token].setLength(0);
  tokens[current_token].setLocation(after.getLocation());
  ++current_token;

  tokens[current_token].setKind(clang::tok::semi);
  tokens[current_token].setLength(0);
  tokens[current_token].setLocation(after.getEndLoc());
  ++current_token;

  PP.EnterTokenStream(std::move(tokens), sNTokens,
                      /*DisableMacroExpansion=*/false,
                      /*IsReinject=*/false);
}

void GenerateIntrinsicCall(clang::Preprocessor& PP, std::string_view name,
                           const clang::Token& after, int64_t argument) {
  clang::Token arg = after;

  thread_local std::list<std::string> literals;
  literals.push_back(absl::StrFormat("%i", argument));

  arg.setKind(clang::tok::numeric_constant);
  arg.setLiteralData(literals.back().c_str());
  arg.setLength(literals.back().size());

  GenerateIntrinsicCall(PP, name, after,
                        /*arguments=*/{arg});
}

void GenerateAnnotation(clang::Preprocessor& PP, std::string_view name,
                        const clang::Token& after,
                        const absl::Span<const clang::Token>& arguments) {
  // TODO(seanhaskell): Attributes with arguments are not yet correctly
  // supported b/371085056
  CHECK(arguments.empty());

  std::unique_ptr<clang::Token[]> tokens = std::make_unique<clang::Token[]>(5);
  tokens[0].setKind(clang::tok::l_square);
  tokens[0].setLength(0);
  tokens[0].setLocation(after.getLocation());
  tokens[1].setKind(clang::tok::l_square);
  tokens[1].setLength(0);
  tokens[1].setLocation(after.getLocation());
  tokens[2].setKind(clang::tok::identifier);
  tokens[2].setLength(after.getLength());
  tokens[2].setLocation(after.getLocation());
  tokens[2].setIdentifierInfo(PP.getIdentifierInfo(name));
  tokens[3].setKind(clang::tok::r_square);
  tokens[3].setLength(0);
  tokens[3].setLocation(after.getEndLoc());
  tokens[4].setKind(clang::tok::r_square);
  tokens[4].setLength(0);
  tokens[4].setLocation(after.getEndLoc());
  PP.EnterTokenStream(std::move(tokens), 5,
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
        "hls_array_allow_default_pad",
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

class HlsArgsPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsArgsPragmaHandler(const std::string& pragma_name,
                                int64_t num_args)
      : clang::PragmaHandler(pragma_name), num_args_(num_args) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override final {
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

class HlsPipelineInitIntervalPragmaHandler : public HlsArgsPragmaHandler {
 public:
  HlsPipelineInitIntervalPragmaHandler()
      : HlsArgsPragmaHandler("hls_pipeline_init_interval", /*num_args=*/1) {}

  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& toks) override {
    const clang::Token& tok = toks.at(0);

    const clang::PresumedLoc presumed_loc =
        PP.getSourceManager().getPresumedLoc(Introducer.Loc);

    if (tok.getKind() != clang::tok::numeric_constant) {
      PP.Diag(tok.getLocation(), clang::diag::err_pragma_message)
          << absl::StrFormat(
                 "Argument to pragma 'hls_pipeline_init_interval' is not valid."
                 "Must be an integer "
                 ">= 1. At %s:%i",
                 presumed_loc.getFilename(), presumed_loc.getLine());
      return;
    }
    const char* literal_data = tok.getLiteralData();
    std::string_view str_identifier(literal_data, tok.getLength());
    int64_t arg = -1;
    if (!absl::SimpleAtoi(str_identifier, &arg) || (arg <= 0)) {
      PP.Diag(tok.getLocation(), clang::diag::err_pragma_message)
          << absl::StrFormat(
                 "Argument '%s' to pragma 'hls_pipeline_init_interval' is not "
                 "valid."
                 "Must be an integer "
                 ">= 1. At %s:%i",
                 str_identifier, presumed_loc.getFilename(),
                 presumed_loc.getLine());
      return;
    }

    GenerateIntrinsicCall(PP, "__xlscc_pipeline", tok,
                          /*arguments=*/{tok});
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
    const clang::PresumedLoc presumed_loc =
        PP.getSourceManager().getPresumedLoc(Introducer.Loc);

    std::vector<clang::Token> toks;
    PP.LexTokensUntilEOF(&toks);

    if (toks.empty()) {  // no arguments
      GenerateIntrinsicCall(PP, "__xlscc_unroll", firstToken, 1);
      return;
    }

    if (toks.size() != 1) {
      PP.Diag(firstToken.getLocation(), clang::diag::err_pragma_message)
          << absl::StrFormat(
                 "#pragma %s must have 1 argument (negative numbers are "
                 "invalid). At %s:%i",
                 clang::PragmaHandler::getName(), presumed_loc.getFilename(),
                 presumed_loc.getLine());
      return;
    }

    const clang::Token& tok = toks.at(0);

    if (tok.is(clang::tok::identifier)) {
      const clang::IdentifierInfo* identifier = tok.getIdentifierInfo();
      if (identifier->isStr("yes")) {
        GenerateIntrinsicCall(PP, "__xlscc_unroll", firstToken, 1);
        return;
      }
      if (identifier->isStr("no")) {
        LOG(WARNING) << "Ignoring #pragma hls_unroll no (at "
                     << presumed_loc.getFilename() << ":"
                     << presumed_loc.getLine()
                     << "). Pragma is not needed and has no effect.";
        return;
      }
    }
    if (tok.getKind() != clang::tok::numeric_constant) {
      PP.Diag(tok.getLocation(), clang::diag::err_pragma_message)
          << absl::StrFormat(
                 "Argument to pragma 'hls_unroll' is not valid. "
                 "Must be 'yes', 'no', or an integer."
                 " At %s:%i",
                 presumed_loc.getFilename(), presumed_loc.getLine());
      return;
    }
    const char* literal_data = tok.getLiteralData();
    std::string_view str_identifier(literal_data, tok.getLength());
    int64_t arg = -1;
    if (!absl::SimpleAtoi(str_identifier, &arg) || (arg <= 0)) {
      if (arg == 0) {
        LOG(WARNING) << "Ignoring #pragma hls_unroll 0 (at "
                     << presumed_loc.getFilename() << ":"
                     << presumed_loc.getLine()
                     << "). Pragma is "
                        "not needed and has no effect.";
        return;
      }
      PP.Diag(tok.getLocation(), clang::diag::err_pragma_message)
          << absl::StrFormat(
                 "Argument '%s' to pragma 'hls_unroll' is not valid. "
                 "Must be 'yes', 'no', or an integer."
                 " At %s:%i",
                 str_identifier, presumed_loc.getFilename(),
                 presumed_loc.getLine());
      return;
    }
    LOG(WARNING) << "Partial unroll not yet supported: fully unrolling";
    GenerateIntrinsicCall(PP, "__xlscc_unroll", tok,
                          /*arguments=*/{tok});
  }
  clang::CompilerInstance* compiler_instance_;
};

static clang::PragmaHandlerRegistry::Add<HlsUnrollPragmaHandler>
    hls_unroll_pragma(
        "hls_unroll",
        "Pragma specifying that a loop should be unrolled, and how.");

class HlsChannelStrictnessPragmaHandler : public HlsArgsPragmaHandler {
 public:
  explicit HlsChannelStrictnessPragmaHandler()
      : HlsArgsPragmaHandler("hls_channel_strictness", /*num_args=*/0) {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken,
                    const std::vector<clang::Token>& tok) override {
    const clang::PresumedLoc presumed_loc =
        PP.getSourceManager().getPresumedLoc(Introducer.Loc);

    // TODO(seanhaskell): Implement with annotations b/371085056
    PP.Diag(firstToken.getLocation(), clang::diag::err_pragma_message)
        << absl::StrFormat(
               "#pragma hls_channel_strictness is currently unimplemented. At "
               "%s:%i",
               presumed_loc.getFilename(), presumed_loc.getLine());
  }
};

static clang::PragmaHandlerRegistry::Add<HlsChannelStrictnessPragmaHandler>
    hls_channel_strictness_pragma(
        "hls_channel_strictness",
        "Pragma specifying that a loop should be unrolled, and how.");

class UnknownPragmaHandler : public clang::PragmaHandler {
 public:
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const std::string& name = firstToken.getIdentifierInfo()->getName().str();
    static const auto* non_hls_names = new absl::flat_hash_set<std::string>(
        {"top", "design", "pipeline_init_interval", "array_allow_default_pad",
         "no_tuple", "synthetic_int", "unroll", "channel_strictness"});
    if (non_hls_names->contains(name)) {
      LOG(WARNING) << "WARNING: #pragma '" << name
                   << "' requires 'hls_' prefix";
      return;
    }
    static const auto* names_upper = new absl::flat_hash_set<std::string>(
        {"TOP", "HLS_TOP", "DESIGN", "HLS_DESIGN", "PIPELINE_INIT_INTERVAL",
         "HLS_PIPELINE_INIT_INTERVAL", "ARRAY_ALLOW_DEFAULT_PAD",
         "HLS_ARRAY_ALLOW_DEFAULT_PAD", "NO_TUPLE", "HLS_NO_TUPLE",
         "SYNTHETIC_INT", "HLS_SYNTHETIC_INT", "UNROLL", "HLS_UNROLL",
         "CHANNEL_STRICTNESS", "HLS_CHANNEL_STRICTNESS"});
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
    if (level >= clang::DiagnosticsEngine::Level::Error) {
      llvm::SmallString<1024> str;
      info.FormatDiagnostic(str);
      parser_.libtool_visit_status_ = absl::FailedPreconditionError(str.str());
      return;
    }
    // Print the message
    clang::TextDiagnosticPrinter::HandleDiagnostic(level, info);
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

// For use with loops
void __xlscc_pipeline(long long factor) { }
void __xlscc_unroll(long long factor) { }

// Place at the beginning of the token graph, connected to the end, in parallel
// to anything else, rather than serializing as by default
void __xlscc_asap() { }

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
