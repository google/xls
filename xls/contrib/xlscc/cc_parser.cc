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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/ASTConsumer.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclBase.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/Basic/Diagnostic.h"
#include "clang/include/clang/Basic/FileSystemOptions.h"
#include "clang/include/clang/Basic/IdentifierTable.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Basic/TokenKinds.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendAction.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Lex/PPCallbacks.h"
#include "clang/include/clang/Lex/Pragma.h"
#include "clang/include/clang/Lex/Token.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "llvm/include/llvm/ADT/IntrusiveRefCntPtr.h"
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

class HlsNoParamPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsNoParamPragmaHandler(clang::CompilerInstance* compiler_instance,
                                   CCParser& parser,
                                   std::string_view pragma_name,
                                   PragmaType pragma_type)
      : clang::PragmaHandler(pragma_name),
        compiler_instance_(compiler_instance),
        parser_(parser),
        pragma_type_(pragma_type) {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const clang::PresumedLoc presumed_loc =
        compiler_instance_->getSourceManager().getPresumedLoc(Introducer.Loc);
    // TODO(b/349335947) - Ensure pragmas are not unintentionally clobbered.
    parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                             presumed_loc.getLine())] =
        Pragma(pragma_type_);
  }
  clang::CompilerInstance* compiler_instance_;
  CCParser& parser_;
  PragmaType pragma_type_;
};

class HlsDesignPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsDesignPragmaHandler(clang::CompilerInstance* compiler_instance,
                                  CCParser& parser)
      : clang::PragmaHandler("hls_design"),
        compiler_instance_(compiler_instance),
        parser_(parser) {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const clang::PresumedLoc presumed_loc =
        compiler_instance_->getSourceManager().getPresumedLoc(Introducer.Loc);
    clang::Token tok;
    PP.Lex(tok);
    if (tok.is(clang::tok::identifier)) {
      const clang::IdentifierInfo* identifier = tok.getIdentifierInfo();
      if (identifier->isStr("block")) {
        parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                                 presumed_loc.getLine())] =
            Pragma(Pragma_Block);
      } else if (identifier->isStr("top")) {
        parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                                 presumed_loc.getLine())] =
            Pragma(Pragma_Top);
      } else {
        LOG(WARNING) << "Ignoring unknown #pragma hls_design: "
                     << identifier->getName().str();
      }
    }
  }
  clang::CompilerInstance* compiler_instance_;
  CCParser& parser_;
};

class HlsPipelineInitIntervalPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsPipelineInitIntervalPragmaHandler(
      clang::CompilerInstance* compiler_instance, CCParser& parser)
      : clang::PragmaHandler("hls_pipeline_init_interval"),
        compiler_instance_(compiler_instance),
        parser_(parser) {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const clang::PresumedLoc presumed_loc =
        compiler_instance_->getSourceManager().getPresumedLoc(Introducer.Loc);
    clang::Token tok;
    PP.Lex(tok);
    if (tok.getKind() != clang::tok::numeric_constant) {
      parser_.libtool_visit_status_ =
          absl::InvalidArgumentError(absl::StrFormat(
              "Argument to pragma 'hls_pipeline_init_interval' is not valid."
              "Must be an integer "
              ">= 1. At %s:%i",
              presumed_loc.getFilename(), presumed_loc.getLine()));
      return;
    }
    const char* literal_data = tok.getLiteralData();
    std::string_view str_identifier(literal_data, tok.getLength());
    int64_t arg = -1;
    if (!absl::SimpleAtoi(str_identifier, &arg) || (arg <= 0)) {
      parser_
          .libtool_visit_status_ = absl::InvalidArgumentError(absl::StrFormat(
          "Argument '%s' to pragma 'hls_pipeline_init_interval' is not valid."
          "Must be an integer "
          ">= 1. At %s:%i",
          str_identifier, presumed_loc.getFilename(), presumed_loc.getLine()));
      return;
    }

    parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                             presumed_loc.getLine())] =
        Pragma(Pragma_InitInterval, arg);
  }
  clang::CompilerInstance* compiler_instance_;
  CCParser& parser_;
};

class HlsUnrollPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsUnrollPragmaHandler(clang::CompilerInstance* compiler_instance,
                                  CCParser& parser)
      : clang::PragmaHandler("hls_unroll"),
        compiler_instance_(compiler_instance),
        parser_(parser) {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const clang::PresumedLoc presumed_loc =
        compiler_instance_->getSourceManager().getPresumedLoc(Introducer.Loc);
    clang::Token tok;
    PP.Lex(tok);
    if (tok.is(clang::tok::eod)) {  // no arguments
      parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                               presumed_loc.getLine())] =
          Pragma(Pragma_Unroll);
      return;
    }
    if (tok.is(clang::tok::identifier)) {
      const clang::IdentifierInfo* identifier = tok.getIdentifierInfo();
      if (identifier->isStr("yes")) {
        parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                                 presumed_loc.getLine())] =
            Pragma(Pragma_Unroll);
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
      parser_.libtool_visit_status_ = absl::InvalidArgumentError(
          absl::StrFormat("Argument to pragma 'hls_unroll' is not valid. "
                          "Must be 'yes', 'no', or an integer."
                          " At %s:%i",
                          presumed_loc.getFilename(), presumed_loc.getLine()));
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
      parser_.libtool_visit_status_ = absl::InvalidArgumentError(
          absl::StrFormat("Argument '%s' to pragma 'hls_unroll' is not valid. "
                          "Must be 'yes', 'no', or an integer."
                          " At %s:%i",
                          str_identifier, presumed_loc.getFilename(),
                          presumed_loc.getLine()));
      return;
    }
    LOG(WARNING) << "Partial unroll not yet supported: fully unrolling";
    parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                             presumed_loc.getLine())] =
        Pragma(Pragma_Unroll, arg);
  }
  clang::CompilerInstance* compiler_instance_;
  CCParser& parser_;
};

class HlsChannelStrictnessPragmaHandler : public clang::PragmaHandler {
 public:
  explicit HlsChannelStrictnessPragmaHandler(
      clang::CompilerInstance* compiler_instance, CCParser& parser)
      : clang::PragmaHandler("hls_channel_strictness"),
        compiler_instance_(compiler_instance),
        parser_(parser) {}
  void HandlePragma(clang::Preprocessor& PP, clang::PragmaIntroducer Introducer,
                    clang::Token& firstToken) override {
    const clang::PresumedLoc presumed_loc =
        compiler_instance_->getSourceManager().getPresumedLoc(Introducer.Loc);
    clang::Token tok;
    PP.Lex(tok);
    if (tok.is(clang::tok::identifier)) {
      const clang::IdentifierInfo* identifier = tok.getIdentifierInfo();
      parser_.hls_pragmas_[CCParser::PragmaLoc(presumed_loc.getFilename(),
                                               presumed_loc.getLine())] =
          Pragma(Pragma_ChannelStrictness,
                 std::string(identifier->getName().str()));
    }
  }
  clang::CompilerInstance* compiler_instance_;
  CCParser& parser_;
};

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
  void ExecuteAction() override {
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new HlsNoParamPragmaHandler(compiler_instance_, parser_, "hls_top",
                                        Pragma_Top));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new HlsDesignPragmaHandler(compiler_instance_, parser_));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "",
        new HlsPipelineInitIntervalPragmaHandler(compiler_instance_, parser_));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new HlsNoParamPragmaHandler(compiler_instance_, parser_,
                                        "hls_array_allow_default_pad",
                                        Pragma_ArrayAllowDefaultPad));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new HlsNoParamPragmaHandler(compiler_instance_, parser_,
                                        "hls_no_tuple", Pragma_NoTuples));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "",
        new HlsNoParamPragmaHandler(compiler_instance_, parser_,
                                    "hls_synthetic_int", Pragma_SyntheticInt));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new HlsUnrollPragmaHandler(compiler_instance_, parser_));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new HlsChannelStrictnessPragmaHandler(compiler_instance_, parser_));
    getCompilerInstance().getPreprocessor().AddPragmaHandler(
        "", new UnknownPragmaHandler());

    clang::ASTFrontendAction::ExecuteAction();
  }

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
      parser_.libtool_visit_status_ = absl::FailedPreconditionError(
          absl::StrFormat("Unable to parse text with clang (libtooling)\n"));
    }
    // Print the message
    clang::TextDiagnosticPrinter::HandleDiagnostic(level, info);
  }

 private:
  CCParser& parser_;
};

Pragma::Pragma(PragmaType type, int64_t argument)
    : type_(type), int_argument_(argument) {}

Pragma::Pragma(PragmaType type, std::string argument)
    : type_(type), str_argument_(std::move(argument)) {}

Pragma::Pragma(PragmaType type) : type_(type), int_argument_(-1) {}

Pragma::Pragma() : type_(Pragma_Null), int_argument_(-1) {}

PragmaType Pragma::type() const { return type_; }

int64_t Pragma::int_argument() const { return int_argument_; }

std::string_view Pragma::str_argument() const { return str_argument_; }

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

absl::StatusOr<Pragma> CCParser::FindPragmaForLoc(
    const clang::SourceLocation& loc, bool ignore_label) {
  return FindPragmaForLoc(sm_->getPresumedLoc(loc), ignore_label);
}

absl::StatusOr<Pragma> CCParser::FindPragmaForLoc(
    const clang::PresumedLoc& ploc, bool ignore_label) {
  // NOTE: Semantics should be the same as Translator::FindIntrinsicCall()!
  if (!files_scanned_for_pragmas_.contains(ploc.getFilename())) {
    XLS_RETURN_IF_ERROR(ScanFileForPragmas(ploc.getFilename()));
  }
  // Look on the line before.
  PragmaLoc loc(ploc.getFilename(), ploc.getLine() - 1);

  if (!hls_pragmas_.contains(loc)) {
    return Pragma(Pragma_Null);
  }

  // Look for a label there. If found, look at the line before that.
  if (ignore_label && hls_pragmas_.at(loc).type() == Pragma_Label &&
      std::get<1>(loc) > 0) {
    loc = PragmaLoc(ploc.getFilename(), ploc.getLine() - 2);
    if (!hls_pragmas_.contains(loc)) {
      return Pragma(Pragma_Null);
    }
  }

  return hls_pragmas_.at(loc);
}

absl::Status CCParser::ScanFileForPragmas(std::string_view filename) {
  std::ifstream fin(std::string(filename).c_str());
  if (!fin.good()) {
    if (!(filename.empty() || filename[0] == '/')) {
      return absl::NotFoundError(absl::StrFormat(
          "Unable to open file to scan for pragmas: %s\n", filename));
    }
  }
  int lineno = 1;
  PragmaLoc prev_location;
  for (std::string line; std::getline(fin, line); ++lineno) {
    prev_location = PragmaLoc(filename, lineno - 1);
    std::string matched;
    // Ignore blank lines, comments, and lines starting with a label
    if (std::all_of(line.begin(), line.end(), isspace) ||
        RE2::FullMatch(line, "\\s*//.*")) {
      if (hls_pragmas_.contains(prev_location)) {
        Pragma found_value = hls_pragmas_[prev_location];
        hls_pragmas_[PragmaLoc(filename, lineno)] = found_value;
        continue;
      }
    }
    if (RE2::PartialMatch(line, "(\\w+)[\\t\\s]*\\:", &matched)) {
      hls_pragmas_[PragmaLoc(filename, lineno)] = Pragma(Pragma_Label, matched);
      continue;
    }
  }
  files_scanned_for_pragmas_.insert(static_cast<std::string>(filename));
  return absl::OkStatus();
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

  XLS_ASSIGN_OR_RETURN(
      Pragma pragma,
      FindPragmaForLoc(GetPresumedLoc(*funcdecl), /*ignore_label=*/true));

  if (pragma.type() == Pragma_Top || fname == top_function_name_) {
    if (top_function_ == nullptr) {
      top_function_ = funcdecl;
    } else if (fname == top_function_name_) {
      // Already found a function with the top name
      if (top_function_->getNameAsString() == top_function_name_) {
        return absl::AlreadyExistsError(absl::StrFormat(
            "Two top functions defined by name, at %s, previously at %s",
            LocString(GetLoc(*top_function_)), LocString(GetLoc(*funcdecl))));
      }
      // Name takes precedence over pragma
      top_function_ = funcdecl;
    } else if (pragma.type() == Pragma_Top) {
      // If the name doesn't match the top, then it was pragma specified
      if (top_function_->getNameAsString() != top_function_name_) {
        return absl::AlreadyExistsError(absl::StrFormat(
            "Two top functions defined by pragma, at %s, previously at %s",
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
