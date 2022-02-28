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

#include <fstream>
#include <iostream>

#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendActions.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Tooling/CommonOptionsParser.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"

namespace xlscc {

class LibToolVisitor : public clang::RecursiveASTVisitor<LibToolVisitor> {
 public:
  explicit LibToolVisitor(clang::CompilerInstance& CI, CCParser& translator)
      : ast_context_(&(CI.getASTContext())), parser_(translator) {}
  virtual ~LibToolVisitor() {}
  virtual bool VisitFunctionDecl(clang::FunctionDecl* func) {
    return parser_.LibToolVisitFunction(func);
  }

 private:
  clang::ASTContext* ast_context_;
  CCParser& parser_;
};
class LibToolASTConsumer : public clang::ASTConsumer {
 public:
  explicit LibToolASTConsumer(clang::CompilerInstance& CI, CCParser& translator)
      : visitor_(new LibToolVisitor(CI, translator)) {}

  void HandleTranslationUnit(clang::ASTContext& Context) override {
    visitor_->TraverseDecl(Context.getTranslationUnitDecl());
  }

 private:
  std::unique_ptr<LibToolVisitor> visitor_;
};
class LibToolFrontendAction : public clang::ASTFrontendAction {
 public:
  explicit LibToolFrontendAction(CCParser& translator) : parser_(translator) {}
  void EndSourceFileAction() override;
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& CI, clang::StringRef /*file*/) override {
    return std::unique_ptr<clang::ASTConsumer>(
        new LibToolASTConsumer(CI, parser_));
  }

 private:
  CCParser& parser_;
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
    : type_(type), argument_(argument) {}

Pragma::Pragma(PragmaType type) : type_(type), argument_(-1) {}

Pragma::Pragma() : type_(Pragma_Null), argument_(-1) {}

PragmaType Pragma::type() const { return type_; }

int64_t Pragma::argument() const { return argument_; }

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
    absl::string_view source_filename,
    absl::Span<absl::string_view> command_line_args) {
  // This function may only be called once in the lifetime of a CCParser.
  XLS_CHECK_EQ(libtool_thread_.get(), nullptr);
  XLS_CHECK_EQ(libtool_wait_for_destruct_.get(), nullptr);

  // The AST is destroyed after ToolInvocation::run() returns
  //
  // However, we want to preserve it to access it across multiple passes and
  //  various subsequent calls, such as GenerateIR().
  //
  // Therefore, ToolInvocation::Run() is executed on another thread,
  //  and the ASTFrontendAction::EndSourceFileAction() blocks it
  //  until ~CCParser(), preserving the AST.
  libtool_thread_ = absl::WrapUnique(
      new LibToolThread(source_filename, command_line_args, *this));

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

clang::PresumedLoc CCParser::GetPresumedLoc(const clang::SourceManager& sm,
                                            const clang::Stmt& stmt) {
  return sm.getPresumedLoc(stmt.getSourceRange().getBegin());
}

xls::SourceLocation CCParser::GetLoc(clang::SourceManager& sm,
                                     const clang::Stmt& stmt) {
  return GetLoc(GetPresumedLoc(sm, stmt));
}

clang::PresumedLoc CCParser::GetPresumedLoc(const clang::Decl& decl) {
  clang::SourceManager& sm = decl.getASTContext().getSourceManager();
  return sm.getPresumedLoc(decl.getSourceRange().getBegin());
}

xls::SourceLocation CCParser::GetLoc(const clang::Decl& decl) {
  return GetLoc(GetPresumedLoc(decl));
}

xls::SourceLocation CCParser::GetLoc(clang::SourceManager& sm,
                                     const clang::Expr& expr) {
  return GetLoc(sm.getPresumedLoc(expr.getExprLoc()));
}

xls::SourceLocation CCParser::GetLoc(const clang::PresumedLoc& loc) {
  if (!loc.isValid()) {
    return xls::SourceLocation(xls::Fileno(-1), xls::Lineno(-1),
                               xls::Colno(-1));
  }

  auto found = file_numbers_.find(loc.getFilename());

  int id = 0;

  if (found == file_numbers_.end()) {
    id = next_file_number_++;
    file_numbers_[loc.getFilename()] = id;
  } else {
    id = found->second;
  }

  return xls::SourceLocation(xls::Fileno(id), xls::Lineno(loc.getLine()),
                             xls::Colno(loc.getColumn()));
}

absl::StatusOr<Pragma> CCParser::FindPragmaForLoc(
    const clang::PresumedLoc& ploc) {
  if (!files_scanned_for_pragmas_.contains(ploc.getFilename())) {
    XLS_RETURN_IF_ERROR(ScanFileForPragmas(ploc.getFilename()));
  }
  // Look on the line before
  auto found =
      hls_pragmas_.find(PragmaLoc(ploc.getFilename(), ploc.getLine() - 1));
  if (found == hls_pragmas_.end()) return Pragma(Pragma_Null);
  return found->second;
}

static size_t match_pragma(absl::string_view pragma_string,
                           absl::string_view name) {
  size_t at = pragma_string.find(name);
  if (at == std::string::npos) return std::string::npos;
  size_t lcs = pragma_string.find("//");
  if (lcs != std::string::npos) {
    if (lcs < at) return std::string::npos;
  }
  size_t bcs = pragma_string.find("*/");
  if (bcs != std::string::npos) {
    if (bcs >= (at + name.length())) return std::string::npos;
  }
  return at;
}

absl::Status CCParser::ScanFileForPragmas(absl::string_view filename) {
  std::ifstream fin(std::string(filename).c_str());
  if (!fin.good()) {
    if (filename != "/xls_builtin.h") {
      return absl::NotFoundError(absl::StrFormat(
          "Unable to open file to scan for pragmas: %s\n", filename));
    }
  }
  const std::string init_interval_pragma = "#pragma hls_pipeline_init_interval";
  int lineno = 1;
  for (std::string line; std::getline(fin, line); ++lineno) {
    size_t at;
    if ((at = line.find("#pragma")) != std::string::npos) {
      PragmaLoc location(filename, lineno);

      if ((at = match_pragma(line, "#pragma hls_no_tuple")) !=
          std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_NoTuples);
      } else if ((at = match_pragma(line, "#pragma hls_unroll yes")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_Unroll);
      } else if ((at = match_pragma(line, "#pragma hls_top")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_Top);
      } else if ((at = match_pragma(line, init_interval_pragma)) !=
                 std::string::npos) {
        const std::string after_pragma = line.substr(
            line.find_first_of('#') + init_interval_pragma.length());
        int64_t arg = -1;
        if (!absl::SimpleAtoi(after_pragma, &arg) || (arg <= 0)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Argument '%s' to pragma '%s' is not valid. Must be an integer "
              ">= 1. At %s:%i",
              after_pragma, init_interval_pragma, filename, lineno));
        }
        hls_pragmas_[location] = Pragma(Pragma_InitInterval, arg);
      }
      // Ignore unknown pragmas
    }
  }

  files_scanned_for_pragmas_.insert(static_cast<std::string>(filename));
  return absl::OkStatus();
}

std::string CCParser::LocString(const xls::SourceLocation& loc) {
  std::string found_str = "Unknown";
  for (const auto& it : file_numbers_) {
    if (it.second == static_cast<int>(loc.fileno())) {
      found_str = it.first;
      break;
    }
  }
  return absl::StrFormat("%s:%i:%i", found_str, static_cast<int>(loc.lineno()),
                         static_cast<int>(loc.fileno()));
}

bool CCParser::LibToolVisitFunction(clang::FunctionDecl* func) {
  if (libtool_visit_status_.ok()) libtool_visit_status_ = VisitFunction(func);
  return libtool_visit_status_.ok();
}

// Scans for top-level function candidates
absl::Status CCParser::VisitFunction(const clang::FunctionDecl* funcdecl) {
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

  XLS_ASSIGN_OR_RETURN(Pragma pragma,
                       FindPragmaForLoc(GetPresumedLoc(*funcdecl)));

  if (pragma.type() == Pragma_Top || fname == top_function_name_) {
    if (top_function_) {
      return absl::AlreadyExistsError(absl::StrFormat(
          "Two top functions defined, at %s, previously at %s",
          LocString(GetLoc(*top_function_)), LocString(GetLoc(*funcdecl))));
    }
    top_function_ = funcdecl;
  }

  return absl::OkStatus();
}

LibToolThread::LibToolThread(absl::string_view source_filename,
                             absl::Span<absl::string_view> command_line_args,
                             CCParser& parser)
    : source_filename_(source_filename),
      command_line_args_(command_line_args),
      parser_(parser) {}

void LibToolThread::Start() {
  thread_.emplace([this] { Run(); });
}

void LibToolThread::Join() { thread_->Join(); }

void LibToolThread::Run() {
  std::vector<std::string> argv;
  argv.emplace_back("binary");
  argv.emplace_back(source_filename_);
  for (const auto& view : command_line_args_) {
    argv.emplace_back(view);
  }
  argv.emplace_back("-fsyntax-only");
  argv.emplace_back("-std=c++17");
  argv.emplace_back("-nostdinc");
  argv.emplace_back("-Wno-unused-label");
  argv.emplace_back("-Wno-constant-logical-operand");

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

template<typename T>
class __xls_channel {
 public:
  T read() {
    return T();
  }
  T write(T val) {
    return val;
  }
};

// Bypass no outputs error
int __xlscc_unimplemented() { return 0; }

#endif//__XLS_BUILTIN_H
          )"));

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
  if (!top_function_) {
    return absl::NotFoundError("No top function found");
  }
  return top_function_->getNameAsString();
}

absl::Status CCParser::SelectTop(absl::string_view top_function_name) {
  top_function_name_ = top_function_name;
  return absl::OkStatus();
}

absl::StatusOr<const clang::FunctionDecl*> CCParser::GetTopFunction() const {
  if (top_function_ == nullptr) {
    return absl::NotFoundError("No top function found");
  }
  return top_function_;
}

}  // namespace xlscc
