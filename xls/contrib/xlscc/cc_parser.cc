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
#include <string_view>

#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "re2/re2.h"

namespace xlscc {

class LibToolVisitor : public clang::RecursiveASTVisitor<LibToolVisitor> {
 public:
  explicit LibToolVisitor(clang::CompilerInstance& CI, CCParser& translator)
      : ast_context_(&(CI.getASTContext())), parser_(translator) {}
  virtual ~LibToolVisitor() {}
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

  auto found = file_numbers_.find(loc.getFilename());

  int id = 0;

  if (found == file_numbers_.end()) {
    id = next_file_number_++;
    file_numbers_[loc.getFilename()] = id;
  } else {
    id = found->second;
  }

  return xls::SourceInfo(xls::SourceLocation(xls::Fileno(id),
                                             xls::Lineno(loc.getLine()),
                                             xls::Colno(loc.getColumn())));
}

absl::StatusOr<Pragma> CCParser::FindPragmaForLoc(
    const clang::SourceLocation& loc) {
  return FindPragmaForLoc(sm_->getPresumedLoc(loc));
}

absl::StatusOr<Pragma> CCParser::FindPragmaForLoc(
    const clang::PresumedLoc& ploc) {
  if (!files_scanned_for_pragmas_.contains(ploc.getFilename())) {
    XLS_RETURN_IF_ERROR(ScanFileForPragmas(ploc.getFilename()));
  }
  // Look on the line before.
  PragmaLoc loc(ploc.getFilename(), ploc.getLine() - 1);
  if (!hls_pragmas_.contains(loc)) {
    return Pragma(Pragma_Null);
  }

  // Look for a label there. If found, look at the line before that.
  if (hls_pragmas_.at(loc).type() == Pragma_Label && std::get<1>(loc) > 0) {
    loc = PragmaLoc(ploc.getFilename(), ploc.getLine() - 2);
    if (!hls_pragmas_.contains(loc)) {
      return Pragma(Pragma_Null);
    }
  }
  return hls_pragmas_.at(loc);
}

static size_t match_pragma(std::string_view pragma_string,
                           std::string_view name) {
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

absl::Status CCParser::ScanFileForPragmas(std::string_view filename) {
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
    const PragmaLoc location(filename, lineno);
    if ((at = line.find("#pragma")) != std::string::npos) {
      if ((at = match_pragma(line, "#pragma hls_no_tuple")) !=
          std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_NoTuples);
      } else if ((at = match_pragma(line, "#pragma hls_unroll yes")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_Unroll);
      } else if ((at = match_pragma(line,
                                    "#pragma hls_array_allow_default_pad")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_ArrayAllowDefaultPad);
      } else if ((at = match_pragma(line, "#pragma hls_top")) !=
                     std::string::npos ||
                 (at = match_pragma(line, "#pragma hls_design top")) !=
                     std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_Top);
      } else if ((at = match_pragma(line, "#pragma hls_synthetic_int")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma(Pragma_SyntheticInt);
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
    } else {
      std::string matched;
      if (RE2::PartialMatch(line, "(\\w+)[\\t\\s]*\\:", &matched)) {
        hls_pragmas_[location] = Pragma(Pragma_Label, matched);
      }
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
  if (libtool_visit_status_.ok()) libtool_visit_status_ = VisitFunction(func);
  return libtool_visit_status_.ok();
}

bool CCParser::LibToolVisitVarDecl(clang::VarDecl* func) {
  if (libtool_visit_status_.ok()) libtool_visit_status_ = VisitVarDecl(func);
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

  XLS_ASSIGN_OR_RETURN(Pragma pragma,
                       FindPragmaForLoc(GetPresumedLoc(*funcdecl)));

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
    XLS_CHECK(xlscc_on_reset_ == nullptr || xlscc_on_reset_ == decl);
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
                             absl::Span<std::string_view> command_line_args,
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
  __xls_channel_dir_In=2          // OpType::kRecv
};

template<typename T, __xls_channel_dir Dir=__xls_channel_dir_Unknown>

class __xls_channel {
 public:
  T read() {
    return T();
  }
  T write(T val) {
    return val;
  }
  void read(T& out) {
    (void)out;
  }
  bool nb_read(T& out) {
    (void)out;
    return true;
  }
};

// Bypass no outputs error
int __xlscc_unimplemented() { return 0; }

bool __xlscc_on_reset = false;

#endif//__XLS_BUILTIN_H
          )"));

  const std::string top_src = absl::StrFormat(R"(
#include "/xls_builtin.h"
#include "%s"
          )",
                                              source_filename_);

  mem_fs->addFile("/xls_top.cc", 0,
                  llvm::MemoryBuffer::getMemBuffer(top_src.c_str()));

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

absl::Status CCParser::SelectTop(std::string_view top_function_name) {
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
