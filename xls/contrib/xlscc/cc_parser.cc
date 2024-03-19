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
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/ASTConsumer.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclBase.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/Basic/Diagnostic.h"
#include "clang/include/clang/Basic/FileSystemOptions.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendAction.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "llvm/include/llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/VirtualFileSystem.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/logging/logging.h"
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
  explicit LibToolVisitor(clang::CompilerInstance& CI, CCParser& translator)
      : ast_context_(&(CI.getASTContext())), parser_(translator) {}
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
  libtool_thread_ = absl::WrapUnique(new LibToolThread(
      source_filename, top_class_name_, command_line_args, *this));

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

namespace {
struct PragmaLine {
  bool is_comment = false;
  std::string_view pragma_name;
  std::vector<std::string_view> args;
};
std::optional<PragmaLine> ExtractPragma(std::string_view line) {
  PragmaLine result;
  enum class TokensExpected : int8_t {
    kPragmaOrComment,
    kHlsPragmaName,
    kArgs,
  };
  TokensExpected expected = TokensExpected::kPragmaOrComment;
  for (std::string_view tok : absl::StrSplit(line, ' ', absl::SkipEmpty())) {
    if (expected == TokensExpected::kPragmaOrComment) {
      if (tok == "#pragma") {
        expected = TokensExpected::kHlsPragmaName;
      } else if (absl::StartsWith(tok, "//")) {
        // Full line comment.
        result.is_comment = true;
        return result;
      } else {
        return std::nullopt;
      }
    } else if (expected == TokensExpected::kHlsPragmaName) {
      expected = TokensExpected::kArgs;
      result.pragma_name = tok;
    } else if (absl::StartsWith(tok, "//")) {
      // A comment. Ignore the rest of the line.
      break;
    } else {
      result.args.push_back(tok);
    }
  }
  return expected == TokensExpected::kArgs ? std::make_optional(result)
                                           : std::nullopt;
}
}  // namespace

absl::Status CCParser::ScanFileForPragmas(std::string_view filename) {
  std::ifstream fin(std::string(filename).c_str());
  if (!fin.good()) {
    if (!(filename.empty() || filename[0] == '/')) {
      return absl::NotFoundError(absl::StrFormat(
          "Unable to open file to scan for pragmas: %s\n", filename));
    }
  }
  std::map<std::string, PragmaType> pragmas = {
      {"hls_array_allow_default_pad", Pragma_ArrayAllowDefaultPad},
      {"hls_design", Pragma_Top},
      {"hls_no_tuple", Pragma_NoTuples},
      {"hls_pipeline_init_interval", Pragma_InitInterval},
      {"hls_synthetic_int", Pragma_SyntheticInt},
      {"hls_top", Pragma_Top},
      {"hls_unroll", Pragma_Unroll},
      {"hls_channel_strictness", Pragma_ChannelStrictness},
  };
  int lineno = 1;
  PragmaLoc prev_location;
  for (std::string line; std::getline(fin, line); ++lineno) {
    if (std::all_of(line.begin(), line.end(), isspace)) {
      if (hls_pragmas_.find(prev_location) != hls_pragmas_.end()) {
        Pragma found_value = hls_pragmas_[prev_location];
        hls_pragmas_[PragmaLoc(filename, lineno)] = found_value;
        continue;
      }
    }
    std::optional<PragmaLine> pragma_line = ExtractPragma(line);
    if (pragma_line) {
      if (pragma_line->is_comment) {
        if (hls_pragmas_.contains(prev_location)) {
          Pragma found_value = hls_pragmas_[prev_location];
          hls_pragmas_[PragmaLoc(filename, lineno)] = found_value;
        }
        continue;
      }
      std::string_view name = pragma_line->pragma_name;

      std::map<std::string, PragmaType>::iterator it =
          pragmas.find(absl::AsciiStrToLower(name));
      if (it != pragmas.end()) {
        if (name != absl::AsciiStrToLower(name)) {
          LOG(WARNING) << "#pragma must be lowercase: " << line;
          continue;
        }
        if (!absl::StartsWith(name, "hls_")) {
          LOG(WARNING) << "WARNING: #pragma '" << name
                       << "' requires 'hls_' prefix";
          prev_location = PragmaLoc("", -1);
          continue;
        }
        // Snip out hls_
        std::string_view short_name = name.substr(4);
        if (pragma_line->args.size() > 1) {
          LOG(WARNING) << "WARNING: #pragma '" << short_name
                       << "' has exra arguments '"
                       << absl::StrJoin(
                              absl::MakeConstSpan(pragma_line->args).subspan(1),
                              ",")
                       << "' that will be ignored";
        }
        const PragmaLoc location(filename, lineno);
        prev_location = location;
        PragmaType pragma_val = it->second;
        int64_t arg = -1;
        std::string_view params =
            pragma_line->args.empty() ? "" : pragma_line->args.front();
        switch (pragma_val) {
          case Pragma_ArrayAllowDefaultPad:
            hls_pragmas_[location] = Pragma(pragma_val);
            break;
          case Pragma_InitInterval:
            if (!absl::SimpleAtoi(params, &arg) || (arg <= 0)) {
              return absl::InvalidArgumentError(
                  absl::StrFormat("Argument '%s' to pragma '%s' is not valid. "
                                  "Must be an integer "
                                  ">= 1. At %s:%i",
                                  params, short_name, filename, lineno));
            }

            hls_pragmas_[location] = Pragma(pragma_val, arg);
            break;
          case Pragma_NoTuples:
            hls_pragmas_[location] = Pragma(pragma_val);
            break;
          case Pragma_SyntheticInt:
            hls_pragmas_[location] = Pragma(pragma_val);
            break;
          case Pragma_Top: {
            if (name == "hls_design") {
              if (params == "block") {
                hls_pragmas_[location] = Pragma(Pragma_Block);
              } else if (params == "top") {
                hls_pragmas_[location] = Pragma(Pragma_Top);
              } else {
                LOG(WARNING)
                    << "Ignoring unknown #pragma hls_design: " << params;
              }
            } else {
              hls_pragmas_[location] = Pragma(Pragma_Top);
            }
            break;
          }
          case Pragma_Unroll:
            if (params.empty() || params == "yes") {
              hls_pragmas_[location] = Pragma(pragma_val);
              break;
            }
            if (params == "no") {
              LOG(WARNING) << "Ignoring #pragma hls_unroll no (at " << filename
                           << ":" << lineno
                           << "). Pragma is "
                              "not needed and has no effect.";
              break;
            }
            if (!absl::SimpleAtoi(params, &arg) || (arg <= 0)) {
              if (arg == 0) {
                LOG(WARNING) << "Ignoring #pragma hls_unroll 0 (at " << filename
                             << ":" << lineno
                             << "). Pragma is "
                                "not needed and has no effect.";
                break;
              }
              return absl::InvalidArgumentError(
                  absl::StrFormat("Argument '%s' to pragma '%s' is not valid. "
                                  "Must be 'yes', 'no', or an integer."
                                  " At %s:%i",
                                  params, short_name, filename, lineno));
            }
            LOG(WARNING) << "Partial unroll not yet supported: "
                         << "fully unrolling";
            hls_pragmas_[location] = Pragma(pragma_val, arg);
            break;
          case Pragma_Null:
          case Pragma_Label:
            prev_location = PragmaLoc("", -1);
            break;
          case Pragma_Block:
            hls_pragmas_[location] = Pragma(pragma_val);
            break;
          case Pragma_ChannelStrictness:
            hls_pragmas_[location] = Pragma(pragma_val, std::string(params));
            break;
        }
      } else if (pragmas.find(absl::StrCat(
                     "hls_", absl::AsciiStrToLower(name))) != pragmas.end()) {
        LOG(WARNING) << "WARNING: #pragma '" << name
                     << "' requires 'hls_' prefix";
        prev_location = PragmaLoc("", -1);
      }
      // Ignore unknown pragmas
    } else {
      if (RE2::FullMatch(line, "\\s*//.*")) {
        if (hls_pragmas_.contains(prev_location)) {
          Pragma found_value = hls_pragmas_[prev_location];
          hls_pragmas_[PragmaLoc(filename, lineno)] = found_value;
          continue;
        }
      }
      prev_location = PragmaLoc("", -1);
      const PragmaLoc location(filename, lineno);
      std::string matched;
      if (RE2::PartialMatch(line, "(\\w+)[\\t\\s]*\\:", &matched)) {
        hls_pragmas_[location] = Pragma(Pragma_Label, matched);
        prev_location = location;
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
