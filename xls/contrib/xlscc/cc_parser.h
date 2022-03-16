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

#ifndef XLS_CONTRIB_XLSCC_PARSE_CPP_H_
#define XLS_CONTRIB_XLSCC_PARSE_CPP_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendActions.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Tooling/CommonOptionsParser.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "xls/common/thread.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/ir/source_location.h"

namespace xlscc {

enum PragmaType {
  Pragma_Null = 0,
  Pragma_NoTuples,
  Pragma_Unroll,
  Pragma_Top,
  Pragma_InitInterval,
  Pragma_Label,
};

class Pragma {
 private:
  PragmaType type_;
  int64_t int_argument_ = -1;
  std::string str_argument_ = "";

 public:
  Pragma(PragmaType type, int64_t argument);
  Pragma(PragmaType type, std::string argument);
  Pragma(PragmaType type);
  Pragma();

  PragmaType type() const;
  int64_t int_argument() const;
  std::string_view str_argument() const;
};

class CCParser;
class LibToolVisitor;
class DiagnosticInterceptor;
class LibToolFrontendAction;

// Needs to be here because std::unique uses sizeof()
class LibToolThread {
 public:
  LibToolThread(absl::string_view source_filename,
                absl::Span<absl::string_view> command_line_args,
                CCParser& parser);

  void Start();
  void Join();

 private:
  void Run();

  absl::optional<xls::Thread> thread_;
  absl::string_view source_filename_;
  absl::Span<absl::string_view> command_line_args_;
  CCParser& parser_;
};

// Parses and then holds ownership of a C++ AST
class CCParser {
  friend class LibToolVisitor;
  friend class DiagnosticInterceptor;
  friend class LibToolFrontendAction;

 public:
  // Deletes the AST
  ~CCParser();

  // Selects a top, or entry, function by name
  // Either this must be called before ScanFile, or a #pragma hls_top
  // must be present in the file(s) being scanned, in order for
  // GetTopFunction() to return a valid pointer
  absl::Status SelectTop(absl::string_view top_function_name);

  // This function uses Clang to parse a source file and then walks its
  //  AST to discover global constructs. It will also scan the file
  //  and includes, recursively, for #pragma statements.
  //
  // Among these are functions, which can be used as entry points
  //  for translation to IR.
  //
  // source_filename must be .cc
  // Retains references to the TU until ~Translator()
  // This function may only be called once in the lifetime of a CCParser.
  absl::Status ScanFile(absl::string_view source_filename,
                        absl::Span<absl::string_view> command_line_args);

  // Call after ScanFile, as the top function may be specified by #pragma
  // If none was found, an error is returned
  absl::StatusOr<std::string> GetEntryFunctionName() const;

  // Returns a pointer into the AST for the top, or entry, function
  // Returns nullptr if no top function has been found
  absl::StatusOr<const clang::FunctionDecl*> GetTopFunction() const;

  void AddSourceInfoToMetadata(xlscc_metadata::MetadataOutput& output);

  absl::StatusOr<Pragma> FindPragmaForLoc(const clang::PresumedLoc& ploc);

  xls::SourceLocation GetLoc(clang::SourceManager& sm, const clang::Stmt& stmt);
  clang::PresumedLoc GetPresumedLoc(const clang::SourceManager& sm,
                                    const clang::Stmt& stmt);
  xls::SourceLocation GetLoc(const clang::Decl& decl);
  clang::PresumedLoc GetPresumedLoc(const clang::Decl& decl);
  xls::SourceLocation GetLoc(clang::SourceManager& sm, const clang::Expr& expr);
  xls::SourceLocation GetLoc(const clang::PresumedLoc& loc);
  std::string LocString(const xls::SourceLocation& loc);

 private:
  bool LibToolVisitFunction(clang::FunctionDecl* func);
  absl::Status libtool_visit_status_ = absl::OkStatus();

  std::unique_ptr<LibToolThread> libtool_thread_;
  std::unique_ptr<absl::BlockingCounter> libtool_wait_for_parse_;
  std::unique_ptr<absl::BlockingCounter> libtool_wait_for_destruct_;

  // Scans for top-level function candidates
  absl::Status VisitFunction(const clang::FunctionDecl* funcdecl);
  absl::Status ScanFileForPragmas(absl::string_view filename);

  using PragmaLoc = std::tuple<std::string, int>;
  absl::flat_hash_map<PragmaLoc, Pragma> hls_pragmas_;
  absl::flat_hash_set<std::string> files_scanned_for_pragmas_;

  const clang::FunctionDecl* top_function_ = nullptr;
  absl::string_view top_function_name_ = "";

  // For source location
  absl::flat_hash_map<std::string, int> file_numbers_;
  int next_file_number_ = 1;
};

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_PARSE_CPP_H_
