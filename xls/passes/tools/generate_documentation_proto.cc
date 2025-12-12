// Copyright 2025 The XLS Authors
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

#include "xls/passes/tools/generate_documentation_proto.h"

#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/ASTConsumer.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/RawCommentList.h"
#include "clang/include/clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/include/clang/ASTMatchers/ASTMatchers.h"
#include "clang/include/clang/Basic/FileManager.h"
#include "clang/include/clang/Basic/FileSystemOptions.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendAction.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "llvm/include/llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/include/llvm/Support/VirtualFileSystem.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/tools/pass_documentation.pb.h"

namespace xls {

namespace {

static constexpr std::string_view kLlvmHeaders = "clang/staging/include";
namespace m = clang::ast_matchers;

// Find the actual matches. Use the exact class comment unless there isn't one
// then just use the first comment on any OptimizationPass in the file (assuming
// it's an ABC).
class Callback : public m::MatchFinder::MatchCallback {
 public:
  void run(const m::MatchFinder::MatchResult& result) override {
    if (const clang::CXXRecordDecl* clz =
            result.Nodes.getNodeAs<clang::CXXRecordDecl>("base_opt_pass")) {
      if (!doc_any_) {
        doc_any_ = GetCommentFromMatch(result.Context, clz);
      } else if (VLOG_IS_ON(3)) {
        std::string dump;
        llvm::raw_string_ostream oss(dump);
        clz->dump(oss, true);
        VLOG(3) << "Already found a result of " << *doc_any_
                << " so ignoring later match of " << dump;
      }
    }
    if (const clang::CXXRecordDecl* clz =
            result.Nodes.getNodeAs<clang::CXXRecordDecl>("exact_opt_pass")) {
      if (!doc_exact_) {
        doc_exact_ = GetCommentFromMatch(result.Context, clz);
      } else if (VLOG_IS_ON(3)) {
        std::string dump;
        llvm::raw_string_ostream oss(dump);
        clz->dump(oss, true);
        VLOG(3) << "Already found a result of " << *doc_exact_
                << " so ignoring later match of " << dump;
      }
    }
  }
  const std::optional<std::string>& doc() const {
    return doc_exact_ && !doc_exact_->empty() ? doc_exact_ : doc_any_;
  }

 private:
  std::string GetCommentFromMatch(const clang::ASTContext* context,
                                  const clang::CXXRecordDecl* clz) {
    const clang::RawComment* comment = context->getRawCommentForAnyRedecl(clz);
    if (!comment) {
      return "";
    }
    return comment->getFormattedText(context->getSourceManager(),
                                     context->getDiagnostics());
  }
  std::optional<std::string> doc_any_;
  std::optional<std::string> doc_exact_;
};

absl::StatusOr<std::string> ParseCppComments(
    std::string_view h_file, std::string_view class_name,
    absl::Span<std::string const> copts) {
  std::vector<std::string> argv = {
      "generate_documentation_proto",
      // Force c++ mode.
      "-x",
      "c++",
      // Actual header to compile.
      std::string(h_file),
      // This flag makes documentation-like comments be linked directly to the
      // ast element they are nearest.
      // "-w",
      "-std=gnu++20",
      // Actually parse the comments.
      "-fparse-all-comments",
      "-fsyntax-only",
      "-o",
      "/dev/null",
  };
  // Allow a fallback of any base-class.
  auto matcher_base =
      m::cxxRecordDecl(m::isDerivedFrom(m::hasName("xls::OptimizationPass")),
                       m::isExpansionInMainFile())
          .bind("base_opt_pass");
  auto matcher_exact =
      m::cxxRecordDecl(m::isDerivedFrom(m::hasName("xls::OptimizationPass")),
                       m::hasName(class_name), m::isExpansionInMainFile())
          .bind("exact_opt_pass");
  absl::c_copy(copts, std::back_inserter(argv));
  // At the very end of the include search list we need to put the llvm base
  // headers. Normally these would be auto-populated so the cmdline doesn't
  // include them but because we aren't actually clang we need to supply them
  // ourselves. These must be after all other include flags in argv so just put
  // them at the very end.
  // TODO(allight): GetXlsRunfilePath doesn't support folders. Just ask for a
  // specific file and get the parent.
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path runfile_path,
      GetXlsRunfilePath(std::string(kLlvmHeaders) + "/stddef.h",
                        /*package=*/"llvm-project"));
  argv.push_back(absl::StrFormat("-isystem%s", runfile_path.parent_path()));

  llvm::IntrusiveRefCntPtr<clang::FileManager> libtool_files;
  m::MatchFinder finder;
  Callback cb;
  finder.addMatcher(matcher_base, &cb);
  finder.addMatcher(matcher_exact, &cb);
  std::unique_ptr<clang::FrontendAction> libtool_action =
      clang::tooling::newFrontendActionFactory(&finder)->create();

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));

  libtool_files =
      new clang::FileManager(clang::FileSystemOptions(), overlay_fs);

  std::unique_ptr<clang::tooling::ToolInvocation> libtool_inv(
      new clang::tooling::ToolInvocation(argv, std::move(libtool_action),
                                         libtool_files.get()));

  // Run the actual tool.
  // Errors are suppressed by -w.
  XLS_RET_CHECK(libtool_inv->run()) << "bad run";
  if (!cb.doc()) {
    LOG(WARNING)
        << "Unable to find comment for a OptimizationPass in header file.";
  }
  // Pull the results out of the callback.
  return cb.doc().value_or("");
}

}  // namespace
absl::StatusOr<PassDocumentationProto> GenerateDocumentationProto(
    const OptimizationPassRegistryBase& registry, std::string_view header,
    absl::Span<std::string const> copts) {
  std::vector<OptimizationPassRegistryBase::RegistrationInfo> infos =
      registry.GetRegisteredInfos();
  xls::PassDocumentationProto res;
  VLOG(5) << "found " << infos.size() << "infos!";
  VLOG(5) << "Looking for " << header;
  for (const auto& info : infos) {
    if (info.header_file == header) {
      VLOG(4) << "Found header file " << header;
      PassDocumentationProto::OnePass* pass = res.mutable_passes()->Add();
      XLS_ASSIGN_OR_RETURN(auto gen, registry.Generator(info.short_name));
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> real_pass,
                           gen->Generate());
      // TODO(allight): Unfortunately we don't really distinguish between
      // multiple passes in a single file.
      XLS_ASSIGN_OR_RETURN(
          *pass->mutable_notes(),
          ParseCppComments(info.header_file, info.class_name, copts));
      *pass->mutable_file() = info.header_file;
      *pass->mutable_short_name() = info.short_name;
      *pass->mutable_long_name() = real_pass->long_name();
    } else {
      VLOG(4) << "Didn't examine header " << info.header_file;
    }
  }
  return res;
}

}  // namespace xls
