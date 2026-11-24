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

#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
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

constexpr std::string_view kLlvmHeaders = "clang/staging/include";
namespace m = clang::ast_matchers;

struct ParsedHeaderComments {
  std::string repo_path;
  bool matched_base = false;
  std::optional<std::string> first_nonempty_base_comment;
  absl::flat_hash_set<std::string> matched_exact;
  absl::flat_hash_map<std::string, std::string> exact_comment;
};

std::string GetCommentFromMatch(const clang::ASTContext* context,
                                const clang::CXXRecordDecl* clz) {
  const clang::RawComment* comment = context->getRawCommentForAnyRedecl(clz);
  if (comment == nullptr) {
    return "";
  }
  return comment->getFormattedText(context->getSourceManager(),
                                   context->getDiagnostics());
}

// Collects exact class matches and OptimizationPass base-class matches within a
// single header parse.
class Callback : public m::MatchFinder::MatchCallback {
 public:
  explicit Callback(absl::Span<std::string const> class_names,
                    ParsedHeaderComments& parsed)
      : class_names_(class_names), parsed_(parsed) {}

  void run(const m::MatchFinder::MatchResult& result) override {
    if (const clang::CXXRecordDecl* clz =
            result.Nodes.getNodeAs<clang::CXXRecordDecl>("base_opt_pass")) {
      parsed_.matched_base = true;
      std::string comment = GetCommentFromMatch(result.Context, clz);
      if (!comment.empty()) {
        if (!parsed_.first_nonempty_base_comment.has_value()) {
          parsed_.first_nonempty_base_comment = std::move(comment);
        } else if (VLOG_IS_ON(3)) {
          std::string dump;
          llvm::raw_string_ostream oss(dump);
          clz->dump(oss, true);
          VLOG(3) << "Already found a base comment of "
                  << *parsed_.first_nonempty_base_comment
                  << " so ignoring later match of " << dump;
        }
      }
    }
    for (int64_t i = 0; i < static_cast<int64_t>(class_names_.size()); ++i) {
      std::string bind_id = absl::StrFormat("exact_opt_pass_%d", i);
      if (const clang::CXXRecordDecl* clz =
              result.Nodes.getNodeAs<clang::CXXRecordDecl>(bind_id)) {
        const std::string& class_name = class_names_[i];
        parsed_.matched_exact.insert(class_name);
        std::string comment = GetCommentFromMatch(result.Context, clz);
        if (!comment.empty()) {
          if (!parsed_.exact_comment.contains(class_name)) {
            parsed_.exact_comment.emplace(class_name, std::move(comment));
          } else if (VLOG_IS_ON(3)) {
            std::string dump;
            llvm::raw_string_ostream oss(dump);
            clz->dump(oss, true);
            VLOG(3) << "Already found an exact comment for " << class_name
                    << " so ignoring later match of " << dump;
          }
        }
      }
    }
  }

 private:
  absl::Span<std::string const> class_names_;
  ParsedHeaderComments& parsed_;
};

absl::StatusOr<ParsedHeaderComments> ParseCppCommentsForHeader(
    const PassHeaderPath& header, absl::Span<std::string const> class_names,
    absl::Span<std::string const> copts,
    const std::filesystem::path& llvm_include_dir) {
  std::vector<std::string> argv = {
      "generate_documentation_proto",
  };
  absl::c_copy(copts, std::back_inserter(argv));
  argv.insert(argv.end(),
              {
                  // Force c++ mode.
                  "-x",
                  "c++",
                  // Actual header to compile.
                  header.file_path,
                  // Suppress warnings.
                  "-w",
                  "-std=gnu++20",
                  // This flag makes documentation-like comments be linked
                  // directly to the ast element they are nearest.
                  "-fparse-all-comments",
                  "-fsyntax-only",
                  "-o",
                  "/dev/null",
                  absl::StrFormat("-isystem%s", llvm_include_dir),
              });

  ParsedHeaderComments parsed;
  parsed.repo_path = header.repo_path;
  m::MatchFinder finder;
  Callback cb(class_names, parsed);

  finder.addMatcher(
      m::cxxRecordDecl(m::isDerivedFrom(m::hasName("xls::OptimizationPass")),
                       m::isDefinition(), m::isExpansionInMainFile())
          .bind("base_opt_pass"),
      &cb);
  for (int64_t i = 0; i < static_cast<int64_t>(class_names.size()); ++i) {
    finder.addMatcher(
        m::cxxRecordDecl(m::isDerivedFrom(m::hasName("xls::OptimizationPass")),
                         m::hasName(class_names[i]), m::isDefinition(),
                         m::isExpansionInMainFile())
            .bind(absl::StrFormat("exact_opt_pass_%d", i)),
        &cb);
  }

  std::unique_ptr<clang::FrontendAction> libtool_action =
      clang::tooling::newFrontendActionFactory(&finder)->create();

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));

  llvm::IntrusiveRefCntPtr<clang::FileManager> libtool_files(
      new clang::FileManager(clang::FileSystemOptions(), overlay_fs));

  auto libtool_inv = std::make_unique<clang::tooling::ToolInvocation>(
      argv, std::move(libtool_action), libtool_files.get());

  // Run the actual tool. Errors are suppressed by -w.
  XLS_RET_CHECK(libtool_inv->run()) << "Failed parsing " << header.file_path;
  return parsed;
}

struct ResolvedPassDoc {
  std::string file;
  std::string notes;
};

ResolvedPassDoc ResolvePassDocForInfo(
    const OptimizationPassRegistryBase::RegistrationInfo& info,
    absl::Span<const ParsedHeaderComments* const> candidate_headers) {
  // 1. Exact class match with a non-empty comment.
  for (const ParsedHeaderComments* h : candidate_headers) {
    if (auto it = h->exact_comment.find(info.class_name);
        it != h->exact_comment.end()) {
      return {.file = h->repo_path, .notes = it->second};
    }
  }
  // 2. Exact class match without its own comment: keep the header that defines
  // the class, and fall back to a base-class comment in the same header or in
  // another candidate header.
  for (const ParsedHeaderComments* h : candidate_headers) {
    if (h->matched_exact.contains(info.class_name)) {
      if (h->first_nonempty_base_comment.has_value()) {
        return {.file = h->repo_path, .notes = *h->first_nonempty_base_comment};
      }
      for (const ParsedHeaderComments* h_base : candidate_headers) {
        if (h_base->first_nonempty_base_comment.has_value()) {
          return {.file = h->repo_path,
                  .notes = *h_base->first_nonempty_base_comment};
        }
      }
      LOG(WARNING) << "Unable to find comment for OptimizationPass "
                   << info.short_name << " (" << info.class_name
                   << ") in header file " << h->repo_path;
      return {.file = h->repo_path, .notes = ""};
    }
  }
  // 3. Fallback when no exact class matched (e.g. class_name is UNKNOWN):
  // prefer the first candidate header with a non-empty base-class comment, else
  // any candidate header defining an OptimizationPass.
  for (const ParsedHeaderComments* h : candidate_headers) {
    if (h->first_nonempty_base_comment.has_value()) {
      return {.file = h->repo_path, .notes = *h->first_nonempty_base_comment};
    }
  }
  for (const ParsedHeaderComments* h : candidate_headers) {
    if (h->matched_base) {
      LOG(WARNING) << "Unable to find comment for OptimizationPass "
                   << info.short_name << " (" << info.class_name
                   << ") in header file " << h->repo_path;
      return {.file = h->repo_path, .notes = ""};
    }
  }
  const std::string& fallback_file = candidate_headers.front()->repo_path;
  LOG(WARNING) << "Unable to find comment for OptimizationPass "
               << info.short_name << " (" << info.class_name
               << ") in header file " << fallback_file;
  return {.file = fallback_file, .notes = ""};
}

}  // namespace

absl::StatusOr<PassDocumentationProto> GenerateDocumentationProto(
    const OptimizationPassRegistryBase& registry,
    absl::Span<PassHeaderPath const> headers,
    absl::Span<std::string const> copts) {
  PassDocumentationProto res;
  if (headers.empty()) {
    return res;
  }

  absl::flat_hash_set<std::string_view> target_repo_paths;
  for (const PassHeaderPath& h : headers) {
    target_repo_paths.insert(h.repo_path);
  }

  std::vector<OptimizationPassRegistryBase::RegistrationInfo> infos =
      registry.GetRegisteredInfos();
  VLOG(5) << "Found " << infos.size() << " registered infos.";

  std::vector<OptimizationPassRegistryBase::RegistrationInfo> matching_infos;
  absl::flat_hash_set<std::string> referenced_repo_paths;
  std::vector<std::string> class_names;
  absl::flat_hash_set<std::string> seen_class_names;
  for (const auto& info : infos) {
    bool matches = false;
    for (const std::string& hdr : info.header_files) {
      if (target_repo_paths.contains(hdr)) {
        matches = true;
        referenced_repo_paths.insert(hdr);
      }
    }
    if (matches) {
      matching_infos.push_back(info);
      if (seen_class_names.insert(info.class_name).second) {
        class_names.push_back(info.class_name);
      }
    }
  }

  if (matching_infos.empty()) {
    return res;
  }

  // At the very end of the include search list we need to put the llvm base
  // headers. Normally these would be auto-populated so the cmdline doesn't
  // include them, so we supply them ourselves after all other include flags.
  // TODO(allight): GetXlsRunfilePath doesn't support folders. Just ask for a
  // specific file and get the parent.
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path runfile_path,
      GetXlsRunfilePath(std::string(kLlvmHeaders) + "/stddef.h",
                        /*package=*/"llvm-project"));
  std::filesystem::path llvm_include_dir = runfile_path.parent_path();

  std::vector<ParsedHeaderComments> parsed_headers;
  parsed_headers.reserve(headers.size());
  absl::flat_hash_map<std::string, int64_t> parsed_index_by_repo_path;
  for (const PassHeaderPath& h : headers) {
    if (!referenced_repo_paths.contains(h.repo_path) ||
        parsed_index_by_repo_path.contains(h.repo_path)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        ParsedHeaderComments parsed,
        ParseCppCommentsForHeader(h, class_names, copts, llvm_include_dir));
    parsed_index_by_repo_path.emplace(
        h.repo_path, static_cast<int64_t>(parsed_headers.size()));
    parsed_headers.push_back(std::move(parsed));
  }

  for (const auto& info : matching_infos) {
    absl::flat_hash_set<std::string_view> info_headers(
        info.header_files.begin(), info.header_files.end());
    std::vector<const ParsedHeaderComments*> candidate_headers;
    for (const ParsedHeaderComments& parsed : parsed_headers) {
      if (info_headers.contains(parsed.repo_path)) {
        candidate_headers.push_back(&parsed);
      }
    }
    XLS_RET_CHECK(!candidate_headers.empty());

    XLS_ASSIGN_OR_RETURN(auto gen, registry.Generator(info.short_name));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> real_pass,
                         gen->Generate());
    ResolvedPassDoc resolved = ResolvePassDocForInfo(info, candidate_headers);

    PassDocumentationProto::OnePass* pass = res.mutable_passes()->Add();
    *pass->mutable_short_name() = info.short_name;
    *pass->mutable_long_name() = real_pass->long_name();
    *pass->mutable_file() = std::move(resolved.file);
    *pass->mutable_notes() = std::move(resolved.notes);
  }
  return res;
}

absl::StatusOr<PassDocumentationProto> GenerateDocumentationProto(
    const OptimizationPassRegistryBase& registry, std::string_view header,
    absl::Span<std::string const> copts) {
  if (header.empty()) {
    return GenerateDocumentationProto(
        registry, absl::Span<PassHeaderPath const>(), copts);
  }
  PassHeaderPath single{
      .file_path = std::string(header),
      .repo_path = std::string(header),
  };
  return GenerateDocumentationProto(registry, absl::MakeConstSpan(&single, 1),
                                    copts);
}

}  // namespace xls
