// Copyright 2023 The XLS Authors.
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

// Program to run clang-tidy on files in a bazel project while caching the
// results as clang-tidy can be pretty slow. The clang-tidy output messages
// are content-addressed in a hash(cc-file-content) cache file.
// Should work on any Posix-compatible system.
//
// Invocation without parameters simply uses the .clang-tidy config to run on
// all *.{cc,h} files. Additional parameters passed to this script are passed
// to clang-tidy as-is. Typical use could be for instance
//   run_clang_tidy_cached --checks="-*,modernize-use-override" --fix
//
// Note: useful environment variables to configure are
//  CLANG_TIDY = binary to run; default would just be clang-tidy.
//  CACHE_DIR  = where to put the cached content; default ~/.cache

// Based on standalone c++-17 scripts found in
//  https://github.com/chipsalliance/verible
//  https://github.com/hzeller/bant
//
// ... but using local absl{thread,strings}/xls{file, process}/RE2 features.
// (so, it is not standalone anymore.)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT (filesystem error reporting)
#include <thread>        // NOLINT for std::thread::hardware_concurrency()
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/subprocess.h"
#include "xls/common/thread.h"
#include "re2/re2.h"

// Some configuration for this project.
static constexpr std::string_view kProjectCachePrefix = "xls_";
static constexpr std::string_view kWorkspaceFile = "WORKSPACE";

// Choices of what files to include and exclude to run clang-tidy on.
static constexpr std::string_view kStartDirectory = "xls";
static constexpr std::string_view kFileIncludeRe = ".*";
static constexpr std::string_view kFileExcludeRe =
    ".git/|.github/|dev_tools/run_clang_tidy_cached.cc/|"
    "xls/common/build_embed\\.cc|"
    "xlscc/(examples|synth_only|build_rules)";
inline bool ConsiderExtension(const std::string& extension) {
  return extension == ".cc" || extension == ".h";
}

// Configuration of clang-tidy itself.
static constexpr std::string_view kClangConfigFile = ".clang-tidy";
static constexpr std::string_view kExtraArgs[] = {"-Wno-unknown-pragmas",
                                                  "-std=c++20"};

// If the compilation DB changed, it might be worthwhile revisiting
// sources that previously had issues. This flag enables that.
// It is good to set if the project is 'clean' and there are only a
// few problematic sources to begin with, otherwise every update of the
// compilation DB will re-trigger revisiting all of them.
// Our baseline is not clean yet, so 'false' for now.
static constexpr bool kRevisitBrokenFilesIfCompilationDBNewer = false;

namespace {

namespace fs = std::filesystem;
using file_time = std::filesystem::file_time_type;
using hash_t = uint64_t;
using filepath_contenthash_t = std::pair<fs::path, hash_t>;

std::optional<std::string> GetCommandOutput(const std::string& prog) {
  auto result = xls::InvokeSubprocess({"/bin/sh", "-c", prog});
  if (result.ok()) {
    return result->stdout_content;
  }
  std::cerr << result.status() << "\n";
  return std::nullopt;
}

// Can't be absl::Hash as we want it stable between invocations.
hash_t hashContent(const std::string& s) { return std::hash<std::string>()(s); }
std::string ToHex(uint64_t value, int show_lower_nibbles = 16) {
  const std::string hex16 = absl::StrCat(absl::Hex(value, absl::kZeroPad16));
  return hex16.substr(16 - show_lower_nibbles);
}

// Mapping filepath_contenthash_t to an actual location in the file system.
class ContentAddressedStore {
 public:
  explicit ContentAddressedStore(const fs::path& project_base_dir)
      : content_dir(project_base_dir / "contents") {
    fs::create_directories(content_dir);
  }

  // Given filepath contenthash, return the path to read/write from.
  fs::path PathFor(const filepath_contenthash_t& c) const {
    // Name is human readable, the content hash makes it unique.
    std::string name_with_contenthash =
        absl::StrCat(c.first.filename().string(), "-", ToHex(c.second));
    return content_dir / name_with_contenthash;
  }

  // Check if this needs to be recreated, either because it is not there,
  // or is not empty and does not fit freshness requirements.
  bool NeedsRefresh(const filepath_contenthash_t& c,
                    file_time min_freshness) const {
    const fs::path content_hash_file = PathFor(c);
    if (!fs::exists(content_hash_file)) {
      return true;
    }

    // If file exists but is broken (i.e. has a non-zero size with messages),
    // consider recreating if if older than compilation db.
    const bool timestamp_trigger =
        kRevisitBrokenFilesIfCompilationDBNewer &&
        (fs::file_size(content_hash_file) > 0 &&
         fs::last_write_time(content_hash_file) < min_freshness);
    return timestamp_trigger;
  }

 private:
  const fs::path content_dir;
};

class ClangTidyRunner {
 public:
  ClangTidyRunner(int argc, char** argv)
      : clang_tidy_(getenv("CLANG_TIDY") ?: "clang-tidy"),
        clang_tidy_args_(AssembleArgs(argc, argv)) {
    project_cache_dir_ = AssembleProjectCacheDir();
  }

  const fs::path& project_cache_dir() const { return project_cache_dir_; }

  // Given a work-queue in/out-file, process it. Using system() for portability.
  void RunClangTidyOn(ContentAddressedStore& output_store,
                      std::list<filepath_contenthash_t> work_queue) {
    if (work_queue.empty()) {
      return;
    }
    const int kJobs = std::thread::hardware_concurrency();
    std::cerr << work_queue.size() << " files to process on ";

    absl::Mutex queue_access_lock;
    auto clang_tidy_runner = [&]() {
      // We use all the same arguments for all invocations and filename at end.
      std::vector<std::string> work_command;
      work_command.push_back(clang_tidy_);
      work_command.push_back("<the file>");
      work_command.insert(work_command.end(), clang_tidy_args_.begin(),
                          clang_tidy_args_.end());
      for (;;) {
        filepath_contenthash_t work;
        {
          absl::MutexLock lock(&queue_access_lock);
          if (work_queue.empty()) {
            return;
          }
          fprintf(stderr, "%5d\b\b\b\b\b", static_cast<int>(work_queue.size()));
          work = work_queue.front();
          work_queue.pop_front();
        }
        work_command[1] = work.first.string();
        auto run_result = xls::InvokeSubprocess(work_command);
        if (!run_result.ok()) {
          std::cerr << "clang-tidy invocation " << run_result.status() << "\n";
          continue;
        }
        std::string output =
            RepairFilenameOccurences(run_result->stdout_content);
        if (auto s = xls::SetFileContents(output_store.PathFor(work), output);
            !s.ok()) {
          std::cerr << "Failed to set output " << s << "\n";
        }
      }
    };

    std::vector<std::unique_ptr<xls::Thread>> workers;
    workers.reserve(kJobs);
    for (auto i = 0; i < kJobs; ++i) {
      workers.push_back(std::make_unique<xls::Thread>(clang_tidy_runner));
    }
    for (auto& t : workers) {
      t->Join();
    }
    fprintf(stderr, "     \n");  // Clean out progress counter.
  }

 private:
  static fs::path GetCacheBaseDir() {
    if (const char* from_env = getenv("CACHE_DIR")) {
      return fs::path{from_env};
    }
    if (const char* home = getenv("HOME")) {
      if (auto cdir = fs::path(home) / ".cache/"; fs::exists(cdir)) {
        return cdir;
      }
    }
    return fs::path{getenv("TMPDIR") ?: "/tmp"};
  }

  static std::vector<std::string> AssembleArgs(int argc, char** argv) {
    std::vector<std::string> result;
    result.push_back("--quiet");
    result.push_back(absl::StrCat("--config-file=", kClangConfigFile));
    for (const std::string_view arg : kExtraArgs) {
      result.push_back(absl::StrCat("--extra-arg=", arg));
    }
    for (int i = 1; i < argc; ++i) {
      result.push_back(argv[i]);
    }
    return result;
  }

  fs::path AssembleProjectCacheDir() const {
    const fs::path cache_dir = GetCacheBaseDir() / "clang-tidy";

    // Use major version as part of name of our configuration specific dir.
    auto version = GetCommandOutput(clang_tidy_ + " --version");
    std::string major_version;
    if (!RE2::PartialMatch(*version, "version ([0-9]+)", &major_version)) {
      major_version = "UNKNOWN";
    }

    // Make sure directory filename depends on .clang-tidy content.
    hash_t cache_unique_id =
        hashContent(*version + absl::StrJoin(clang_tidy_args_, " "));
    cache_unique_id ^= hashContent(*xls::GetFileContents(kClangConfigFile));
    return cache_dir /
           fs::path(absl::StrCat(kProjectCachePrefix, "v", major_version, "_",
                                 ToHex(cache_unique_id, 8)));
  }

  // Fix filename paths found in logfiles that are not emitted relative to
  // project root in the log (bazel has its own)
  static std::string RepairFilenameOccurences(std::string_view in_content) {
    static const RE2 sFixPathsRe = []() {
      std::string canonicalize_expr = "(^|\\n)(";  // fix names at start of line
      auto root_or = GetCommandOutput("bazel info execution_root 2>/dev/null");
      CHECK(root_or.has_value()) << "No bazel to execute ?";
      std::string root = root_or.value();
      if (!root.empty()) {
        root.pop_back();  // remove newline.
        canonicalize_expr += root + "/|";
      }
      canonicalize_expr += fs::current_path().string() + "/";  // $(pwd)/
      canonicalize_expr +=
          ")?(\\./)?";  // Some start with, or have a trailing ./
      return RE2{canonicalize_expr};
    }();

    std::string result{in_content};
    RE2::GlobalReplace(&result, sFixPathsRe, "\\1");
    return result;
  }

  const std::string clang_tidy_;
  const std::vector<std::string> clang_tidy_args_;
  fs::path project_cache_dir_;
};

class FileGatherer {
 public:
  FileGatherer(ContentAddressedStore& store, std::string_view search_dir)
      : store_(store), root_dir_(search_dir) {}

  // Find all the files we're interested in, and assemble a list of
  // paths that need refreshing.
  std::list<filepath_contenthash_t> BuildWorkList(file_time min_freshness) {
    // Gather all *.cc and *.h files; remember content hashes of includes.
    static const RE2 include_re(kFileIncludeRe);
    static const RE2 exclude_re(kFileExcludeRe);
    std::map<std::string, hash_t> header_hashes;
    for (const auto& dir_entry : fs::recursive_directory_iterator(root_dir_)) {
      const fs::path& p = dir_entry.path().lexically_normal();
      if (!fs::is_regular_file(p)) {
        continue;
      }
      const std::string file = p.string();
      if (!kFileIncludeRe.empty() && !RE2::PartialMatch(file, include_re)) {
        continue;
      }
      if (!kFileExcludeRe.empty() && RE2::PartialMatch(file, exclude_re)) {
        continue;
      }
      const auto extension = p.extension();
      if (ConsiderExtension(extension)) {
        files_of_interest_.emplace_back(p, 0);  // <- hash to be filled later.
      }
      // Remember content hash of header, so that we can make changed headers
      // influence the hash of a file including this.
      if (extension == ".h") {
        auto header_content = xls::GetFileContents(p);
        if (header_content.ok()) {
          header_hashes[file] = hashContent(*header_content);
        }
      }
    }
    std::cerr << files_of_interest_.size() << " files of interest.\n";

    // Create content hash address. If any header a file depends on changes, we
    // want to reprocess. So we make the hash dependent on header content as
    // well.
    std::list<filepath_contenthash_t> work_queue;
    const RE2 inc_re("\"([0-9a-zA-Z_/-]+\\.h)\"");  // match include
    for (filepath_contenthash_t& f : files_of_interest_) {
      const auto content = xls::GetFileContents(f.first);
      if (!content.ok()) {
        continue;
      }
      f.second = hashContent(*content);
      std::string_view re2_consumable(*content);
      std::string header_path;
      while (RE2::FindAndConsume(&re2_consumable, inc_re, &header_path)) {
        f.second ^= header_hashes[header_path];
      }

      // Recreate if we don't have it yet or if it contains messages but is
      // older than WORKSPACE or compilation db. Maybe something got fixed.
      if (store_.NeedsRefresh(f, min_freshness)) {
        work_queue.emplace_back(f);
      }
    }
    return work_queue;
  }

  // Tally up findings for files of interest and assemble in one file.
  // (BuildWorkList() needs to be called first).
  std::map<std::string, int> CreateReport(const fs::path& project_dir,
                                          std::string_view symlink_to) {
    const fs::path tidy_outfile = project_dir / "tidy.out";
    // Assemble the separate outputs into a single file. Tally up per-check
    const RE2 check_re("(\\[[a-zA-Z.-]+\\])\n");
    std::map<std::string, int> checks_seen;
    std::ofstream tidy_collect(tidy_outfile);
    for (const filepath_contenthash_t& f : files_of_interest_) {
      const auto tidy = xls::GetFileContents(store_.PathFor(f));
      if (!tidy.ok()) {
        continue;
      }
      if (!tidy->empty()) {
        tidy_collect << f.first.string() << ":\n" << tidy;
      }
      std::string_view re2_consumable(*tidy);
      std::string check_name;
      while (RE2::FindAndConsume(&re2_consumable, check_re, &check_name)) {
        checks_seen[check_name]++;
      }
    }

    std::error_code ignored_error;
    fs::remove(symlink_to, ignored_error);
    fs::create_symlink(tidy_outfile, symlink_to, ignored_error);
    return checks_seen;
  }

 private:
  ContentAddressedStore& store_;
  const std::string root_dir_;
  std::vector<filepath_contenthash_t> files_of_interest_;
};

}  // namespace

int main(int argc, char* argv[]) {
  // Test that key files exist and remember their last change.
  std::error_code ec;
  const auto workspace_ts = fs::last_write_time(kWorkspaceFile, ec);
  if (ec.value() != 0) {
    std::cerr << "Script needs to be executed in toplevel bazel project dir\n";
    return EXIT_FAILURE;
  }
  const auto compdb_ts = fs::last_write_time("compile_commands.json", ec);
  if (ec.value() != 0) {
    std::cerr << "No compilation db found. First, run make-compilation-db.sh\n";
    return EXIT_FAILURE;
  }
  const auto build_env_latest_change = std::max(workspace_ts, compdb_ts);

  ClangTidyRunner runner(argc, argv);
  ContentAddressedStore store(runner.project_cache_dir());
  std::cerr << "Cache dir " << runner.project_cache_dir() << "\n";

  FileGatherer cc_file_gatherer(store, kStartDirectory);
  auto work_list = cc_file_gatherer.BuildWorkList(build_env_latest_change);

  // Now the expensive part...
  runner.RunClangTidyOn(store, work_list);

  const std::string kTidySymlink =
      absl::StrCat(kProjectCachePrefix, "clang-tidy.out");
  auto checks_seen =
      cc_file_gatherer.CreateReport(runner.project_cache_dir(), kTidySymlink);

  if (checks_seen.empty()) {
    std::cerr << "No clang-tidy complaints. 😎\n";
  } else {
    std::cerr << "--- Summary --- (details in " << kTidySymlink << ")\n";
    using check_count_t = std::pair<std::string, int>;
    std::vector<check_count_t> by_count(checks_seen.begin(), checks_seen.end());
    std::stable_sort(by_count.begin(), by_count.end(),
                     [](const check_count_t& a, const check_count_t& b) {
                       return b.second < a.second;  // reverse count
                     });
    for (const auto& counts : by_count) {
      std::cout << absl::StrFormat("%5d %s\n", counts.second, counts.first);
    }
  }
  return checks_seen.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
}
