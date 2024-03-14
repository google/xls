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

#include <algorithm>
#include <memory>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <string_view>
#include <system_error>  // NOLINT (filesystem error reporting)
#include <thread>  // NOLINT for std::thread::hardware_concurrency()
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/subprocess.h"
#include "xls/common/thread.h"
#include "re2/re2.h"

namespace {

namespace fs = std::filesystem;
using hash_t = uint64_t;
using filepath_contenthash_t = std::pair<fs::path, hash_t>;

// Can't be absl::Hash as we want it stable between invocations.
hash_t hashContent(const std::string& s) { return std::hash<std::string>()(s); }
std::string ToHex(uint64_t value, int show_lower_nibbles = 16) {
  const std::string hex16 = absl::StrCat(absl::Hex(value, absl::kZeroPad16));
  return hex16.substr(16 - show_lower_nibbles);
}

std::optional<std::string> ReadAndVerifyTidyConfig(const fs::path& config) {
  const auto content = xls::GetFileContents(config);
  if (!content.ok()) {
    return std::nullopt;
  }
  const auto start_config = content->find("\nChecks:");
  if (start_config == std::string::npos) {
    std::cerr << "Not seen 'Checks:' in config " << config << "\n";
    return std::nullopt;
  }
  if (content->find('#', start_config) != std::string::npos) {
    std::cerr << "Comment found in check section of " << config << "\n";
    return std::nullopt;
  }
  return content->substr(start_config);
}

fs::path GetCacheDir() {
  if (const char* from_env = getenv("CACHE_DIR")) {
    return fs::path(from_env);
  }
  if (const char* home = getenv("HOME")) {
    if (auto cdir = fs::path(home) / ".cache/"; fs::exists(cdir)) {
      return cdir;
    }
  }
  return fs::path(getenv("TMPDIR") ?: "/tmp");
}

// Fix filename paths that are not emitted relative to project root.
std::string CanonicalizeSourcePaths(const std::string& content) {
  static const RE2 sFixPathsRe = []() {
    std::string canonicalize_expr = "(^|\\n)(";  // fix names at start of line
    auto root =
        xls::InvokeSubprocess({"/bin/sh", "-c", "bazel info execution_root"});
    if (root.ok() && !root->stdout.empty()) {
      root->stdout.pop_back();  // remove newline.
      canonicalize_expr += root->stdout + "/|";
    }
    if (const auto cwd = xls::GetCurrentDirectory(); cwd.ok()) {
      canonicalize_expr += cwd->string() + "/";
    }
    canonicalize_expr += ")?(\\./)?";  // Some start with, or have a trailing ./
    return RE2{canonicalize_expr};
  }();
  std::string result = content;
  RE2::GlobalReplace(&result, sFixPathsRe, "\\1");
  return result;
}

// Given a work-queue in/out-file, process it. Using system() for portability.
void ClangTidyProcessFiles(const fs::path& content_dir,
                           const std::vector<std::string>& base_cmd,
                           std::list<filepath_contenthash_t>* work_queue) {
  if (!work_queue || work_queue->empty()) {
    return;
  }
  const int kJobs = std::thread::hardware_concurrency();
  std::cerr << work_queue->size() << " files to process...";

  absl::Mutex queue_access_lock;
  auto clang_tidy_runner = [&]() {
    std::vector<std::string> work_command(base_cmd);
    std::string* cmd_in_file = &work_command.emplace_back();
    for (;;) {
      filepath_contenthash_t work;
      {
        absl::MutexLock lock(&queue_access_lock);
        if (work_queue->empty()) {
          return;
        }
        fprintf(stderr, "%5d\b\b\b\b\b", static_cast<int>(work_queue->size()));
        work = work_queue->front();
        work_queue->pop_front();
      }
      *cmd_in_file = work.first.string();
      auto run_result = xls::InvokeSubprocess(work_command);
      if (!run_result.ok()) {
        std::cerr << "clang-tidy invocation " << run_result.status() << "\n";
        continue;
      }
      const std::string output = CanonicalizeSourcePaths(run_result->stdout);
      if (auto set_content_status =
              xls::SetFileContents(content_dir / ToHex(work.second), output);
          !set_content_status.ok()) {
        std::cerr << "Failed to set output " << set_content_status << "\n";
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

}  // namespace

int main(int argc, char* argv[]) {
  const std::string kProjectPrefix = "xls_";
  const std::string kSearchDir = "xls";
  const std::string kFileExcludeRe = "xlscc/(examples|synth_only)";

  const std::string kTidySymlink = kProjectPrefix + "clang-tidy.out";
  const fs::path cache_dir = GetCacheDir() / "clang-tidy";

  if (!fs::exists("compile_commands.json")) {
    std::cerr << "No compilation db found. First, run make-compilation-db.sh\n";
    return EXIT_FAILURE;
  }
  const auto config = ReadAndVerifyTidyConfig(".clang-tidy");
  if (!config) {
    return EXIT_FAILURE;
  }

  // We'll invoke clang-tidy with all the additional flags user provides.
  const std::string clang_tidy_binary_name =
      getenv("CLANG_TIDY") ?: "clang-tidy";
  // InvokeSubprocess() needs a fully qualified binary path later. Let's
  // resolve, and use the opportunity to see if clang-tidy even exists.
  auto find_clang_tidy = xls::InvokeSubprocess(
      {"/bin/sh", "-c", std::string("command -v ") + clang_tidy_binary_name});
  if (!find_clang_tidy.ok()) {
    std::cerr << find_clang_tidy.status() << "\n";
    return EXIT_FAILURE;
  }
  if (find_clang_tidy->exit_status != 0 || find_clang_tidy->stdout.empty()) {
    std::cerr << "Can't find " << clang_tidy_binary_name << "\n";
    return EXIT_FAILURE;
  }
  find_clang_tidy->stdout.pop_back();  // command -v adds a newline.
  const std::string clang_tidy = find_clang_tidy->stdout;

  std::vector<std::string> clang_tidy_invocation = {clang_tidy, "--quiet",
                                                    "--config", *config};
  for (int i = 1; i < argc; ++i) {
    clang_tidy_invocation.push_back(argv[i]);
  }

  // Use major version as part of name of our configuration specific dir.
  std::string version;
  if (auto v = xls::InvokeSubprocess({clang_tidy, "--version"}); v.ok()) {
    version = v->stdout;
  } else {
    std::cerr << v.status() << "\n";
    return EXIT_FAILURE;
  }
  std::string major_version;
  if (!RE2::PartialMatch(version, "version ([0-9]+)", &major_version)) {
    major_version = "UNKNOWN";
  }

  // Cache directory name based on configuration.
  const fs::path project_base_dir =
      cache_dir /
      fs::path(kProjectPrefix + "v" + major_version + "_" +
               ToHex(hashContent(version +
                                 absl::StrJoin(clang_tidy_invocation, " ")),
                     8));
  const fs::path tidy_outfile = project_base_dir / "tidy.out";
  const fs::path content_dir = project_base_dir / "contents";
  fs::create_directories(content_dir);
  std::cerr << "Cache dir " << project_base_dir << "\n";

  // Gather all *.cc and *.h files; remember content hashes of includes.
  std::vector<filepath_contenthash_t> files_of_interest;
  std::map<std::string, hash_t> header_hashes;
  const RE2 exclude_re(kFileExcludeRe);
  for (const auto& dir_entry : fs::recursive_directory_iterator(kSearchDir)) {
    const fs::path& p = dir_entry.path().lexically_normal();
    if (!fs::is_regular_file(p)) {
      continue;
    }
    if (!kFileExcludeRe.empty() && RE2::PartialMatch(p.string(), exclude_re)) {
      continue;
    }
    if (auto ext = p.extension(); ext == ".cc" || ext == ".h") {
      const auto contents = xls::GetFileContents(p);
      if (contents.ok()) {
        files_of_interest.emplace_back(p, 0);
        if (ext == ".h") {
          header_hashes[p.string()] = hashContent(*contents);
        }
      }
    }
  }
  std::cerr << files_of_interest.size() << " files of interest.\n";

  // Create content hash address. If any header a file depends on changes, we
  // want to reprocess. So we make the hash dependent on header content as well.
  std::list<filepath_contenthash_t> work_queue;
  const RE2 inc_re("\"([0-9a-zA-Z_/-]+\\.h)\"");  // match include file
  for (filepath_contenthash_t& f : files_of_interest) {
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
    const fs::path content_hash_file = content_dir / ToHex(f.second);
    if (!exists(content_hash_file)) {
      work_queue.emplace_back(f);
    }
  }

  // Run clang tidy in parallel on the files to process.
  ClangTidyProcessFiles(content_dir, clang_tidy_invocation, &work_queue);

  // Assemble the separate outputs into a single file. Tally up per-check stats.
  const RE2 check_re("(\\[[a-zA-Z.-][a-zA-Z.0-9-]+\\])\n");
  std::map<std::string, int> checks_seen;
  std::ofstream tidy_collect(tidy_outfile);
  for (const filepath_contenthash_t& f : files_of_interest) {
    const auto tidy = xls::GetFileContents(content_dir / ToHex(f.second));
    if (!tidy.ok()) {
      continue;
    }
    if (!tidy->empty()) {
      tidy_collect << f.first.string() << ":\n" << *tidy;
    }
    std::string_view re2_consumable(*tidy);
    std::string check_name;
    while (RE2::FindAndConsume(&re2_consumable, check_re, &check_name)) {
      checks_seen[check_name]++;
    }
  }
  std::error_code ignored_error;
  fs::remove(kTidySymlink, ignored_error);
  fs::create_symlink(tidy_outfile, kTidySymlink, ignored_error);

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
