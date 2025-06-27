// Copyright 2020 The XLS Authors
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

#include "xls/common/file/get_runfile_path.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/const_init.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/status_macros.h"
#include "tools/cpp/runfiles/runfiles.h"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <sys/syslimits.h>
#endif /* __APPLE__ */

namespace xls {
namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;

static absl::Mutex mutex(absl::kConstInit);
static Runfiles* runfiles;

absl::StatusOr<std::filesystem::path> GetSelfExecutablePath() {
#if __linux__
  return GetRealPath("/proc/self/exe");
#elif __APPLE__
  char path[PATH_MAX + 1];
  uint32_t size = PATH_MAX;
  if (_NSGetExecutablePath(path, &size) == 0) {
    return std::filesystem::path(path);
  }
  return absl::InvalidArgumentError("Self path could not fit into buffer");
#else
#error "Unknown platform"
#endif
}

absl::StatusOr<Runfiles*> GetRunfiles(
    std::optional<std::string_view> argv0 = std::nullopt) {
  absl::MutexLock lock(&mutex);
  if (runfiles == nullptr) {
    // Need to dereference the path, in case it's a link (as with the default).
    std::filesystem::path path;
    if (argv0.has_value()) {
      XLS_ASSIGN_OR_RETURN(path, GetRealPath(std::string(argv0.value())));
    } else {
      XLS_ASSIGN_OR_RETURN(path, GetSelfExecutablePath());
    }

    std::string error;
    runfiles = Runfiles::Create(path.string(), &error);
    if (runfiles == nullptr) {
      return absl::UnknownError(
          absl::StrCat("Failed to initialize Runfiles: ", error));
    }
  }

  return runfiles;
}

}  // namespace

absl::StatusOr<std::filesystem::path> GetXlsRunfilePath(
    const std::filesystem::path& path) {
  XLS_ASSIGN_OR_RETURN(Runfiles * runfiles, GetRunfiles());
  return runfiles->Rlocation("com_google_xls" / path);
}

absl::Status InitRunfilesDir(const std::string& argv0) {
  return GetRunfiles(argv0).status();
}

}  // namespace xls
