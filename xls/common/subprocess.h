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

#ifndef XLS_COMMON_SUBPROCESS_H_
#define XLS_COMMON_SUBPROCESS_H_

#include <filesystem>  // NOLINT
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/span.h"

namespace xls {

struct SubprocessResult {
  std::string stdout_content;
  std::string stderr_content;
  int exit_status;

  // The results of WIFEXITED for the subprocess.
  // https://pubs.opengroup.org/onlinepubs/9699919799/functions/wait.html
  bool normal_termination;

  // Whether the timeout duration expired and we attempted to kill the
  // subprocess on account of the timeout as a result.
  //
  // Note that the process *may or may not* have exited on account of us
  // attempting to kill it after the timeout, e.g. there can be a race where it
  // exits normally right as we're trying to send it the signal; however, this
  // is an easy boolean to check to see if it clearly timed out and was likely
  // killed on account of that fact.
  bool timeout_expired;
};
std::ostream &operator<<(std::ostream &os, const SubprocessResult &other);
inline void PrintTo(const SubprocessResult& result, std::ostream* os) {
  *os << absl::StreamFormat(
      "SubprocessResult "
      "{\n\tstdout=%s\n\tstderr=%s\n\texit_status=%d\n\tnormal_termination=%"
      "d\n\ttimeout_expired=%d\n}}",
      result.stdout_content, result.stderr_content, result.exit_status,
      result.normal_termination, result.timeout_expired);
}

// Returns Status::kInternalError if the subprocess unexpectedly terminated or
// terminated with return code != 0 rather than StatusOk(). This is helpful when
// invoking a subprocess in the circumstance where a zero return code and
// clean termination is the only case of interest.
//
// Ex:
//   StatusOr<SubprocessResult> result_or_status
//     = SubprocessErrorAsStatus(InvokeSubprocess())
// ...
// At this moment there is only a result if the subprocess terminated normally
// with exit code of zero. Failure to spawn the process or unexpected
// termination or exit code of non-zero will produce a kInternalError status.
absl::StatusOr<SubprocessResult> SubprocessErrorAsStatus(
    absl::StatusOr<SubprocessResult>);

// Discard everything from the result except what came back on stdout and
// stderr. Preserve all status. This is helpful when caring solely about what
// came back from the subprocess.
//
// Ex:
//   auto streams_or_status = SubprocessResultToStrings(InvokeSubprocess(...));
//   if (streams_or_status.ok()) {
//     VLOG() << "stdout:" << streams_or_status->first;
//     VLOG() << "stderr:" << streams_or_status->second;
//   }
absl::StatusOr<std::pair<std::string, std::string>> SubprocessResultToStrings(
    absl::StatusOr<SubprocessResult> result);

// Invokes a subprocess with the given argv. If 'cwd' is supplied, the
// subprocess will be invoked in the given directory. Problems in invocation
// result in a non-OK Status code. Unexpected termination of the subprocess
// also (currently) results in a non-OK status code.
//
// Subprocesses that run beyond 'optional_timeout' will be stopped. Nullopt is
// equivalent to absl::ZeroDuration and means "wait forever".
absl::StatusOr<SubprocessResult> InvokeSubprocess(
    absl::Span<const std::string> argv,
    std::optional<std::filesystem::path> cwd = std::nullopt,
    std::optional<absl::Duration> optional_timeout = std::nullopt);

}  // namespace xls
#endif  // XLS_COMMON_SUBPROCESS_H_
