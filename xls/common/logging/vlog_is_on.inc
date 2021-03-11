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

#include <cstdint>

#include "absl/flags/flag.h"
#include "absl/strings/numbers.h"

ABSL_FLAG(int32_t, v, 0,
          "Show all XLS_VLOG(m) messages for m <= this. Overridable by "
          "--vmodule.");

ABSL_FLAG(
    std::string, vmodule, "",
    "per-module verbose level."
    " Argument is a comma-separated list of <module name>=<log level>."
    " <module name> is a glob pattern, matched against the filename base"
    " (that is, name ignoring .cc/.h./-inl.h)."
    " A pattern without slashes matches just the file name portion, otherwise"
    " the whole file path (still without .cc/.h./-inl.h) is "
    "matched."
    " ? and * in the glob pattern match any single or sequence of characters"
    " respectively including slashes."
    " <log level> overrides any value given by --v.")
    .OnUpdate([] {
      for (absl::string_view glob_level :
           absl::StrSplit(absl::GetFlag(FLAGS_vmodule), ',')) {
        const size_t eq = glob_level.rfind('=');
        if (eq == glob_level.npos) continue;
        const absl::string_view glob = glob_level.substr(0, eq);
        int level;
        if (!absl::SimpleAtoi(glob_level.substr(eq + 1), &level)) continue;
        xls::SetVLOGLevel(glob, level);
      }
    });

namespace xls {
namespace logging_internal {

// TODO(leary): 2020-05-03 ABSL_CONST_INIT doesn't work as an annotation on
// this in OSS clang build. Need to determine why. Notably API is in
// transition.
// Protects accesses to the "vmodule_list" variable in vlog_is_on.cc.
// Reads may be done without the lock if Acquire_Load() is used.
// Writes must use Release_Store().
static absl::base_internal::SpinLock vmodule_lock(
    absl::kConstInit, absl::base_internal::SCHEDULE_COOPERATIVE_AND_KERNEL);

}  // namespace logging_internal
}  // namespace xls
