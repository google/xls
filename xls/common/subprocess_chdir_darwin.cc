// Copyright 2026 The XLS Authors
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

#include <spawn.h>

#include "xls/common/subprocess_chdir.h"

namespace xls::internal {

int AddChdirFileAction(posix_spawn_file_actions_t* file_actions,
                       const char* cwd) {
  // Darwin still advertises POSIX.1-2001. macOS 26 introduced the standard
  // spelling and deprecated _np, which has been available since macOS 10.15.
  // SDK availability and the deployment target are separate: a new SDK can
  // build a binary for an older runtime.
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_26_0
  // The minimum deployment target guarantees that the new API exists.
  return posix_spawn_file_actions_addchdir(file_actions, cwd);
#else
// An older SDK lacks the new declaration, even inside a runtime check.
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  // __builtin_available is Clang's C/C++ runtime availability check. For older
  // deployment targets, the SDK annotation makes the new function a weak
  // import; this check prevents calling it on an OS that lacks it. The required
  // '*' covers unlisted platforms; Bazel selects this file for Darwin.
  // https://clang.llvm.org/docs/LanguageExtensions.html#objective-c-available
  if (__builtin_available(macOS 26.0, *)) {
    return posix_spawn_file_actions_addchdir(file_actions, cwd);
  }
#endif
  // Use the macOS 10.15 API with older SDKs or on pre-26 runtimes.
  return posix_spawn_file_actions_addchdir_np(file_actions, cwd);
#endif
}

}  // namespace xls::internal
