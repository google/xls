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

#ifndef XLS_COMMON_SUBPROCESS_CHDIR_H_
#define XLS_COMMON_SUBPROCESS_CHDIR_H_

#include <spawn.h>

namespace xls::internal {

// Records a child working-directory action. Returns zero on success or a POSIX
// error number directly; errno need not match. Bazel selects the implementation
// for the configured libc API and deployment target.
int AddChdirFileAction(posix_spawn_file_actions_t* file_actions,
                       const char* cwd);

}  // namespace xls::internal

#endif  // XLS_COMMON_SUBPROCESS_CHDIR_H_
