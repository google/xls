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

// glibc has provided this extension since 2.29. The build enables _GNU_SOURCE
// to expose its declaration, without requiring full POSIX.1-2024 conformance.
int AddChdirFileAction(posix_spawn_file_actions_t* file_actions,
                       const char* cwd) {
  return posix_spawn_file_actions_addchdir_np(file_actions, cwd);
}

}  // namespace xls::internal
