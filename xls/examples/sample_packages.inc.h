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
#ifndef XLS_EXAMPLES_SAMPLE_PACKAGES_INC_H_
#define XLS_EXAMPLES_SAMPLE_PACKAGES_INC_H_

#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/status_macros.h"

namespace xls {

inline absl::StatusOr<std::vector<std::string>> GetExamplePaths() {
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path example_file_list_path,
      GetXlsRunfilePath("xls/examples/ir_example_file_list.txt"));
  XLS_ASSIGN_OR_RETURN(std::string example_paths_string,
                       GetFileContents(example_file_list_path));
  return absl::StrSplit(example_paths_string, '\n');
}

}  // namespace xls

#endif  // XLS_EXAMPLES_SAMPLE_PACKAGES_INC_H_
