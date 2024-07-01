// Copyright 2024 The XLS Authors
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

#ifndef XLS_CODEGEN_VAST_TRANSLATE_VAST_TO_DSLX_H_
#define XLS_CODEGEN_VAST_TRANSLATE_VAST_TO_DSLX_H_

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"

namespace xls {

// Translates a corpus of VAST `VerilogFile` objects into DSLX code. Currently
// this only works for a subset of VAST focused on typedefs and constants, and
// is intended for porting such entities from Verilog to a DSLX code base.
//
// Input:
//   module_name: The name to give to the generated DSLX Module.
//   dslx_stdlib_path: The path to the DSLX standard library, in case it needs
//      to be used, e.g., for translating `clog2` to `std::clog2`.
//   verilog_paths_in_order: The paths in the Verilog corpus being translated,
//      in the order the caller would like the corresponding DSLX to be output.
//      It is assumed that the last one is "main" file and the others are its
//      dependencies. This function may add pseudo-namespacing to the DSLX for
//      non-main files.
//   warnings: Optional vector into which any generated warning messages will
//       be placed.
//
// Return value:
//   The DSLX code, which is guaranteed to parse and type-check successfully, if
//   this function does not error.
absl::StatusOr<std::string> TranslateVastToDslx(
    std::string_view module_name, const std::filesystem::path& dslx_stdlib_path,
    const std::vector<std::filesystem::path>& verilog_paths_in_order,
    const absl::flat_hash_map<std::filesystem::path,
                              std::unique_ptr<verilog::VerilogFile>>&
        verilog_files,
    std::vector<std::string>* warnings = nullptr);

}  // namespace xls

#endif  // XLS_CODEGEN_VAST_TRANSLATE_VAST_TO_DSLX_H_
