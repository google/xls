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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"

namespace xls {

// Translates a corpus of VAST `VerilogFile` objects into corresponding DSLX
// files in `out_dir`. Currently this only works for a subset of VAST focused on
// typedefs and constants, and is intended for porting such entities from
// Verilog to a DSLX code base.
//
// The generated DSLX is guaranteed to parse and type-check successfully, if
// this function does not error.
//
// Input:
//   out_dir: The path in which to generate the DSLX files, overwriting existing
//      files by the same name where necessary.
//   dslx_stdlib_path: The path to the DSLX standard library, in case it needs
//      to be used, e.g., for translating `clog2` to `std::clog2`.
//   verilog_paths_in_order: The paths in the Verilog corpus being translated,
//      in the order the caller would like the corresponding DSLX to be output.
//      It is assumed that the last one is "main" file and the others are its
//      dependencies.
//   warnings: Optional vector into which any generated warning messages will
//       be placed.
absl::Status TranslateVastToDslx(
    std::filesystem::path out_dir, std::string_view dslx_stdlib_path,
    const std::vector<std::filesystem::path>& verilog_paths_in_order,
    const absl::flat_hash_map<std::filesystem::path,
                              std::unique_ptr<verilog::VerilogFile>>&
        verilog_files,
    std::vector<std::string>* warnings = nullptr);

// Variant of `TranslateVastToDslx` that generates one combined DSLX module from
// the whole Verilog corpus, applying pseudo-namespacing to distinguish names in
// non-main modules. Returns the DSLX source code for the combined module.
absl::StatusOr<std::string> TranslateVastToCombinedDslx(
    std::string_view dslx_stdlib_path,
    const std::vector<std::filesystem::path>& verilog_paths_in_order,
    const absl::flat_hash_map<std::filesystem::path,
                              std::unique_ptr<verilog::VerilogFile>>&
        verilog_files,
    std::vector<std::string>* warnings = nullptr);

}  // namespace xls

#endif  // XLS_CODEGEN_VAST_TRANSLATE_VAST_TO_DSLX_H_
