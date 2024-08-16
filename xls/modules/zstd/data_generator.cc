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
//

#include "xls/modules/zstd/data_generator.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"

namespace xls::zstd {

static std::string CreateNameForGeneratedFile(
    absl::Span<const std::string> args, std::string_view ext,
    std::optional<std::string_view> prefix) {
  std::string output;

  if (prefix.has_value()) {
    absl::StrAppend(&output, prefix.value(), "_");
  }
  absl::StrAppend(&output, absl::StrJoin(args, ""));

  std::erase(output, ' ');
  absl::c_replace(output, '-', '_');
  absl::c_replace(output, '=', '_');

  output += ext;

  return output;
}

static absl::StatusOr<SubprocessResult> CallDecodecorpus(
    absl::Span<const std::string> args,
    const std::optional<std::filesystem::path>& cwd = std::nullopt,
    std::optional<absl::Duration> timeout = std::nullopt) {
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path path,
      xls::GetXlsRunfilePath("external/zstd/decodecorpus"));

  std::vector<std::string> cmd = {path};
  cmd.insert(cmd.end(), args.begin(), args.end());
  return SubprocessErrorAsStatus(xls::InvokeSubprocess(cmd));
}

absl::StatusOr<std::vector<uint8_t>> GenerateFrameHeader(int seed, bool magic) {
  std::array<std::string, 4> args;
  args[0] = "-s" + std::to_string(seed);
  args[1] = (magic) ? "" : "--no-magic";
  args[2] = "--frame-header-only";
  std::filesystem::path output_path =
      std::filesystem::temp_directory_path() /
      std::filesystem::path(
          CreateNameForGeneratedFile(absl::MakeSpan(args), ".zstd", "fh"));
  args[3] = "-p" + std::string(output_path);

  XLS_ASSIGN_OR_RETURN(auto result, CallDecodecorpus(args));
  XLS_ASSIGN_OR_RETURN(auto raw_data, xls::GetFileContents(output_path));
  std::remove(output_path.c_str());
  return std::vector<uint8_t>(raw_data.begin(), raw_data.end());
}

absl::StatusOr<std::vector<uint8_t>> GenerateFrame(int seed, BlockType btype) {
  std::vector<std::string> args;
  args.push_back("-s" + std::to_string(seed));
  if (btype != BlockType::RANDOM) {
    args.push_back("--block-type=" + std::to_string(static_cast<int>(btype)));
  }
  if (btype == BlockType::RLE) {
    args.push_back("--content-size");
  }
  // Test payloads up to 16KB
  args.push_back("--max-content-size-log=14");
  std::filesystem::path output_path =
      std::filesystem::temp_directory_path() /
      std::filesystem::path(
          CreateNameForGeneratedFile(absl::MakeSpan(args), ".zstd", "frame"));
  args.push_back("-p" + std::string(output_path));

  XLS_ASSIGN_OR_RETURN(auto result, CallDecodecorpus(args));
  XLS_ASSIGN_OR_RETURN(auto raw_data, xls::GetFileContents(output_path));
  std::remove(output_path.c_str());
  return std::vector<uint8_t>(raw_data.begin(), raw_data.end());
}

}  // namespace xls::zstd
