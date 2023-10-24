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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/ir/value.h"

namespace xls::zstd {

static void PrintRawDataVector(std::vector<uint8_t> vector) {
  for (int i = 0; i < vector.size(); ++i) {
    std::cout << std::setfill('0') << std::setw(8) << std::hex << i << ": "
              << "0x" << std::setw(2) << std::hex << int(vector[i])
              << std::endl;
  }
}

static absl::StatusOr<std::vector<uint8_t>> ReadFileAsRawData(
    const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return absl::NotFoundError("Unable to open a test file");
  }

  std::vector<uint8_t> raw_data((std::istreambuf_iterator<char>(file)),
                                (std::istreambuf_iterator<char>()));
  return raw_data;
}

static std::string CreateNameForGeneratedFile(
    absl::Span<std::string> args, std::string_view ext,
    std::optional<std::string_view> prefix) {
  std::string output;

  if (prefix.has_value()) {
    output += prefix.value();
    output += "_";
  }

  for (auto const& x : args) {
    output += x;
  }
  auto nospace_output = std::remove(output.begin(), output.end(), ' ');
  output.erase(nospace_output, output.end());
  std::replace(output.begin(), output.end(), '-', '_');

  output += ext;

  return output;
}

static absl::StatusOr<SubprocessResult> CallDecodecorpus(
    absl::Span<const std::string> args,
    std::optional<std::filesystem::path> cwd = std::nullopt,
    std::optional<absl::Duration> timeout = std::nullopt) {
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path path,
      xls::GetXlsRunfilePath("external/com_github_facebook_zstd/decodecorpus"));

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
  auto raw_data = ReadFileAsRawData(output_path);
  std::remove(output_path.c_str());
  return raw_data;
}

absl::StatusOr<std::vector<uint8_t>> GenerateFrame(int seed, BlockType btype) {
  std::vector<std::string> args;
  args.push_back("-s" + std::to_string(seed));
  std::filesystem::path output_path =
      std::filesystem::temp_directory_path() /
      std::filesystem::path(
          CreateNameForGeneratedFile(absl::MakeSpan(args), ".zstd", "fh"));
  args.push_back("-p" + std::string(output_path));
  if (btype != BlockType::RANDOM)
    args.push_back("--block-type=" + std::to_string(btype));
  if (btype == BlockType::RLE) args.push_back("--content-size");
  // Test payloads up to 16KB
  args.push_back("--max-content-size-log=14");

  XLS_ASSIGN_OR_RETURN(auto result, CallDecodecorpus(args));
  auto raw_data = ReadFileAsRawData(output_path);
  std::remove(output_path.c_str());
  return raw_data;
}

}  // namespace xls::zstd
