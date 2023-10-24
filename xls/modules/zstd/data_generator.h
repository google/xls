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
#ifndef XLS_MODULES_ZSTD_DATA_GENERATOR_H_
#define XLS_MODULES_ZSTD_DATA_GENERATOR_H_

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

enum BlockType {
  RAW,
  RLE,
  COMPRESSED,
  RANDOM,
};

absl::StatusOr<std::vector<uint8_t>> GenerateFrameHeader(int seed, bool magic);
absl::StatusOr<std::vector<uint8_t>> GenerateFrame(int seed, BlockType btype);

}  // namespace xls::zstd

#endif  // XLS_MODULES_ZSTD_DATA_GENERATOR_H_
