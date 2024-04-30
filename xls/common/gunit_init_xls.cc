// Copyright 2023 The XLS Authors
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

#include "xls/common/gunit_init_xls.h"

#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/init_xls.h"

namespace xls {

std::vector<std::string_view> InitXlsForTest(std::string_view usage, int argc,
                                             char* argv[]) {
  // InitGoogleTest calls ParseAbslFlags with gtest configured to use absl (like
  // we do).
  testing::InitGoogleTest(&argc, argv);

  internal::InitXlsPostAbslFlagParse();

  return std::vector<std::string_view>(argv + 1, argv + argc);
}
}  // namespace xls
