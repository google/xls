// Copyright 2021 The XLS Authors
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

#include "xls/dslx/type_info_to_proto.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls::dslx {
namespace {

TEST(TypeInfoToProtoTest, IdentityFunction) {
  auto import_data = ImportData::CreateForTest();
  std::string program = R"(fn id(x: u32) -> u32 { x })";
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip, TypeInfoToProto(*tm.type_info));
  const std::vector<std::string> kWant = {
      /*0=*/
      "0:0-0:26: FUNCTION :: `fn id(x: u32) -> u32 {\n  x\n}` :: (uN[32]) -> "
      "uN[32]",
      /*1=*/"0:3-0:5: NAME_DEF :: `id` :: (uN[32]) -> uN[32]",
      /*2=*/"0:6-0:7: NAME_DEF :: `x` :: uN[32]",
      /*3=*/"0:6-0:12: PARAM :: `x: u32` :: uN[32]",
      /*4=*/"0:9-0:12: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*5=*/"0:17-0:20: TYPE_ANNOTATION :: `u32` :: uN[32]",
      /*6=*/"0:23-0:24: NAME_REF :: `x` :: uN[32]",
  };
  ASSERT_EQ(kWant.size(), tip.nodes_size());
  std::vector<std::string> got;
  for (int64_t i = 0; i < tip.nodes_size(); ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(std::string node_str,
                             ToHumanString(tip.nodes(i), *tm.module));
    EXPECT_EQ(node_str, kWant[i]) << "at index: " << i;
  }
}

}  // namespace
}  // namespace xls::dslx
