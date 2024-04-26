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

#include "xls/fdo/extract_nodes.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"

namespace xls {
namespace {

class ExtractNodesTest : public IrTestBase {};

TEST_F(ExtractNodesTest, SimpleExtraction) {
  const std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  ret add.1: bits[3] = add(i0, i1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes({FindNode("i0", function),
                                    FindNode("i1", function),
                                    FindNode("add.1", function)});

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> tmp_package,
                           ExtractNodes(nodes, "test"));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f,
                           tmp_package->GetFunction("test"));
  const std::string expected_result_ir =
      R"(fn test(i0: bits[3], i1: bits[3]) -> bits[3] {
  ret add.3: bits[3] = add(i0, i1, id=3)
}
)";
  EXPECT_EQ(tmp_f->DumpIr(), expected_result_ir);
}

TEST_F(ExtractNodesTest, ExtractionWithLivein) {
  const std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  ret sub.2: bits[3] = sub(add.1, i1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes({FindNode("sub.2", function)});

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> tmp_package,
      ExtractNodes(nodes, "test", /*flop_inputs_outputs=*/false));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f,
                           tmp_package->GetFunction("test"));
  const std::string expected_result_ir =
      R"(fn test(add_1: bits[3], i1: bits[3]) -> bits[3] {
  ret sub.3: bits[3] = sub(add_1, i1, id=3)
}
)";
  EXPECT_EQ(tmp_f->DumpIr(), expected_result_ir);
}

TEST_F(ExtractNodesTest, ExtractionWithLiveout) {
  const std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  sub.2: bits[3] = sub(add.1, i1)
  ret or.3: bits[3] = or(sub.2, add.1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes(
      {FindNode("add.1", function), FindNode("sub.2", function)});

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> tmp_package,
                           ExtractNodes(nodes, "test"));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f,
                           tmp_package->GetFunction("test"));
  const std::string expected_result_ir =
      R"(fn test(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.3: bits[3] = add(i0, i1, id=3)
  ret sub.4: bits[3] = sub(add.3, i1, id=4)
}
)";
  EXPECT_EQ(tmp_f->DumpIr(), expected_result_ir);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> tmp_package_with_liveouts,
                           ExtractNodes(nodes, "test",
                                        /*return_all_liveouts=*/true));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f_with_liveouts,
                           tmp_package_with_liveouts->GetFunction("test"));
  const std::string expected_result_ir_with_liveouts =
      R"(fn test(i0: bits[3], i1: bits[3]) -> (bits[3], bits[3]) {
  add.3: bits[3] = add(i0, i1, id=3)
  sub.4: bits[3] = sub(add.3, i1, id=4)
  ret tuple.5: (bits[3], bits[3]) = tuple(add.3, sub.4, id=5)
}
)";
  EXPECT_EQ(tmp_f_with_liveouts->DumpIr(), expected_result_ir_with_liveouts);
}

}  // namespace
}  // namespace xls
