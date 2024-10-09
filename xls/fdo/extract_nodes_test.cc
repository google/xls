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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

class ExtractNodesTest : public IrTestBase {};

TEST_F(ExtractNodesTest, SimpleExtraction) {
  const std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  ret myadd: bits[3] = add(i0, i1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes({FindNode("i0", function),
                                    FindNode("i1", function),
                                    FindNode("myadd", function)});

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> tmp_package,
                           ExtractNodes(nodes, "test"));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f,
                           tmp_package->GetFunction("test"));
  EXPECT_THAT(tmp_f->AsFunctionOrDie()->return_value(),
              m::Add(m::Param("i0"), m::Param("i1")));
}

TEST_F(ExtractNodesTest, ExtractionWithLivein) {
  const std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  myadd: bits[3] = add(i0, i1)
  ret mysub: bits[3] = sub(myadd, i1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes({FindNode("mysub", function)});

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> tmp_package,
      ExtractNodes(nodes, "test", /*return_all_liveouts=*/false));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f,
                           tmp_package->GetFunction("test"));
  EXPECT_THAT(tmp_f->AsFunctionOrDie()->return_value(),
              m::Sub(m::Param("myadd"), m::Param("i1")));
}

TEST_F(ExtractNodesTest, ExtractionWithLiveout) {
  const std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  myadd: bits[3] = add(i0, i1)
  mysub: bits[3] = sub(myadd, i1)
  ret result: bits[3] = or(mysub, myadd)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes(
      {FindNode("myadd", function), FindNode("mysub", function)});

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> tmp_package,
                           ExtractNodes(nodes, "test"));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f,
                           tmp_package->GetFunction("test"));
  EXPECT_THAT(tmp_f->AsFunctionOrDie()->return_value(),
              m::Sub(m::Add(), m::Param("i1")));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> tmp_package_with_liveouts,
                           ExtractNodes(nodes, "test",
                                        /*return_all_liveouts=*/true));
  XLS_ASSERT_OK_AND_ASSIGN(FunctionBase * tmp_f_with_liveouts,
                           tmp_package_with_liveouts->GetFunction("test"));
  EXPECT_THAT(tmp_f_with_liveouts->AsFunctionOrDie()->return_value(),
              m::Tuple(m::Add(), m::Sub()));
}

}  // namespace
}  // namespace xls
