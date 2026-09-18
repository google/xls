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

#include "xls/ir/ir_test_base.h"

#include <memory>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

class IrTestBaseTest : public IrTestBase {
 protected:
  static constexpr char kTestPackage[] = R"(
package test_package

top fn main(p: bits[8], q: bits[8]) -> bits[8] {
  add.1: bits[8] = add(p, q)
  ret add.2: bits[8] = add(add.1, q)
}
)";

  // Creates an invalid VerifiedPackage, then destroys it.
  void DestructInvalidVerifiedPackage() {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                             ParsePackage(kTestPackage));
    // Set an id to a duplicate value.
    FindNode("add.2", p.get())->SetId(1);
  }
};

TEST_F(IrTestBaseTest, DestructInvalidVerifiedPackage) {
  EXPECT_NONFATAL_FAILURE(
      DestructInvalidVerifiedPackage(),
      "verifier failed on package test_package during destruction");
}

TEST_F(IrTestBaseTest, ValidVerifiedPackageSucceeds) {
  // Verify that a valid VerifiedPackage does not raise any errors during
  // destruction. The package is created then immediately dropped.
  XLS_ASSERT_OK(ParsePackage(kTestPackage).status());
}

TEST_F(IrTestBaseTest, HasNodes) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedPackage> p,
                           ParsePackage(kTestPackage));

  EXPECT_TRUE(HasNode("p", p.get()));
  EXPECT_TRUE(HasNode("q", p.get()));
  EXPECT_TRUE(HasNode("add.1", p.get()));
  EXPECT_TRUE(HasNode("add.2", p.get()));

  FunctionBase* f = FindFunction("main", p.get());
  EXPECT_TRUE(HasNode("p", f));
  EXPECT_TRUE(HasNode("q", f));
  EXPECT_TRUE(HasNode("add.1", f));
  EXPECT_TRUE(HasNode("add.2", f));
}

TEST_F(IrTestBaseTest, Matchers) {
  auto p = CreatePackage();
  Type* type = p->GetArrayType(2, p->GetBitsType(2));
  LeafTypeTree<TernaryVector> ltt(
      type, {TernaryVector{TernaryValue::kKnownOne, TernaryValue::kKnownZero},
             TernaryVector{TernaryValue::kUnknown, TernaryValue::kKnownOne}});
  EXPECT_THAT(ltt, LttIs<TernaryVector>(
                       ElementsAre(TernaryIs("0b01"), TernaryIs("0b1X"))));
  EXPECT_THAT(ltt,
              LttIs<TernaryVector>(ElementsAre(testing::_, TernaryIs("0b1X"))));
  EXPECT_THAT(ltt, LttIs<TernaryVector>(ElementsAre(
                       testing::Not(TernaryIs("0b00")), TernaryIs("0b1X"))));
}

}  // namespace
}  // namespace xls
