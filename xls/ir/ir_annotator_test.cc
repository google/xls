// Copyright 2026 The XLS Authors
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

#include "xls/ir/ir_annotator.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"

namespace xls {
namespace {

class FakeAnnotator : public IrAnnotator {
 public:
  FakeAnnotator(std::string prefix = "", std::string suffix = "")
      : prefix_(std::move(prefix)), suffix_(std::move(suffix)) {}
  Annotation NodeAnnotation(Node* node) const override {
    return {.prefix = prefix_.empty() ? std::nullopt : std::optional(prefix_),
            .suffix = suffix_.empty() ? std::nullopt : std::optional(suffix_)};
  }

 private:
  std::string prefix_;
  std::string suffix_;
};

TEST(AnnotationTest, Decorate) {
  EXPECT_EQ(Annotation().Decorate("node"), "node");
  EXPECT_EQ(Annotation{.prefix = "pre"}.Decorate("node"), "pre node");
  EXPECT_EQ(Annotation{.suffix = "suf"}.Decorate("node"), "node suf");
  EXPECT_EQ((Annotation{.prefix = "pre", .suffix = "suf"}.Decorate("node")),
            "pre node suf");
}

TEST(AnnotationTest, Combine) {
  {
    Annotation q = Annotation::Combine(Annotation(), Annotation());
    EXPECT_FALSE(q.filter);
    EXPECT_FALSE(q.prefix.has_value());
    EXPECT_FALSE(q.suffix.has_value());
  }
  {
    Annotation a{.filter = true, .prefix = "pre_a", .suffix = "suf_a"};
    Annotation b{.filter = false, .prefix = "pre_b", .suffix = "suf_b"};
    Annotation q = Annotation::Combine(a, b);
    EXPECT_TRUE(q.filter);
    EXPECT_EQ(*q.prefix, "pre_a pre_b");
    EXPECT_EQ(*q.suffix, "suf_a suf_b");
  }
  {
    Annotation a{.prefix = "pre_a"};
    Annotation b{.suffix = "suf_b"};
    Annotation q = Annotation::Combine(a, b);
    EXPECT_EQ(*q.prefix, "pre_a");
    EXPECT_EQ(*q.suffix, "suf_b");
  }
}

TEST(IrAnnotatorTest, JoinerCtad) {
  FakeAnnotator a1("a1");
  FakeAnnotator a2("a2");
  IrAnnotatorJoiner joiner(a1, a2);
  // If the above compiles, CTAD is working.
}

TEST(IrAnnotatorTest, JoinerBasic) {
  FakeAnnotator a1("a1", "s1");
  FakeAnnotator a2("a2", "s2");
  IrAnnotatorJoiner joiner(a1, a2);
  Annotation q = joiner.NodeAnnotation(nullptr);
  EXPECT_EQ(*q.prefix, "a1 a2");
  EXPECT_EQ(*q.suffix, "s1 s2");
}

class OrderAnnotator : public IrAnnotator {
 public:
  explicit OrderAnnotator(std::vector<Node*> order)
      : order_(std::move(order)) {}
  std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const override {
    return order_;
  }

 private:
  std::vector<Node*> order_;
};

TEST(IrAnnotatorTest, JoinerNodeOrder) {
  Node* n1 = reinterpret_cast<Node*>(1);
  Node* n2 = reinterpret_cast<Node*>(2);
  OrderAnnotator a1({n1, n2});
  OrderAnnotator a2({n2, n1});
  IrAnnotatorJoiner j1(a1, a2);
  EXPECT_THAT(*j1.NodeOrder(nullptr), testing::ElementsAre(n1, n2));

  IrAnnotatorJoiner j2(a2, a1);
  EXPECT_THAT(*j2.NodeOrder(nullptr), testing::ElementsAre(n2, n1));
}

class TopoSortAnnotatorTest : public IrTestBase {};

TEST_F(TopoSortAnnotatorTest, Basic) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
    package my_package

    fn my_fn(a: bits[32], b: bits[32]) -> bits[32] {
      add.1: bits[32] = add(a, b)
      ret sub.2: bits[32] = sub(add.1, a)
    }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("my_fn"));
  TopoSortAnnotator annotator;
  std::optional<std::vector<Node*>> order = annotator.NodeOrder(f);
  ASSERT_TRUE(order.has_value());
  // Verify it's a topo sort.
  absl::flat_hash_set<Node*> seen;
  for (Node* node : *order) {
    for (Node* operand : node->operands()) {
      EXPECT_TRUE(seen.contains(operand))
          << "Node " << node->GetName() << " appears before its operand "
          << operand->GetName();
    }
    seen.insert(node);
  }
}

TEST_F(TopoSortAnnotatorTest, Disabled) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
    package my_package

    fn my_fn(a: bits[32], b: bits[32]) -> bits[32] {
      ret add.1: bits[32] = add(a, b)
    }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("my_fn"));
  TopoSortAnnotator annotator(/*topo_sort=*/false);
  EXPECT_FALSE(annotator.NodeOrder(f).has_value());
}

}  // namespace
}  // namespace xls
