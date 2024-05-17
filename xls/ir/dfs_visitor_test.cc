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

#include "xls/ir/dfs_visitor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

// A testing visitor which records which nodes it has visited.
class TestVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status DefaultHandler(Node* node) override {
    VLOG(1) << "Visiting " << node->GetName();
    visited_.push_back(node);
    visited_set_.insert(node);
    return absl::OkStatus();
  }

  // Returns the ordered set of visited nodes.
  const std::vector<Node*>& visited() const { return visited_; }

  int64_t visited_count() const { return visited_.size(); }

  // Returns the total number of unique nodes visited.
  int64_t unique_visited_count() const { return visited_set_.size(); }

 private:
  std::vector<Node*> visited_;
  absl::flat_hash_set<Node*> visited_set_;
};

class DfsVisitorTest : public IrTestBase {};

TEST_F(DfsVisitorTest, VisitSingleNode) {
  std::string input = R"(
fn single_node() -> bits[32] {
   ret literal.1: bits[32] = literal(value=42)
})";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(input, p.get()));

  TestVisitor v;
  EXPECT_FALSE(v.IsVisited(f->return_value()));
  XLS_ASSERT_OK(f->return_value()->Accept(&v));
  EXPECT_TRUE(v.IsVisited(f->return_value()));
  EXPECT_EQ(1, v.visited_count());
  EXPECT_THAT(v.visited(), ::testing::ElementsAre(f->return_value()));

  // Calling it again should not revisit the node.
  XLS_ASSERT_OK(f->return_value()->Accept(&v));
  EXPECT_EQ(1, v.visited_count());
}

TEST_F(DfsVisitorTest, VisitGraph) {
  std::string input = R"(
fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(input, p.get()));

  {
    // Visit from the root.
    TestVisitor v;
    XLS_ASSERT_OK(f->return_value()->Accept(&v));
    EXPECT_EQ(5, v.visited_count());
    EXPECT_EQ(5, v.unique_visited_count());
    EXPECT_EQ(FindNode("sub.3", f), v.visited().back());
    EXPECT_THAT(v.visited(),
                ::testing::UnorderedElementsAre(
                    FindNode("sub.3", f), FindNode("add.2", f),
                    FindNode("and.1", f), FindNode("p", f), FindNode("q", f)));
  }

  {
    // Visit from an interior node.
    TestVisitor v;
    XLS_ASSERT_OK(FindNode("and.1", f)->Accept(&v));
    EXPECT_EQ(3, v.visited_count());
    EXPECT_THAT(v.visited(),
                ::testing::UnorderedElementsAre(
                    FindNode("and.1", f), FindNode("p", f), FindNode("q", f)));
    EXPECT_FALSE(v.IsVisited(FindNode("add.2", f)));
    EXPECT_FALSE(v.IsVisited(FindNode("sub.3", f)));
  }
}

TEST_F(DfsVisitorTest, FunctionVisit) {
  // The umul operation is dead.
  std::string input = R"(
fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  umul.4: bits[42] = umul(add.2, p)
  ret sub.3: bits[42] = sub(add.2, add.2)
}
)";
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(input, p.get()));

  {
    // Visit from the root should only visit 5 nodes, not the dead umul.
    TestVisitor v;
    XLS_ASSERT_OK(f->return_value()->Accept(&v));
    EXPECT_EQ(5, v.visited_count());
    EXPECT_FALSE(v.IsVisited(FindNode("umul.4", f)));
  }

  {
    // Calling Function::Accept should visit all nodes in the graph including
    // the dead multiply.
    TestVisitor v;
    XLS_ASSERT_OK(f->Accept(&v));
    EXPECT_EQ(6, v.visited_count());
    EXPECT_TRUE(v.IsVisited(FindNode("umul.4", f)));
  }
}

}  // namespace
}  // namespace xls
