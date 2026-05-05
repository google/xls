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

#include "xls/contrib/eco/xls_ir_to_graph.h"

#include <memory>
#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/eco/graph.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

TEST(XlsIrToGraphTest, ConvertsFunctionToGraph) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
package test

top fn main(x: bits[32], y: bits[32]) -> bits[32] {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(x, y, id=4)
  ret result: bits[32] = sub(add.4, literal.3, id=5)
}
)"));
  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());

  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph graph, XlsIrToGraph(*top));

  EXPECT_EQ(graph.nodes.size(), 5);
  EXPECT_EQ(graph.edges.size(), 4);
  ASSERT_TRUE(graph.return_node_name.has_value());
  EXPECT_EQ(*graph.return_node_name, "result");

  ASSERT_TRUE(graph.node_name_to_index.contains("x"));
  ASSERT_TRUE(graph.node_name_to_index.contains("y"));
  ASSERT_TRUE(graph.node_name_to_index.contains("literal.3"));
  ASSERT_TRUE(graph.node_name_to_index.contains("add.4"));
  ASSERT_TRUE(graph.node_name_to_index.contains("result"));

  const int x = graph.node_name_to_index.at("x");
  const int y = graph.node_name_to_index.at("y");
  const int literal = graph.node_name_to_index.at("literal.3");
  const int add = graph.node_name_to_index.at("add.4");
  const int result = graph.node_name_to_index.at("result");

  EXPECT_TRUE(graph.has_edge(x, add, 0));
  EXPECT_TRUE(graph.has_edge(y, add, 1));
  EXPECT_TRUE(graph.has_edge(add, result, 0));
  EXPECT_TRUE(graph.has_edge(literal, result, 1));

  EXPECT_TRUE(Contains(graph.nodes[literal].cost_attributes, "op=literal"));
  EXPECT_TRUE(
      Contains(graph.nodes[literal].cost_attributes, "node_attributes="));
  EXPECT_TRUE(Contains(graph.nodes[literal].cost_attributes, "\"value\""));
  EXPECT_TRUE(Contains(graph.nodes[add].cost_attributes,
                       "operand_dtype_strs=bits[32],bits[32]"));
}

TEST(XlsIrToGraphTest, IncludesDebugNodesFromIr) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
package test

top fn main(tkn: token, cond: bits[1], x: bits[5]) -> bits[5] {
  assert.4: token = assert(tkn, cond, message="boom", label="label", id=4)
  trace.5: token = trace(assert.4, cond, format="x is {}", data_operands=[x], id=5)
  ret gate.6: bits[5] = gate(cond, x, id=6)
}
)"));
  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());

  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph graph, XlsIrToGraph(*top));

  ASSERT_TRUE(graph.node_name_to_index.contains("assert.4"));
  ASSERT_TRUE(graph.node_name_to_index.contains("trace.5"));
  const int assert_node = graph.node_name_to_index.at("assert.4");
  const int trace_node = graph.node_name_to_index.at("trace.5");

  EXPECT_TRUE(graph.has_edge(assert_node, trace_node, 0));
  EXPECT_TRUE(
      Contains(graph.nodes[assert_node].cost_attributes, "op=assert"));
  EXPECT_TRUE(
      Contains(graph.nodes[assert_node].cost_attributes, "\"message_\""));
  EXPECT_TRUE(Contains(graph.nodes[trace_node].cost_attributes, "op=trace"));
  EXPECT_TRUE(Contains(graph.nodes[trace_node].cost_attributes, "\"format\""));
}

TEST(XlsIrToGraphTest, IncludesProcStateReadAttributes) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(R"(
package test

top proc main(st: bits[32], init={42}) {
  st: bits[32] = state_read(state_element=st, id=1)
  next_value.2: () = next_value(param=st, value=st, id=2)
}
)"));
  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());

  XLS_ASSERT_OK_AND_ASSIGN(XLSGraph graph, XlsIrToGraph(*top));

  ASSERT_TRUE(graph.node_name_to_index.contains("st"));
  const int state_read = graph.node_name_to_index.at("st");
  EXPECT_TRUE(Contains(graph.nodes[state_read].cost_attributes,
                       "op=state_read"));
  EXPECT_TRUE(Contains(graph.nodes[state_read].cost_attributes,
                       "state_element=st"));
  EXPECT_TRUE(Contains(graph.nodes[state_read].cost_attributes, "init=42"));
  EXPECT_TRUE(Contains(graph.nodes[state_read].cost_attributes, "index=0"));
}

}  // namespace
}  // namespace xls
