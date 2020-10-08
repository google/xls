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

#include "xls/ir/verifier.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

class VerifierTest : public IrTestBase {
 protected:
  VerifierTest() {}
};

TEST_F(VerifierTest, WellFormedPackage) {
  std::string input = R"(
package WellFormedPackage

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

fn graph2(a: bits[16]) -> bits[16] {
  neg.4: bits[16] = neg(a)
  ret not.5: bits[16] = not(neg.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK(VerifyFunction(FindFunction("graph", p.get())));
  XLS_ASSERT_OK(VerifyFunction(FindFunction("graph2", p.get())));
}

TEST_F(VerifierTest, NonUniqueNodeId) {
  std::string input = R"(
package NonUniqueNodeId

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.2: bits[42] = sub(add.2, add.2)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("graph"));
  EXPECT_THAT(VerifyFunction(f), StatusIs(absl::StatusCode::kInternal,
                                          HasSubstr("ID 2 is not unique")));
}

TEST_F(VerifierTest, NonUniqueFunctionName) {
  std::string input = R"(
package NonUniqueFunctionName

fn graph(p: bits[42], q: bits[42]) -> bits[42] {
  and.1: bits[42] = and(p, q)
  add.2: bits[42] = add(and.1, q)
  ret sub.3: bits[42] = sub(add.2, add.2)
}

fn graph(a: bits[16]) -> bits[16] {
  neg.4: bits[16] = neg(a)
  ret not.5: bits[16] = not(neg.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Function or proc with name graph is not unique")));
}

TEST_F(VerifierTest, BinOpOperandTypeMismatch) {
  std::string input = R"(
package BinOpOperandTypeMismatch

fn graph(p: bits[2], q: bits[42], r: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(q, r)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("graph"));
  // Replace lhs of the 'and' with a different bit-width value.
  FindNode("and.1", f)->ReplaceOperand(FindNode("q", f), FindNode("p", f));
  EXPECT_THAT(VerifyFunction(f),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Type of operand 0 (bits[2] via p) "
                                 "does not match type of and.1")));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Type of operand 0 (bits[2] via p) does not "
                                 "match type of and.1")));
}

TEST_F(VerifierTest, SelectWithUselessDefault) {
  std::string input = R"(
package p

fn f(p: bits[1], q: bits[42], r: bits[42]) -> bits[42] {
  literal.1: bits[42] = literal(value=42)
  ret sel.2: bits[42] = sel(p, cases=[q, r], default=literal.1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Select has useless default value")));
}

TEST_F(VerifierTest, SelectWithMissingDefault) {
  std::string input = R"(
package p

fn f(p: bits[2], q: bits[42], r: bits[42], s:bits[42]) -> bits[42] {
  ret sel.1: bits[42] = sel(p, cases=[q, r, s])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Select has no default value")));
}

TEST_F(VerifierTest, SelectWithTooNarrowSelector) {
  std::string input = R"(
package p

fn f(p: bits[1], q: bits[42], r: bits[42], s:bits[42], t:bits[42]) -> bits[42] {
  ret sel.1: bits[42] = sel(p, cases=[q, r, s, t])
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Selector must have at least 2 bits to select amongst 4 cases")));
}

TEST_F(VerifierTest, WellFormedProc) {
  std::string input = R"(
package test_package

proc my_proc(t: token, s: bits[42], init=45) {
  ret tuple.1: (token, bits[42]) = tuple(t, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK(VerifyProc(FindProc("my_proc", p.get())));
}

TEST_F(VerifierTest, ProcMissingReceive) {
  std::string input = R"(
package test_package

chan ch(data: bits[32], id=42, kind=send_receive, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init=45) {
  send.1: token = send(t, data=[s], channel_id=42)
  ret tuple.2: (token, bits[42]) = tuple(send.1, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr("Channel 'ch' (id 42) has no associated receive node")));
}

TEST_F(VerifierTest, MultipleSendNodes) {
  std::string input = R"(
package test_package

chan ch(data: bits[32], id=42, kind=send_only, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init=45) {
  send.1: token = send(t, data=[s], channel_id=42)
  send.2: token = send(send.1, data=[s], channel_id=42)
  ret tuple.3: (token, bits[42]) = tuple(send.2, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(VerifyPackage(p.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Multiple send nodes associated with channel "
                                 "'ch': send.2 and send.1")));
}

TEST_F(VerifierTest, DisconnectedSendNode) {
  std::string input = R"(
package test_package

chan ch(data: bits[32], id=42, kind=send_only, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init=45) {
  after_all.1: token = after_all()
  send.2: token = send(after_all.1, data=[s], channel_id=42)
  ret tuple.3: (token, bits[42]) = tuple(send.2, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Send and receive nodes must be connected to the "
                         "token parameter via a path of tokens: send.2")));
}

TEST_F(VerifierTest, DisconnectedReceiveNode) {
  std::string input = R"(
package test_package

chan ch(data: bits[32], id=42, kind=receive_only, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init=45) {
  receive.1: (token, bits[32]) = receive(t, channel_id=42)
  ret tuple.2: (token, bits[42]) = tuple(t, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Send and receive nodes must be connected to the "
                         "return value via a path of tokens: receive.")));
}

TEST_F(VerifierTest, DisconnectedReturnValueInProc) {
  std::string input = R"(
package test_package

proc my_proc(t: token, s: bits[42], init=45) {
  after_all.1: token = after_all()
  ret tuple.2: (token, bits[42]) = tuple(after_all.1, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Return value of proc must be connected to the token "
                         "parameter via a path of tokens: tuple.2")));
}

TEST_F(VerifierTest, SendOnReceiveOnlyChannel) {
  std::string input = R"(
package test_package

chan ch(data: bits[32], id=42, kind=receive_only, metadata="""module_port { flopped: true }""")

proc my_proc(t: token, s: bits[42], init=45) {
  send.1: token = send(t, data=[s], channel_id=42)
  receive.2: (token, bits[32]) = receive(send.1, channel_id=42)
  tuple_index.3: token = tuple_index(receive.2, index=0)
  ret tuple.4: (token, bits[42]) = tuple(tuple_index.3, s)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackageNoVerify(input));
  EXPECT_THAT(
      VerifyPackage(p.get()),
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "Cannot send over channel ch (42), send operation: send.1")));
}

}  // namespace
}  // namespace xls
