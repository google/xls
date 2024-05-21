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

#include "xls/codegen/ffi_instantiation_pass.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::ElementsAre;

class FfiInstantiationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    PassResults results;
    CodegenPassUnit unit(block->package(), block);
    return FfiInstantiationPass().Run(&unit, CodegenPassOptions(), &results);
  }
};

TEST_F(FfiInstantiationPassTest, InvocationsReplacedByInstance) {
  auto p = CreatePackage();
  BitsType* const u32 = p->GetBitsType(32);
  BitsType* const u17 = p->GetBitsType(17);

  // Simple function that has foreign function data attached.
  FunctionBuilder fb(TestName() + "ffi_fun", p.get());
  const BValue param_a = fb.Param("a", u32);
  const BValue param_b = fb.Param("b", u17);
  const BValue add = fb.Add(param_a, fb.ZeroExtend(param_b, 32));
  XLS_ASSERT_OK_AND_ASSIGN(ForeignFunctionData ffd,
                           ForeignFunctionDataCreateFromTemplate(
                               "foo {fn} (.ma({a}), .mb{b}) .out({return})"));
  fb.SetForeignFunctionData(ffd);
  XLS_ASSERT_OK_AND_ASSIGN(Function * ffi_fun, fb.BuildWithReturnValue(add));

  // A block that contains one invocation of that ffi_fun.
  BlockBuilder bb(TestName(), p.get());
  const BValue input_port_a = bb.InputPort("block_a_input", u32);
  const BValue input_port_b = bb.InputPort("block_b_input", u17);
  bb.OutputPort("out", bb.Invoke({input_port_a, input_port_b}, ffi_fun));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Precondition: one invoke(), and no instantiations
  EXPECT_EQ(1, std::count_if(block->nodes().begin(), block->nodes().end(),
                             [](Node* n) { return n->Is<Invoke>(); }));
  EXPECT_THAT(block->GetInstantiations(), testing::IsEmpty());

  // First round we find an invoke to create an instantiation from.
  EXPECT_THAT(Run(block), IsOkAndHolds(true));

  // Nothing to be done the second time around.
  EXPECT_THAT(Run(block), IsOkAndHolds(false));

  // Postcondition: no invoke() and one instantiation.
  // instantiation to be in the block referencing the original function.
  EXPECT_EQ(0, std::count_if(block->nodes().begin(), block->nodes().end(),
                             [](Node* n) { return n->Is<Invoke>(); }));
  ASSERT_EQ(block->GetInstantiations().size(), 1);

  xls::Instantiation* const instantiation = block->GetInstantiations()[0];
  ASSERT_EQ(instantiation->kind(), InstantiationKind::kExtern);

  xls::ExternInstantiation* const extern_inst =
      down_cast<xls::ExternInstantiation*>(instantiation);
  EXPECT_EQ(extern_inst->function(), ffi_fun);
  for (std::string_view param : {"a", "b"}) {
    XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort input_param,
                             extern_inst->GetInputPort(param));
    EXPECT_EQ(input_param.name, param);
    EXPECT_EQ(input_param.type, param == "a" ? u32 : u17);
  }

  XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort return_port,
                           extern_inst->GetOutputPort("return"));
  EXPECT_EQ(return_port.name, "return");
  EXPECT_EQ(return_port.type, u32);

  // Requesting a non-existent port.
  EXPECT_THAT(extern_inst->GetInputPort("bogus"),
              StatusIs(absl::StatusCode::kNotFound));

  // Explicitly testing the resulting block passes verification.
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(FfiInstantiationPassTest, FunctionParameterIsTuple) {
  auto p = CreatePackage();
  BitsType* const u32 = p->GetBitsType(32);
  BitsType* const u17 = p->GetBitsType(17);

  // Simple function that has foreign function data attached.
  FunctionBuilder fb(TestName() + "ffi_fun", p.get());
  const BValue param_x = fb.Param("x", p->GetTupleType({u32, u17}));
  XLS_ASSERT_OK_AND_ASSIGN(ForeignFunctionData ffd,
                           ForeignFunctionDataCreateFromTemplate(
                               "foo {fn} (.xa({x.0}), .mb({x.1}))"));
  fb.SetForeignFunctionData(ffd);
  XLS_ASSERT_OK_AND_ASSIGN(Function * ffi_fun,
                           fb.BuildWithReturnValue(param_x));

  // A block that contains one invocation of that ffi_fun.
  BlockBuilder bb(TestName(), p.get());
  const BValue input_port_a = bb.InputPort("block_a_input", u32);
  const BValue input_port_b = bb.InputPort("block_b_input", u17);
  const BValue param = bb.Tuple({input_port_a, input_port_b});

  bb.OutputPort("out", bb.Invoke({param}, ffi_fun));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Convert to Instationation
  EXPECT_THAT(Run(block), IsOkAndHolds(true));
  ASSERT_EQ(block->GetInstantiations().size(), 1);

  xls::Instantiation* const instantiation = block->GetInstantiations()[0];
  ASSERT_EQ(instantiation->kind(), InstantiationKind::kExtern);
  xls::ExternInstantiation* const extern_inst =
      down_cast<xls::ExternInstantiation*>(instantiation);

  // Make sure that all elements are expanded so that they are accessible
  // to the user in the text template.
  const auto instantiation_inputs = block->GetInstantiationInputs(extern_inst);
  EXPECT_EQ(instantiation_inputs.size(), 3);  // x, x.0, x.1
  std::vector<std::string> input_ports;
  for (const InstantiationInput* input : instantiation_inputs) {
    input_ports.push_back(input->port_name());
  }
  EXPECT_THAT(input_ports, ElementsAre("x", "x.0", "x.1"));

  EXPECT_EQ(extern_inst->function(), ffi_fun);
  for (std::string_view param_name : {"x.0", "x.1"}) {
    XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort input_param,
                             extern_inst->GetInputPort(param_name));
    EXPECT_EQ(input_param.name, param_name);
    EXPECT_EQ(input_param.type, param_name == "x.0" ? u32 : u17);
  }

  // Explicitly testing the resulting block passes verification.
  XLS_EXPECT_OK(VerifyPackage(p.get()));
}

TEST_F(FfiInstantiationPassTest, FunctionTakingNoParametersJustReturns) {
  constexpr int kReturnBitCount = 17;
  auto p = CreatePackage();

  // Simple function that has foreign function data attached.
  FunctionBuilder fb(TestName() + "ffi_fun", p.get());
  BValue retval = fb.Literal(UBits(42, kReturnBitCount));
  XLS_ASSERT_OK_AND_ASSIGN(
      ForeignFunctionData ffd,
      ForeignFunctionDataCreateFromTemplate("foo {fn} (.out({return}))"));
  fb.SetForeignFunctionData(ffd);
  XLS_ASSERT_OK_AND_ASSIGN(Function * ffi_fun, fb.BuildWithReturnValue(retval));

  // A block that contains one invocation of that ffi_fun.
  BlockBuilder bb(TestName(), p.get());
  bb.OutputPort("out", bb.Invoke({}, ffi_fun));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Convert to instance
  EXPECT_THAT(Run(block), IsOkAndHolds(true));

  xls::Instantiation* const instantiation = block->GetInstantiations()[0];
  ASSERT_EQ(instantiation->kind(), InstantiationKind::kExtern);

  xls::ExternInstantiation* const extern_inst =
      down_cast<xls::ExternInstantiation*>(instantiation);

  XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort return_port,
                           extern_inst->GetOutputPort("return"));
  EXPECT_EQ(return_port.name, "return");
  EXPECT_EQ(return_port.type, p->GetBitsType(kReturnBitCount));
}

TEST_F(FfiInstantiationPassTest, FunctionReturningTuple) {
  constexpr int kReturnBitCount[] = {17, 27};
  auto p = CreatePackage();

  // Function returning a tuple.
  FunctionBuilder fb(TestName() + "ffi_fun", p.get());
  BValue retval = fb.Tuple({fb.Literal(UBits(42, kReturnBitCount[0])),
                            fb.Literal(UBits(24, kReturnBitCount[1]))});
  XLS_ASSERT_OK_AND_ASSIGN(
      ForeignFunctionData ffd,
      ForeignFunctionDataCreateFromTemplate("foo {fn} (.foo({return.0}), "
                                            ".bar({return.1}))"));
  fb.SetForeignFunctionData(ffd);
  XLS_ASSERT_OK_AND_ASSIGN(Function * ffi_fun, fb.BuildWithReturnValue(retval));

  // A block that contains one invocation of that ffi_fun.
  BlockBuilder bb(TestName(), p.get());
  bb.OutputPort("out", bb.Invoke({}, ffi_fun));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Convert to instance
  EXPECT_THAT(Run(block), IsOkAndHolds(true));

  xls::Instantiation* const instantiation = block->GetInstantiations()[0];
  ASSERT_EQ(instantiation->kind(), InstantiationKind::kExtern);

  xls::ExternInstantiation* const extern_inst =
      down_cast<xls::ExternInstantiation*>(instantiation);

  for (int i = 0; i < 2; ++i) {
    const std::string return_name = absl::StrCat("return.", i);
    XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort return_port,
                             extern_inst->GetOutputPort(return_name));
    EXPECT_EQ(return_port.name, return_name);
    EXPECT_EQ(return_port.type, p->GetBitsType(kReturnBitCount[i]));
  }
}

TEST_F(FfiInstantiationPassTest, FunctionReturningNestedTuple) {
  constexpr int kFirstTupleElementBitCount = 13;
  constexpr int kNestedTupleBitCount[] = {17, 27};
  auto p = CreatePackage();

  // Function returning a tuple.
  FunctionBuilder fb(TestName() + "ffi_fun", p.get());
  BValue nested = fb.Tuple({fb.Literal(UBits(42, kNestedTupleBitCount[0])),
                            fb.Literal(UBits(24, kNestedTupleBitCount[1]))});
  BValue retval =
      fb.Tuple({fb.Literal(UBits(123, kFirstTupleElementBitCount)), nested});
  XLS_ASSERT_OK_AND_ASSIGN(ForeignFunctionData ffd,
                           ForeignFunctionDataCreateFromTemplate(
                               "foo {fn} (.foo({return.0}), "
                               ".bar({return.1.0}), .baz({return.1.0}))"));
  fb.SetForeignFunctionData(ffd);
  XLS_ASSERT_OK_AND_ASSIGN(Function * ffi_fun, fb.BuildWithReturnValue(retval));

  // A block that contains one invocation of that ffi_fun.
  BlockBuilder bb(TestName(), p.get());
  bb.OutputPort("out", bb.Invoke({}, ffi_fun));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Convert to instance
  EXPECT_THAT(Run(block), IsOkAndHolds(true));

  xls::Instantiation* const instantiation = block->GetInstantiations()[0];
  ASSERT_EQ(instantiation->kind(), InstantiationKind::kExtern);

  xls::ExternInstantiation* const extern_inst =
      down_cast<xls::ExternInstantiation*>(instantiation);

  // first, non-nested element
  std::string return_name = "return.0";
  XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort return_port,
                           extern_inst->GetOutputPort(return_name));
  EXPECT_EQ(return_port.name, return_name);
  EXPECT_EQ(return_port.type, p->GetBitsType(kFirstTupleElementBitCount));

  // Get nested elements return.1.0,  return.1.1
  for (int i = 0; i < 2; ++i) {
    return_name = absl::StrCat("return.1.", i);
    XLS_ASSERT_OK_AND_ASSIGN(InstantiationPort return_port,
                             extern_inst->GetOutputPort(return_name));
    EXPECT_EQ(return_port.name, return_name);
    EXPECT_EQ(return_port.type, p->GetBitsType(kNestedTupleBitCount[i]));
  }

  // Test some invalid accesses and usefulness of error messages
  // by Instantiation (ir/instantiation.cc)
  EXPECT_THAT(
      extern_inst->GetOutputPort("return.1"),
      StatusIs(absl::StatusCode::kNotFound,
               testing::HasSubstr("return.1 is a tuple (with 2 fields), "
                                  "expected sub-access by .<number>")));

  EXPECT_THAT(
      extern_inst->GetOutputPort("return.42"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("Invalid index into tuple `return.42`; "
                                  "expected to be in range 0..1")));

  EXPECT_THAT(extern_inst->GetOutputPort("return.1.1.1"),
              StatusIs(absl::StatusCode::kNotFound,
                       testing::HasSubstr(
                           "Attempting to access tuple-field `return.1.1.1` "
                           "but `return.1.1` is already a scalar")));
}

}  // namespace
}  // namespace xls::verilog
