// Copyright 2025 The XLS Authors
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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/text_format.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

// Performs tests on the IrFuzzBuilder by manually creating a FuzzProgramProto,
// instantiating it into its IR version, and manually verifying the IR is
// correct.
absl::Status EquateProtoToIrTest(
    std::string proto_string, testing::Matcher<const Node*> expected_ir_node) {
  // Create the package.
  std::unique_ptr<Package> p = std::make_unique<VerifiedPackage>(kFuzzTestName);
  FunctionBuilder fb(kFuzzTestName, p.get());
  // Create the proto from the string.
  FuzzProgramProto fuzz_program;
  XLS_RET_CHECK(
      google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  // Generate the IR from the proto.
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue proto_ir = ir_fuzz_builder.BuildIr();
  XLS_ASSIGN_OR_RETURN(Function * f, fb.BuildWithReturnValue(proto_ir));
  VLOG(3) << "IR Fuzzer-2: IR:" << "\n" << f->DumpIr() << "\n";
  // Verify that the proto_ir_node matches the expected_ir_node.
  EXPECT_THAT(f->return_value(), expected_ir_node);
  return absl::OkStatus();
}

TEST(IrFuzzBuilderTest, AddTwoLiterals) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            bit_width: 64
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          literal {
            bit_width: 64
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          add {
            bit_width: 64
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 1
            }
          }
        }
      )",
      10, 20);
  auto expected_ir_node =
      m::Add(m::Literal(UBits(10, 64)), m::Literal(UBits(20, 64)));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddTwoParams) {
  std::string proto_string = absl::StrFormat(
      R"(
      combine_stack_method: LAST_ELEMENT_METHOD
      fuzz_ops {
        param {
          bit_width: 64
        }
      }
      fuzz_ops {
        param {
          bit_width: 64
        }
      }
      fuzz_ops {
        add {
          bit_width: 64
          lhs_idx {
            stack_idx: 0
          }
          rhs_idx {
            stack_idx: 1
          }
        }
      }
    )");
  auto expected_ir_node = m::Add(m::Param("p0"), m::Param("p1"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddLiteralsAndParamsAndAdds) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            bit_width: 64
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          param {
            bit_width: 64
          }
        }
        fuzz_ops {
          add {
            bit_width: 64
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 1
            }
          }
        }
        fuzz_ops {
          literal {
            bit_width: 64
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          param {
            bit_width: 64
          }
        }
        fuzz_ops {
          add {
            bit_width: 64
            lhs_idx {
              stack_idx: 3
            }
            rhs_idx {
              stack_idx: 4
            }
          }
        }
        fuzz_ops {
          add {
            bit_width: 64
            lhs_idx {
              stack_idx: 2
            }
            rhs_idx {
              stack_idx: 5
            }
          }
        }
      )",
      10, 20);
  auto expected_ir_node =
      m::Add(m::Add(m::Literal(UBits(10, 64)), m::Param("p1")),
             m::Add(m::Literal(UBits(20, 64)), m::Param("p4")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, SingleOpAddStack) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: ADD_STACK_METHOD
        fuzz_ops {
          param {
            bit_width: 64
          }
        }
      )");
  auto expected_ir_node = m::Param("p0");
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddOpThenAddStack) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: ADD_STACK_METHOD
        fuzz_ops {
          literal {
            bit_width: 64
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          param {
            bit_width: 64
          }
        }
        fuzz_ops {
          add {
            bit_width: 64
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 1
            }
          }
        }
      )",
      10);
  auto expected_ir_node =
      m::Add(m::Add(m::Literal(UBits(10, 64)), m::Param("p1")),
             m::Add(m::Literal(UBits(10, 64)), m::Param("p1")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddOutOfBoundsIdxs) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            bit_width: 64
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          param {
            bit_width: 64
          }
        }
        fuzz_ops {
          add {
            bit_width: 64
            lhs_idx {
              stack_idx: 2
            }
            rhs_idx {
              stack_idx: -1
            }
          }
        }
      )",
      10);
  auto expected_ir_node = m::Add(m::Literal(UBits(10, 64)), m::Param("p1"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, LiteralValueOverBoundsOfSmallWidth) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            bit_width: 1
            value_bytes: "\x%x"
          }
        }
      )",
      1000000);
  auto expected_ir_node = m::Literal(UBits(0, 1));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddDifferentWidthsWithExtensions) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          add {
            bit_width: 40
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: ZERO_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Add(m::ZeroExt(m::Param("p0")), m::SignExt(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddWithSliceAndExtension) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 1
          }
        }
        fuzz_ops {
          param {
            bit_width: 50
          }
        }
        fuzz_ops {
          add {
            bit_width: 25
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: ZERO_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 1
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Add(m::ZeroExt(m::Param("p0")), m::BitSlice(m::Param("p1"), 0, 25));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddStackWithDifferentWidths) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: ADD_STACK_METHOD
        fuzz_ops {
          param {
            bit_width: 50
          }
        }
        fuzz_ops {
          param {
            bit_width: 1
          }
        }
        fuzz_ops {
          param {
            bit_width: 25
          }
        }
      )");
  auto expected_ir_node = m::Add(
      m::ZeroExt(m::Add(m::BitSlice(m::Param("p0"), 0, 1), m::Param("p1"))),
      m::Param("p2"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AddWithLargeWidths) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 800
          }
        }
        fuzz_ops {
          param {
            bit_width: 500
          }
        }
        fuzz_ops {
          add {
            bit_width: 1000
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Add(m::ZeroExt(m::Param("p0")), m::ZeroExt(m::Param("p1")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ConcatOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 0
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::Param("p0"), m::Param("p1"), m::Param("p2"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, EmptyConcat) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          concat {
          }
        }
        fuzz_ops {
          add {
            bit_width: 100
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: ZERO_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Add(m::ZeroExt(m::Concat(m::Literal(UBits(0, 64)))),
             m::SignExt(m::Concat(m::Literal(UBits(0, 64)))));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ShiftOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          shra {
            operand_idx: 0
            amount_idx: 1
          }
        }
        fuzz_ops {
          shrl {
            operand_idx: 0
            amount_idx: 1
          }
        }
        fuzz_ops {
          shll {
            operand_idx: 0
            amount_idx: 1
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
            operand_idxs: 4
          }
        }
      )");
  auto expected_ir_node = m::Concat(m::Shra(m::Param("p0"), m::Param("p1")),
                                    m::Shrl(m::Param("p0"), m::Param("p1")),
                                    m::Shll(m::Param("p0"), m::Param("p1")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, NaryOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          or_op {
            bit_width: 20
            operand_idxs {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: ZERO_EXTEND_METHOD
              }
            }
            operand_idxs {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            operand_idxs {
              stack_idx: 1
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
        fuzz_ops {
          nor {
          }
        }
        fuzz_ops {
          xor_op {
            bit_width: 10
            operand_idxs {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          and_op {
            bit_width: 10
            operand_idxs {
              stack_idx: 0
            }
            operand_idxs {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          nand {
            bit_width: 10
            operand_idxs {
              stack_idx: 0
            }
            operand_idxs {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
            operand_idxs: 4
            operand_idxs: 5
            operand_idxs: 6
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::Or(m::ZeroExt(m::Param("p0")), m::SignExt(m::Param("p0")),
                      m::BitSlice(m::Param("p1"), 0, 20)),
                m::Nor(m::Literal(UBits(0, 1))), m::Xor(m::Param("p0")),
                m::And(m::Param("p0"), m::Param("p0")),
                m::Nand(m::Param("p0"), m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ReduceOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          and_reduce {
            operand_idx: 0
          }
        }
        fuzz_ops {
          or_reduce {
            operand_idx: 0
          }
        }
        fuzz_ops {
          xor_reduce {
            operand_idx: 0
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
            operand_idxs: 3
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::AndReduce(m::Param("p0")), m::OrReduce(m::Param("p0")),
                m::XorReduce(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, MulOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          umul {
            lhs_idx: 0
            rhs_idx: 1
          }
        }
        fuzz_ops {
          umul {
            bit_width: 500
            lhs_idx: 0
            rhs_idx: 1
          }
        }
        fuzz_ops {
          smul {
            bit_width: 10
            lhs_idx: 0
            rhs_idx: 1
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
            operand_idxs: 4
          }
        }
      )");
  auto expected_ir_node = m::Concat(m::UMul(m::Param("p0"), m::Param("p1")),
                                    m::UMul(m::Param("p0"), m::Param("p1")),
                                    m::SMul(m::Param("p0"), m::Param("p1")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, DivOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          udiv {
            bit_width: 20
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 1
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
        fuzz_ops {
          sdiv {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
          }
        }
      )");
  auto expected_ir_node = m::Concat(
      m::UDiv(m::SignExt(m::Param("p0")), m::BitSlice(m::Param("p1"), 0, 20)),
      m::SDiv(m::Param("p0"), m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ModOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          umod {
            bit_width: 20
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 1
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
        fuzz_ops {
          smod {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
          }
        }
      )");
  auto expected_ir_node = m::Concat(
      m::UMod(m::SignExt(m::Param("p0")), m::BitSlice(m::Param("p1"), 0, 20)),
      m::SMod(m::Param("p0"), m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, AssociativeOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          add {
            bit_width: 20
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 1
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
        fuzz_ops {
          subtract {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
          }
        }
      )");
  auto expected_ir_node = m::Concat(
      m::Add(m::SignExt(m::Param("p0")), m::BitSlice(m::Param("p1"), 0, 20)),
      m::Sub(m::Param("p0"), m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ComparisonOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          ule {
            bit_width: 20
            lhs_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            rhs_idx {
              stack_idx: 1
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
        fuzz_ops {
          ult {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          uge {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          ugt {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          sle {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          slt {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          sge {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          sgt {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          eq {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          ne {
            bit_width: 10
            lhs_idx {
              stack_idx: 0
            }
            rhs_idx {
              stack_idx: 0
            }
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 2
            operand_idxs: 3
            operand_idxs: 4
            operand_idxs: 5
            operand_idxs: 6
            operand_idxs: 7
            operand_idxs: 8
            operand_idxs: 9
            operand_idxs: 10
            operand_idxs: 11
          }
        }
      )");
  auto expected_ir_node = m::Concat(
      m::ULe(m::SignExt(m::Param("p0")), m::BitSlice(m::Param("p1"), 0, 20)),
      m::ULt(m::Param("p0"), m::Param("p0")),
      m::UGe(m::Param("p0"), m::Param("p0")),
      m::UGt(m::Param("p0"), m::Param("p0")),
      m::SLe(m::Param("p0"), m::Param("p0")),
      m::SLt(m::Param("p0"), m::Param("p0")),
      m::SGe(m::Param("p0"), m::Param("p0")),
      m::SGt(m::Param("p0"), m::Param("p0")),
      m::Eq(m::Param("p0"), m::Param("p0")),
      m::Ne(m::Param("p0"), m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, InvertOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          negate {
            operand_idx: 0
          }
        }
        fuzz_ops {
          not_op {
            operand_idx: 0
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::Neg(m::Param("p0")), m::Not(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, SelectOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 1
          }
        }
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          select {
            bit_width: 20
            selector_idx: 0
            case_idxs {
              stack_idx: 1
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            case_idxs {
              stack_idx: 2
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
      )");
  auto expected_ir_node = m::Select(
      m::Param("p0"),
      {m::SignExt(m::Param("p1")), m::BitSlice(m::Param("p2"), 0, 20)});
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, SelectWithLargeSelectorWidth) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 1000
          }
        }
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          select {
            bit_width: 10
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
            default_value_idx {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Select(m::Param("p0"), {m::Param("p1")}, m::Param("p1"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, SelectWithSmallSelectorWidth) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 1
          }
        }
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          select {
            bit_width: 10
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Select(m::Param("p0"), {m::Param("p1"), m::Param("p1")});
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, SelectWithUselessDefault) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 1
          }
        }
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          select {
            bit_width: 10
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
            default_value_idx {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Select(m::Param("p0"), {m::Param("p1"), m::Param("p1")});
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, SelectNeedingDefault) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 1
          }
        }
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          select {
            bit_width: 10
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node =
      m::Select(m::Param("p0"), {m::Param("p1")}, m::ZeroExt(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, OneHotOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          one_hot {
            input_idx: 0
            priority: LSB_PRIORITY
          }
        }
        fuzz_ops {
          one_hot {
            input_idx: 0
            priority: MSB_PRIORITY
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::OneHot(LsbOrMsb::kLsb), m::OneHot(LsbOrMsb::kMsb));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, OneHotSelectOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 3
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          param {
            bit_width: 100
          }
        }
        fuzz_ops {
          one_hot_select {
            bit_width: 40
            selector_idx: 0
            case_idxs {
              stack_idx: 1
              width_fitting_method {
                increase_width_method: ZERO_EXTEND_METHOD
              }
            }
            case_idxs {
              stack_idx: 2
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            case_idxs {
              stack_idx: 3
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
      )");
  auto expected_ir_node = m::OneHotSelect(
      m::Param("p0"), {m::ZeroExt(m::Param("p1")), m::SignExt(m::Param("p2")),
                       m::BitSlice(m::Param("p3"), 0, 40)});
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, OneHotSelectWithLargeSelectorWidth) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 100
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          one_hot_select {
            bit_width: 20
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node = m::OneHotSelect(m::Literal(0, 1), {m::Param("p1")});
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, OneHotSelectWithExtraCases) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 3
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          one_hot_select {
            bit_width: 20
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node = m::OneHotSelect(
      m::Param("p0"), {m::Param("p1"), m::Param("p1"), m::Param("p1")});
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, PrioritySelectOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 2
          }
        }
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          priority_select {
            bit_width: 20
            selector_idx: 0
            case_idxs {
              stack_idx: 1
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            case_idxs {
              stack_idx: 3
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
            default_value_idx {
              stack_idx: 2
            }
          }
        }
      )");
  auto expected_ir_node = m::PrioritySelect(
      m::Param("p0"),
      {m::SignExt(m::Param("p1")), m::BitSlice(m::Param("p3"), 0, 20)},
      m::Param("p2"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, PrioritySelectWithLargeSelectorWidth) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 100
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          priority_select {
            bit_width: 20
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
            default_value_idx {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node =
      m::PrioritySelect(m::Literal(0, 1), {m::Param("p1")}, m::Param("p1"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, PrioritySelectWithExtraCases) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 3
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          priority_select {
            bit_width: 20
            selector_idx: 0
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
            case_idxs {
              stack_idx: 1
            }
          }
        }
      )");
  auto expected_ir_node = m::PrioritySelect(
      m::Param("p0"), {m::Param("p1"), m::Param("p1"), m::Param("p1")},
      m::ZeroExt(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, CountOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          clz {
            operand_idx: 0
          }
        }
        fuzz_ops {
          ctz {
            operand_idx: 0
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node = m::Concat(
      m::ZeroExt(
          m::Encode(m::OneHot(m::Reverse(m::Param("p0")), LsbOrMsb::kLsb))),
      m::ZeroExt(m::Encode(m::OneHot(m::Param("p0"), LsbOrMsb::kLsb))));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, MatchOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          param {
            bit_width: 40
          }
        }
        fuzz_ops {
          param {
            bit_width: 50
          }
        }
        fuzz_ops {
          match {
            condition_idx: 2
            case_protos {
              clause_idx {
                stack_idx: 0
                width_fitting_method {
                  increase_width_method: ZERO_EXTEND_METHOD
                }
              }
              value_idx {
                stack_idx: 1
                width_fitting_method {
                  increase_width_method: SIGN_EXTEND_METHOD
                }
              }
            }
            case_protos {
              clause_idx {
                stack_idx: 2
              }
              value_idx {
                stack_idx: 3
                width_fitting_method {
                  decrease_width_method: BIT_SLICE_METHOD
                }
              }
            }
            default_value_idx {
              stack_idx: 3
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
          }
        }
      )");
  auto expected_ir_node = m::PrioritySelect(
      m::Concat(m::Eq(m::Param("p2"), m::Param("p2")),
                m::Eq(m::Param("p2"), m::ZeroExt(m::Param("p0")))),
      {m::SignExt(m::Param("p1")), m::BitSlice(m::Param("p3"), 0, 40)},
      m::BitSlice(m::Param("p3"), 0, 40));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, MatchTrueOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          param {
            bit_width: 40
          }
        }
        fuzz_ops {
          param {
            bit_width: 50
          }
        }
        fuzz_ops {
          match_true {
            case_protos {
              clause_idx {
                stack_idx: 0
              }
              value_idx {
                stack_idx: 1
              }
            }
            case_protos {
              clause_idx {
                stack_idx: 2
              }
              value_idx {
                stack_idx: 3
              }
            }
            default_value_idx {
              stack_idx: 3
            }
          }
        }
      )");
  auto expected_ir_node = m::PrioritySelect(
      m::Concat(m::BitSlice(m::Param("p2"), 0, 1),
                m::BitSlice(m::Param("p0"), 0, 1)),
      {m::BitSlice(m::Param("p1"), 0, 1), m::BitSlice(m::Param("p3"), 0, 1)},
      m::BitSlice(m::Param("p3"), 0, 1));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ReverseOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          reverse {
            operand_idx: 0
          }
        }
      )");
  auto expected_ir_node = m::Reverse(m::Param("p0"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, IdentityOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          identity {
            operand_idx: 0
          }
        }
      )");
  auto expected_ir_node = m::Identity(m::Param("p0"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, ExtendOps) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          sign_extend {
            bit_width: 20
            operand_idx: 0
          }
        }
        fuzz_ops {
          zero_extend {
            bit_width: 5
            operand_idx: 0
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::SignExt(m::Param("p0")), m::ZeroExt(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, BitSliceOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          bit_slice {
            bit_width: 10
            operand_idx: 0
            start: 0
          }
        }
        fuzz_ops {
          bit_slice {
            bit_width: 10
            operand_idx: 0
            start: 100
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node = m::Concat(m::BitSlice(m::Param("p0"), 0, 10),
                                    m::BitSlice(m::Param("p0"), 0, 10));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, BitSliceUpdateOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          param {
            bit_width: 30
          }
        }
        fuzz_ops {
          bit_slice_update {
            operand_idx: 0
            start_idx: 1
            update_value_idx: 2
          }
        }
      )");
  auto expected_ir_node =
      m::BitSliceUpdate(m::Param("p0"), m::Param("p1"), m::Param("p2"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, DynamicBitSliceOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          dynamic_bit_slice {
            bit_width: 30
            operand_idx {
              stack_idx: 0
              width_fitting_method {
                increase_width_method: SIGN_EXTEND_METHOD
              }
            }
            start_idx: 1
          }
        }
      )");
  auto expected_ir_node =
      m::DynamicBitSlice(m::SignExt(m::Param("p0")), m::Param("p1"), 30);
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, EncodeOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          encode {
            operand_idx: 0
          }
        }
      )");
  auto expected_ir_node = m::Encode(m::Param("p0"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, DecodeOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          decode {
            operand_idx: 0
          }
        }
        fuzz_ops {
          decode {
            bit_width: 20
            operand_idx: 0
          }
        }
        fuzz_ops {
          concat {
            operand_idxs: 1
            operand_idxs: 2
          }
        }
      )");
  auto expected_ir_node =
      m::Concat(m::Decode(m::Param("p0")), m::Decode(m::Param("p0")));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

TEST(IrFuzzBuilderTest, GateOp) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            bit_width: 10
          }
        }
        fuzz_ops {
          param {
            bit_width: 20
          }
        }
        fuzz_ops {
          gate {
            condition_idx {
              stack_idx: 0
              width_fitting_method {
                decrease_width_method: BIT_SLICE_METHOD
              }
            }
            data_idx: 1
          }
        }
      )");
  auto expected_ir_node =
      m::Gate(m::BitSlice(m::Param("p0"), 0, 1), m::Param("p1"));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}

}  // namespace
}  // namespace xls
