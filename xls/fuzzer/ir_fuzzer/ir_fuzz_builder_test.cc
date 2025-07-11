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
#include "absl/strings/str_format.h"
#include "google/protobuf/text_format.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace m = ::xls::op_matchers;

// Performs tests on the IrFuzzBuilder by manually creating a FuzzProgramProto,
// instantiating it into its IR version, and manually verifying the IR is
// correct.

namespace xls {
namespace {

TEST(IrFuzzBuilderTest, AddTwoLiterals) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Literal(UBits(10, 64)), m::Literal(UBits(20, 64))));
}

TEST(IrFuzzBuilderTest, AddTwoParams) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Add(m::Param("p0"), m::Param("p1")));
}

TEST(IrFuzzBuilderTest, AddLiteralsAndParamsAndAdds) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Literal(UBits(10, 64)), m::Param("p1")),
                     m::Add(m::Literal(UBits(20, 64)), m::Param("p4"))));
}

TEST(IrFuzzBuilderTest, SingleOpAddStack) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
  std::string proto_string = absl::StrFormat(
      R"(
        combine_stack_method: ADD_STACK_METHOD
        fuzz_ops {
          param {
            bit_width: 64
          }
        }
      )");
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Param("p0"));
}

TEST(IrFuzzBuilderTest, AddOpThenAddStack) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Literal(UBits(10, 64)), m::Param("p1")),
                     m::Add(m::Literal(UBits(10, 64)), m::Param("p1"))));
}

TEST(IrFuzzBuilderTest, AddOutOfBoundsIdxs) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Literal(UBits(10, 64)), m::Param("p1")));
}

TEST(IrFuzzBuilderTest, LiteralValueOverBoundsOfSmallWidth) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 1)));
}

TEST(IrFuzzBuilderTest, AddDifferentWidthsWithExtensions) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::ZeroExt(m::Param("p0")), m::SignExt(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, AddWithSliceAndExtension) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Add(m::ZeroExt(m::Param("p0")),
                                        m::BitSlice(m::Param("p1"), 0, 25)));
}

TEST(IrFuzzBuilderTest, AddStackWithDifferentWidths) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::ZeroExt(m::Add(m::BitSlice(m::Param("p0"), 0, 1),
                                       m::Param("p1"))),
                     m::Param("p2")));
}

TEST(IrFuzzBuilderTest, AddWithLargeWidths) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::ZeroExt(m::Param("p0")), m::ZeroExt(m::Param("p1"))));
}

TEST(IrFuzzBuilderTest, ConcatOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Param("p0"), m::Param("p1"), m::Param("p2")));
}

TEST(IrFuzzBuilderTest, EmptyConcat) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::ZeroExt(m::Concat(m::Literal(UBits(0, 64)))),
                     m::SignExt(m::Concat(m::Literal(UBits(0, 64))))));
}

TEST(IrFuzzBuilderTest, ShiftOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Shra(m::Param("p0"), m::Param("p1")),
                        m::Shrl(m::Param("p0"), m::Param("p1")),
                        m::Shll(m::Param("p0"), m::Param("p1"))));
}

TEST(IrFuzzBuilderTest, NaryOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Or(m::ZeroExt(m::Param("p0")), m::SignExt(m::Param("p0")),
                      m::BitSlice(m::Param("p1"), 0, 20)),
                m::Nor(m::Literal(UBits(0, 1))), m::Xor(m::Param("p0")),
                m::And(m::Param("p0"), m::Param("p0")),
                m::Nand(m::Param("p0"), m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, ReduceOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Concat(m::AndReduce(m::Param("p0")),
                                           m::OrReduce(m::Param("p0")),
                                           m::XorReduce(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, MulOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::UMul(m::Param("p0"), m::Param("p1")),
                        m::UMul(m::Param("p0"), m::Param("p1")),
                        m::SMul(m::Param("p0"), m::Param("p1"))));
}

TEST(IrFuzzBuilderTest, DivOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::UDiv(m::SignExt(m::Param("p0")),
                                m::BitSlice(m::Param("p1"), 0, 20)),
                        m::SDiv(m::Param("p0"), m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, ModOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::UMod(m::SignExt(m::Param("p0")),
                                m::BitSlice(m::Param("p1"), 0, 20)),
                        m::SMod(m::Param("p0"), m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, AssociativeOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Add(m::SignExt(m::Param("p0")),
                               m::BitSlice(m::Param("p1"), 0, 20)),
                        m::Sub(m::Param("p0"), m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, ComparisonOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::ULe(m::SignExt(m::Param("p0")),
                               m::BitSlice(m::Param("p1"), 0, 20)),
                        m::ULt(m::Param("p0"), m::Param("p0")),
                        m::UGe(m::Param("p0"), m::Param("p0")),
                        m::UGt(m::Param("p0"), m::Param("p0")),
                        m::SLe(m::Param("p0"), m::Param("p0")),
                        m::SLt(m::Param("p0"), m::Param("p0")),
                        m::SGe(m::Param("p0"), m::Param("p0")),
                        m::SGt(m::Param("p0"), m::Param("p0")),
                        m::Eq(m::Param("p0"), m::Param("p0")),
                        m::Ne(m::Param("p0"), m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, InvertOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Neg(m::Param("p0")), m::Not(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, SelectOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p0"), {m::SignExt(m::Param("p1")),
                                         m::BitSlice(m::Param("p2"), 0, 20)}));
}

TEST(IrFuzzBuilderTest, SelectWithLargeSelectorWidth) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p0"), {m::Param("p1")}, m::Param("p1")));
}

TEST(IrFuzzBuilderTest, SelectWithSmallSelectorWidth) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p0"), {m::Param("p1"), m::Param("p1")}));
}

TEST(IrFuzzBuilderTest, SelectWithUselessDefault) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p0"), {m::Param("p1"), m::Param("p1")}));
}

TEST(IrFuzzBuilderTest, SelectNeedingDefault) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Select(m::Param("p0"), {m::Param("p1")},
                                           m::ZeroExt(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, OneHotOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::OneHot(LsbOrMsb::kLsb), m::OneHot(LsbOrMsb::kMsb)));
}

TEST(IrFuzzBuilderTest, OneHotSelectOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::OneHotSelect(m::Param("p0"),
                      {m::ZeroExt(m::Param("p1")), m::SignExt(m::Param("p2")),
                       m::BitSlice(m::Param("p3"), 0, 40)}));
}

TEST(IrFuzzBuilderTest, OneHotSelectWithLargeSelectorWidth) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::OneHotSelect(m::Literal(0, 1), {m::Param("p1")}));
}

TEST(IrFuzzBuilderTest, OneHotSelectWithExtraCases) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::OneHotSelect(m::Param("p0"), {m::Param("p1"), m::Param("p1"),
                                               m::Param("p1")}));
}

TEST(IrFuzzBuilderTest, PrioritySelectOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Param("p0"),
                                {m::SignExt(m::Param("p1")),
                                 m::BitSlice(m::Param("p3"), 0, 20)},
                                m::Param("p2")));
}

TEST(IrFuzzBuilderTest, PrioritySelectWithLargeSelectorWidth) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Literal(0, 1), {m::Param("p1")}, m::Param("p1")));
}

TEST(IrFuzzBuilderTest, PrioritySelectWithExtraCases) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(m::Param("p0"),
                        {m::Param("p1"), m::Param("p1"), m::Param("p1")},
                        m::ZeroExt(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, CountOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(
          m::ZeroExt(
              m::Encode(m::OneHot(m::Reverse(m::Param("p0")), LsbOrMsb::kLsb))),
          m::ZeroExt(m::Encode(m::OneHot(m::Param("p0"), LsbOrMsb::kLsb)))));
}

TEST(IrFuzzBuilderTest, MatchOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(
          m::Concat(m::Eq(m::Param("p2"), m::Param("p2")),
                    m::Eq(m::Param("p2"), m::ZeroExt(m::Param("p0")))),
          {m::SignExt(m::Param("p1")), m::BitSlice(m::Param("p3"), 0, 40)},
          m::BitSlice(m::Param("p3"), 0, 40)));
}

TEST(IrFuzzBuilderTest, MatchTrueOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(m::Concat(m::BitSlice(m::Param("p2"), 0, 1),
                                          m::BitSlice(m::Param("p0"), 0, 1)),
                                {m::BitSlice(m::Param("p1"), 0, 1),
                                 m::BitSlice(m::Param("p3"), 0, 1)},
                                m::BitSlice(m::Param("p3"), 0, 1)));
}

TEST(IrFuzzBuilderTest, ReverseOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Reverse(m::Param("p0")));
}

TEST(IrFuzzBuilderTest, IdentityOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Identity(m::Param("p0")));
}

TEST(IrFuzzBuilderTest, ExtendOps) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Concat(m::SignExt(m::Param("p0")),
                                           m::ZeroExt(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, BitSliceOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Concat(m::BitSlice(m::Param("p0"), 0, 10),
                                           m::BitSlice(m::Param("p0"), 0, 10)));
}

TEST(IrFuzzBuilderTest, BitSliceUpdateOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(
      f->return_value(),
      m::BitSliceUpdate(m::Param("p0"), m::Param("p1"), m::Param("p2")));
}

TEST(IrFuzzBuilderTest, DynamicBitSliceOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::DynamicBitSlice(m::SignExt(m::Param("p0")),
                                                    m::Param("p1"), 30));
}

TEST(IrFuzzBuilderTest, EncodeOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Encode(m::Param("p0")));
}

TEST(IrFuzzBuilderTest, DecodeOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Decode(m::Param("p0")), m::Decode(m::Param("p0"))));
}

TEST(IrFuzzBuilderTest, GateOp) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto_string, &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Gate(m::BitSlice(m::Param("p0"), 0, 1), m::Param("p1")));
}

}  // namespace
}  // namespace xls
