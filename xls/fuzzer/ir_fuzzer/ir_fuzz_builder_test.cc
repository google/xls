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

}  // namespace
}  // namespace xls
