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

#include <memory>
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
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
// instantiating into its IR version, and manually verifying the IR is correct.

namespace xls {
namespace {

TEST(IrFuzzBuilderTest, AddTwoLiterals) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal { 
            value: 10
          }
        }
        fuzz_ops {
          literal {
            value: 20
          }
        }
        fuzz_ops {
          add {
            lhs_ref {
              stack_idx: 0
            }
            rhs_ref {
              stack_idx: 1
            }
          }
        }
      )",
      &fuzz_program));
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(
      combine_stack_method: LAST_ELEMENT_METHOD
      fuzz_ops {
        param {}
      }
      fuzz_ops {
        param {}
      }
      fuzz_ops {
        add {
          lhs_ref {
            stack_idx: 0
          }
          rhs_ref {
            stack_idx: 1
          }
        }
      }
    )",
      &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Add(m::Param("p0"), m::Param("p1")));
}

TEST(IrFuzzBuilderTest, AddLiteralsAndParamsAndAdds) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            value: -10
          }
        }
        fuzz_ops {
          param {}
        }
        fuzz_ops {
          add {
            lhs_ref {
              stack_idx: 0
            }
            rhs_ref {
              stack_idx: 1
            }
          }
        }
        fuzz_ops {
          literal {
            value: -20
          }
        }
        fuzz_ops {
          param {}
        }
        fuzz_ops {
          add {
            lhs_ref {
              stack_idx: 3
            }
            rhs_ref {
              stack_idx: 4
            }
          }
        }
        fuzz_ops {
          add {
            lhs_ref {
              stack_idx: 2
            }
            rhs_ref {
              stack_idx: 5
            }
          }
        }
      )",
      &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Literal(UBits(-10, 64)), m::Param("p1")),
                     m::Add(m::Literal(UBits(-20, 64)), m::Param("p4"))));
}

TEST(IrFuzzBuilderTest, SingleOpAddStack) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(
        combine_stack_method: ADD_STACK_METHOD
        fuzz_ops {
          param {}
        }
      )",
      &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(), m::Param("p0"));
}

TEST(IrFuzzBuilderTest, AddOpThenAddStack) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  FuzzProgramProto fuzz_program;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(
        combine_stack_method: ADD_STACK_METHOD
        fuzz_ops {
          literal {
            value: 10
          }
        }
        fuzz_ops {
          param {}
        }
        fuzz_ops {
          add {
            lhs_ref {
              stack_idx: 0
            }
            rhs_ref {
              stack_idx: 1
            }
          }
        }
      )",
      &fuzz_program));
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
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"(
        combine_stack_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            value: 10
          }
        }
        fuzz_ops {
          param {}
        }
        fuzz_ops {
          add {
            lhs_ref {
              stack_idx: 2
            }
            rhs_ref {
              stack_idx: -1
            }
          }
        }
      )",
      &fuzz_program));
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ir));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Literal(UBits(10, 64)), m::Param("p1")));
}

}  // namespace
}  // namespace xls
