// Copyright 2024 The XLS Authors
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

#include "xls/passes/merge_selector_pass.h"

#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

namespace m = ::xls::op_matchers;

using ::absl_testing::IsOkAndHolds;

class MergeSelectorPassTest : public IrTestBase {};

TEST_F(MergeSelectorPassTest, NoPossibleMerges_DoesNotUpdateFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(Parser::ParseFunction(R"(
     fn simple_select(x: bits[1]) -> bits[1] {
        literal.1: bits[1] = literal(value=0)
        literal.2: bits[1] = literal(value=1)
        ret result: bits[1] = sel(x, cases=[literal.1, literal.2])
     }
  )",
                                      p.get()));
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(MergeSelectorPass().Run(p.get(), OptimizationPassOptions(),
                                      &results, context),
              IsOkAndHolds(false));
}

TEST_F(MergeSelectorPassTest, ChildNarrowerThanParent_DoesNotUpdateFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK(Parser::ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[5] {
        literal.1: bits[2] = literal(value=0)
        literal.2: bits[2] = literal(value=1)
        literal.3: bits[2] = literal(value=2)
        literal.4: bits[2] = literal(value=3)
        literal.5: bits[2] = literal(value=0)
        literal.6: bits[5] = literal(value=10)
        literal.7: bits[5] = literal(value=11)
        literal.8: bits[5] = literal(value=12)
        sel.9: bits[2] = sel(x, cases=[literal.1, literal.2, literal.3, literal.4], default=literal.5)
        ret result: bits[5] = sel(sel.9, cases=[literal.6, literal.7], default=literal.8)
     }
  )",
                                      p.get()));
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(MergeSelectorPass().Run(p.get(), OptimizationPassOptions(),
                                      &results, context),
              IsOkAndHolds(false));
}

TEST_F(MergeSelectorPassTest, SimpleMerge_UpdatesFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(R"(
     fn simple_select(x: bits[2]) -> bits[3] {
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        literal.4: bits[3] = literal(value=4)
        literal.5: bits[3] = literal(value=5)
        literal.6: bits[3] = literal(value=6)
        sel.7: bits[3] = sel(x, cases=[literal.1, literal.2, literal.1, literal.2])
        ret result: bits[3] = sel(sel.7, cases=[literal.1, literal.2, literal.3, literal.4, literal.5], default=literal.6)
     }
  )",
                                                               p.get()));
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(MergeSelectorPass().Run(p.get(), OptimizationPassOptions(),
                                      &results, context),
              IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("x"), {m::Literal(2), m::Literal(3),
                                        m::Literal(2), m::Literal(3)}));
}

TEST_F(MergeSelectorPassTest, SameWidthMerge_UpdatesFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(R"(
     fn simple_select(x: bits[3]) -> bits[3] {
        literal.1: bits[3] = literal(value=1)
        literal.2: bits[3] = literal(value=2)
        literal.3: bits[3] = literal(value=3)
        literal.4: bits[3] = literal(value=4)
        literal.5: bits[3] = literal(value=5)
        parent_selector: bits[3] = add(x, x)
        parent_select: bits[3] = sel(parent_selector, cases=[literal.1, literal.2,
          literal.3, literal.4], default=literal.1)
        ret result: bits[3] = sel(parent_select, cases=[literal.1, literal.2,
          literal.3, literal.4], default=literal.5)
     }
  )",
                                                               p.get()));
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(MergeSelectorPass().Run(p.get(), OptimizationPassOptions(),
                                      &results, context),
              IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Select(m::Add(m::Param("x"), m::Param("x")),
                                           {m::Literal(2), m::Literal(3),
                                            m::Literal(4), m::Literal(5)},
                                           m::Literal(2)));
}

void IrFuzzMergeSelectorConversion(FuzzPackageWithArgs fuzz_package_with_args) {
  MergeSelectorPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzMergeSelectorConversion)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
