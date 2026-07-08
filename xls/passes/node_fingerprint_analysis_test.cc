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

#include "xls/passes/node_fingerprint_analysis.h"

#include <cstdint>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

class NodeFingerprintAnalysisTest : public IrTestBase {};

TEST_F(NodeFingerprintAnalysisTest, SimpleFingerprints) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto add1 = fb.Add(x, y);
  auto add2 = fb.Add(x, y);
  auto sub = fb.Subtract(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  NodeFingerprintAnalysis analysis;
  XLS_ASSERT_OK(analysis.Attach(f).status());

  uint64_t fp_x = analysis.GetFingerprint(x.node());
  uint64_t fp_y = analysis.GetFingerprint(y.node());
  uint64_t fp_add1 = analysis.GetFingerprint(add1.node());
  uint64_t fp_add2 = analysis.GetFingerprint(add2.node());
  uint64_t fp_sub = analysis.GetFingerprint(sub.node());

  EXPECT_NE(fp_x, fp_y);
  EXPECT_EQ(fp_add1, fp_add2);
  EXPECT_NE(fp_add1, fp_sub);
}

TEST_F(NodeFingerprintAnalysisTest, IdenticalTreesDifferentNames) {
  auto p = CreatePackage();

  auto build_func = [&](std::string name) -> absl::StatusOr<Function*> {
    FunctionBuilder fb(name, p.get());
    auto x = fb.Param("x", p->GetBitsType(32));
    fb.Add(x, fb.Literal(Value(UBits(1, 32))));
    return fb.Build();
  };

  XLS_ASSERT_OK_AND_ASSIGN(auto f1, build_func("f1"));
  XLS_ASSERT_OK_AND_ASSIGN(auto f2, build_func("f2"));

  NodeFingerprintAnalysis analysis;
  XLS_ASSERT_OK(analysis.Attach(f1).status());
  uint64_t fp1 = analysis.GetFingerprint(f1->return_value());

  XLS_ASSERT_OK(analysis.Attach(f2).status());
  uint64_t fp2 = analysis.GetFingerprint(f2->return_value());

  EXPECT_EQ(fp1, fp2);
}

TEST_F(NodeFingerprintAnalysisTest, ParametersTrackedByPosition) {
  auto p = CreatePackage();

  FunctionBuilder fb1("f1", p.get());
  auto x1 = fb1.Param("x", p->GetBitsType(32));
  auto y1 = fb1.Param("y", p->GetBitsType(32));
  fb1.Add(x1, y1);
  XLS_ASSERT_OK_AND_ASSIGN(auto f1, fb1.Build());

  FunctionBuilder fb2("f2", p.get());
  auto y2 = fb2.Param("y", p->GetBitsType(32));
  auto x2 = fb2.Param("x", p->GetBitsType(32));
  fb2.Add(y2, x2);
  XLS_ASSERT_OK_AND_ASSIGN(auto f2, fb2.Build());

  NodeFingerprintAnalysis analysis;
  XLS_ASSERT_OK(analysis.Attach(f1).status());
  uint64_t fp1 = analysis.GetFingerprint(f1->return_value());

  XLS_ASSERT_OK(analysis.Attach(f2).status());
  uint64_t fp2 = analysis.GetFingerprint(f2->return_value());

  // These should be equal because (Param0 + Param1) is structurally same in
  // both functions.
  EXPECT_EQ(fp1, fp2);
}

}  // namespace
}  // namespace xls
