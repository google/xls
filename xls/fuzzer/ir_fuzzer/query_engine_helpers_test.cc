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

#include "xls/fuzzer/ir_fuzzer/query_engine_helpers.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {

class QueryEngineHelpersTest : public IrTestBase {
 public:
  FuzzPackageWithArgs MakeTwoArgPackage(int64_t arg_width, int64_t num_samples,
                                        int64_t result_width) {
    auto p = std::make_unique<Package>(TestName());
    FunctionBuilder fb(TestName(), p.get());
    auto x = fb.Param("x", p->GetBitsType(arg_width));
    auto y = fb.Param("y", p->GetBitsType(arg_width));
    fb.Add(x, y);
    XLS_EXPECT_OK(fb.Build().status());
    std::vector<std::vector<Value>> args;
    for (int64_t i = 0; i < num_samples; ++i) {
      args.push_back({
          Value(UBits(i, arg_width)),
          Value(UBits(i + 1, arg_width)),
      });
    }
    return FuzzPackageWithArgs{
        .fuzz_package =
            FuzzPackage{
                .p = std::move(p),
            },
        .arg_sets = args,
    };
  }
};

TEST_F(QueryEngineHelpersTest, CallsEach) {
  int64_t cnt = 0;
  auto pkg = MakeTwoArgPackage(/*arg_width=*/8,
                               /*num_samples=*/4,
                               /*result_width=*/8);
  CheckQueryEngineConsistency<StatelessQueryEngine>(
      pkg, [&](const StatelessQueryEngine& qe, Node* n, const Value& v) {
        cnt++;
        return true;
      });
  EXPECT_EQ(cnt, pkg.fuzz_package.p->functions().front()->node_count() *
                     pkg.arg_sets.size());
}

TEST_F(QueryEngineHelpersTest, DedupsCalls) {
  int64_t cnt = 0;
  StatelessQueryEngine qe;
  auto pkg = MakeTwoArgPackage(/*arg_width=*/8,
                               /*num_samples=*/4,
                               /*result_width=*/8);
  int64_t uniq_args = pkg.arg_sets.size();
  for (int64_t i = 0; i < 10; ++i) {
    pkg.arg_sets.push_back(pkg.arg_sets[i]);
  }
  CheckQueryEngineInstanceConsistency<StatelessQueryEngine>(
      pkg, qe, [&](const StatelessQueryEngine& qe, Node* n, const Value& v) {
        ++cnt;
        return true;
      });
  EXPECT_EQ(cnt,
            pkg.fuzz_package.p->functions().front()->node_count() * uniq_args);
}

TEST_F(QueryEngineHelpersTest, FailureDetected) {
  TernaryQueryEngine qe;
  // We should fail here since the checker always returns false.
  EXPECT_DEATH(
      {
        CheckQueryEngineInstanceConsistency(
            MakeTwoArgPackage(/*arg_width=*/8,
                              /*num_samples=*/4,
                              /*result_width=*/8),
            qe, [](const TernaryQueryEngine& qe, Node* n, const Value& v) {
              return false;
            });
        CHECK(!Test::HasFailure()) << "Check failed";
      },
      "Check failed");
}

}  // namespace
}  // namespace xls
