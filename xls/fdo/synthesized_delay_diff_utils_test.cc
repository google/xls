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

#include "xls/fdo/synthesized_delay_diff_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/analyze_critical_path.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace synthesis {
namespace {

using ::testing::IsEmpty;

class FakeSynthesizer : public Synthesizer {
 public:
  FakeSynthesizer() : Synthesizer("FakeSynthesizer") {}

  absl::StatusOr<int64_t> SynthesizeVerilogAndGetDelay(
      std::string_view verilog_text,
      std::string_view top_module_name) const override {
    return base_delay_ * verilog_text.size();
  }

  absl::StatusOr<int64_t> SynthesizeNodesAndGetDelay(
      const absl::flat_hash_set<Node*>& nodes) const override {
    return base_delay_ * nodes.size();
  }

  absl::StatusOr<int64_t> SynthesizeFunctionBaseAndGetDelay(
      FunctionBase* f) const override {
    return base_delay_ * f->node_count();
  }

  void SetBaseDelay(int64_t delay) { base_delay_ = delay; }

 private:
  int64_t base_delay_ = 0;
};

class SynthesizedDelayDiffUtilsTest : public IrTestBase {
 public:
  void SetUp() override {
    package_ = CreatePackage();
    XLS_ASSERT_OK_AND_ASSIGN(f_, ParseFunction(R"(
fn foobar(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
  and.1: bits[8] = and(x, y)
  or.2: bits[8] = or(and.1, y)
  add.3: bits[8] = add(or.2, and.1)
  ret add.4: bits[8] = add(add.3, z)
}
)",
                                               package_.get()));
    XLS_ASSERT_OK_AND_ASSIGN(
        schedule_, RunPipelineSchedule(f_, delay_estimator_,
                                       SchedulingOptions().pipeline_stages(3)));
  }

  std::unique_ptr<Package> package_;
  FunctionBase* f_;
  std::optional<PipelineSchedule> schedule_;
  TestDelayEstimator delay_estimator_;
  FakeSynthesizer synthesizer_;
};

TEST_F(SynthesizedDelayDiffUtilsTest, SynthesizeAndGetDelayDiff) {
  synthesizer_.SetBaseDelay(42);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> critical_path,
      AnalyzeCriticalPath(f_, /*clock_period_ps=*/std::nullopt,
                          delay_estimator_));
  XLS_ASSERT_OK_AND_ASSIGN(
      SynthesizedDelayDiff diff,
      SynthesizeAndGetDelayDiff(f_, critical_path, &synthesizer_));
  EXPECT_EQ(diff.synthesized_delay_ps, 294);
  EXPECT_EQ(diff.xls_delay_ps, 4);
  EXPECT_EQ(diff.critical_path.size(), critical_path.size());
}

TEST_F(SynthesizedDelayDiffUtilsTest, CreateDelayDiffByStage) {
  synthesizer_.SetBaseDelay(42);
  XLS_ASSERT_OK_AND_ASSIGN(
      SynthesizedDelayDiffByStage diff,
      CreateDelayDiffByStage(f_, *schedule_, delay_estimator_, &synthesizer_));
  EXPECT_EQ(diff.total_diff.synthesized_delay_ps, 504);
  EXPECT_EQ(diff.total_diff.xls_delay_ps, 5);
  EXPECT_THAT(diff.total_diff.critical_path, IsEmpty());
  EXPECT_EQ(diff.total_stage_percent_diff_abs, 20.0);
  EXPECT_EQ(diff.max_stage_percent_diff_abs, 10.0);
  ASSERT_EQ(diff.stage_diffs.size(), 3);
  EXPECT_EQ(diff.stage_diffs[0].critical_path.size(), 4);
  EXPECT_EQ(diff.stage_diffs[0].synthesized_delay_ps, 252);
  EXPECT_EQ(diff.stage_diffs[0].xls_delay_ps, 3);
  EXPECT_EQ(diff.stage_percent_diffs[0].xls_percent, 60.0);
  EXPECT_EQ(diff.stage_percent_diffs[0].synthesized_percent, 50.0);
  EXPECT_EQ(diff.stage_diffs[1].critical_path.size(), 2);
  EXPECT_EQ(diff.stage_diffs[1].synthesized_delay_ps, 126);
  EXPECT_EQ(diff.stage_diffs[1].xls_delay_ps, 1);
  EXPECT_EQ(diff.stage_percent_diffs[1].xls_percent, 20.0);
  EXPECT_EQ(diff.stage_percent_diffs[1].synthesized_percent, 25.0);
  EXPECT_EQ(diff.stage_diffs[2].critical_path.size(), 2);
  EXPECT_EQ(diff.stage_diffs[2].synthesized_delay_ps, 126);
  EXPECT_EQ(diff.stage_diffs[2].xls_delay_ps, 1);
  EXPECT_EQ(diff.stage_percent_diffs[2].xls_percent, 20.0);
  EXPECT_EQ(diff.stage_percent_diffs[2].synthesized_percent, 25.0);
}

TEST_F(SynthesizedDelayDiffUtilsTest, CreateDelayDiffForSpecifiedStage) {
  synthesizer_.SetBaseDelay(42);
  XLS_ASSERT_OK_AND_ASSIGN(
      SynthesizedDelayDiff diff,
      CreateDelayDiffForStage(f_, *schedule_, delay_estimator_, &synthesizer_,
                              1));
  EXPECT_EQ(diff.critical_path.size(), 2);
  EXPECT_EQ(diff.synthesized_delay_ps, 126);
  EXPECT_EQ(diff.xls_delay_ps, 1);
}

TEST_F(SynthesizedDelayDiffUtilsTest, CreateDelayDiffByStageWithoutSynthesis) {
  XLS_ASSERT_OK_AND_ASSIGN(
      SynthesizedDelayDiffByStage diff,
      CreateDelayDiffByStage(f_, *schedule_, delay_estimator_,
                             /*synthesizer=*/nullptr));
  EXPECT_EQ(diff.total_diff.synthesized_delay_ps, 0);
  EXPECT_EQ(diff.total_diff.xls_delay_ps, 5);
  EXPECT_THAT(diff.total_diff.critical_path, IsEmpty());
  EXPECT_EQ(diff.total_stage_percent_diff_abs, 100.0);
  EXPECT_EQ(diff.max_stage_percent_diff_abs, 60.0);
  ASSERT_EQ(diff.stage_diffs.size(), 3);
  EXPECT_EQ(diff.stage_diffs[0].critical_path.size(), 4);
  EXPECT_EQ(diff.stage_diffs[0].synthesized_delay_ps, 0);
  EXPECT_EQ(diff.stage_diffs[0].xls_delay_ps, 3);
  EXPECT_EQ(diff.stage_percent_diffs[0].xls_percent, 60.0);
  EXPECT_EQ(diff.stage_percent_diffs[0].synthesized_percent, 0);
  EXPECT_EQ(diff.stage_diffs[1].critical_path.size(), 2);
  EXPECT_EQ(diff.stage_diffs[1].synthesized_delay_ps, 0);
  EXPECT_EQ(diff.stage_diffs[1].xls_delay_ps, 1);
  EXPECT_EQ(diff.stage_percent_diffs[1].xls_percent, 20.0);
  EXPECT_EQ(diff.stage_percent_diffs[1].synthesized_percent, 0);
  EXPECT_EQ(diff.stage_diffs[2].critical_path.size(), 2);
  EXPECT_EQ(diff.stage_diffs[2].synthesized_delay_ps, 0);
  EXPECT_EQ(diff.stage_diffs[2].xls_delay_ps, 1);
  EXPECT_EQ(diff.stage_percent_diffs[2].xls_percent, 20.0);
  EXPECT_EQ(diff.stage_percent_diffs[2].synthesized_percent, 0);
}

TEST_F(SynthesizedDelayDiffUtilsTest, SynthesizedDelayDiffToString) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> critical_path,
      AnalyzeCriticalPath(f_, /*clock_period_ps=*/std::nullopt,
                          delay_estimator_));
  SynthesizedDelayDiff diff{.critical_path = critical_path,
                            .xls_delay_ps = 65,
                            .synthesized_delay_ps = 31};
  EXPECT_EQ(SynthesizedDelayDiffToString(diff),
            "     31ps as synthesized (-34ps); 47.69%\n" +
                CriticalPathToString(critical_path));
}

TEST_F(SynthesizedDelayDiffUtilsTest, SynthesizedStageDelayDiffToString) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<CriticalPathEntry> critical_path,
      AnalyzeCriticalPath(f_, /*clock_period_ps=*/std::nullopt,
                          delay_estimator_));
  SynthesizedDelayDiff diff{.critical_path = critical_path,
                            .xls_delay_ps = 65,
                            .synthesized_delay_ps = 31};
  // To keep this simple, just pretend `f_` is one stage of a fictitious
  // containing pipeline.
  StagePercentDiff percent_diff{.xls_percent = 59.09,
                                .synthesized_percent = 77.50};
  EXPECT_EQ(SynthesizedStageDelayDiffToString(diff, percent_diff),
            "     31ps as synthesized (-34ps); 77.50% of synthesized pipeline "
            "vs. 59.09% according to XLS.\n" +
                CriticalPathToString(critical_path));
}

}  // namespace
}  // namespace synthesis
}  // namespace xls
