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

#include "xls/estimators/delay_model/sample_point_extraction_utils.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "google/protobuf/text_format.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_model.pb.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"

namespace xls::delay_model {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::xls::proto_testing::EqualsProto;

OpModels CreateFakeOpModels() {
  OpModels op_models;
  QCHECK(google::protobuf::TextFormat::ParseFromString(R"pb(
                                               op_models {
                                                 op: "kSub"
                                                 estimator { regression {} }
                                               }
                                               op_models {
                                                 op: "kShll"
                                                 estimator { regression {} }
                                               }
                                               op_models {
                                                 op: "kShra"
                                                 estimator { alias_op: "kShll" }
                                               }
                                               op_models {
                                                 op: "kAnd"
                                                 estimator { logical_effort {} }
                                               }
                                               op_models {
                                                 op: "kSignExt"
                                                 estimator { regression {} }
                                               }
                                               op_models {
                                                 op: "kArrayIndex"
                                                 estimator { regression {} }
                                               }
                                             )pb",
                                             &op_models));
  return op_models;
}

template <typename... T>
Parameterization CreateParams(int result_width, T... operand_widths) {
  Parameterization params;
  params.set_result_width(result_width);
  (params.add_operand_widths(operand_widths), ...);
  return params;
}

class FakeDelayEstimator : public DelayEstimator {
 public:
  FakeDelayEstimator() : DelayEstimator("fake") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kSub:
        return node->operand(0)->BitCountOrDie() > 16 ? 345 : 123;
      case Op::kShra:
        return 56;
      default:
        return 20;
    }
  }

  absl::StatusOr<int64_t> GetLogicalEffortDelayInPs(Node* node,
                                                    int64_t tau_in_ps) {
    return tau_in_ps;
  }
};

class SamplePointExtractionUtilsTest : public IrTestBase {
 protected:
  FakeDelayEstimator delay_estimator_;
  OpModels op_models_ = CreateFakeOpModels();
};

constexpr char kBasicPackageCode[] = R"(
package foo

fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=3, id=1)
  ret sub.2: bits[32] = sub(literal.1, literal.1, id=2)
}

fn g() -> bits[32] {
  literal.3: bits[32] = literal(value=4, id=3)
  sub.4: bits[32] = sub(literal.3, literal.3, id=4)
  literal.5: bits[4] = literal(value=5, id=5)
  literal.6: bits[4] = literal(value=1, id=6)
  sub.7: bits[4] = sub(literal.5, literal.6, id=7)
  ret shra.8: bits[32] = shra(sub.4, sub.7, id=8)
}

)";

constexpr char kPackageCodeWithArrayIndex[] = R"(
package foo

fn f(x: bits[32][2]) -> bits[32] {
  literal.1: bits[32] = literal(value=1, id=1)
  ret array_index.2: bits[32] = array_index(x, indices=[literal.1], id=2)
}
)";

TEST_F(SamplePointExtractionUtilsTest, ExtractSamplePointsFromEmptyPackage) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage("package foo"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SamplePoint> points,
                           ExtractSamplePoints(*package, op_models_));
  EXPECT_THAT(points, IsEmpty());
}

TEST_F(SamplePointExtractionUtilsTest, ExtractSamplePointsFromBasicPackage) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kBasicPackageCode));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SamplePoint> points,
                           ExtractSamplePoints(*package, op_models_));
  EXPECT_THAT(
      points,
      ElementsAre(
          FieldsAre("kShll", EqualsProto(CreateParams(32, 32, 4)), 1, 0),
          FieldsAre("kSub", EqualsProto(CreateParams(4, 4, 4)), 1, 0),
          FieldsAre("kSub", EqualsProto(CreateParams(32, 32, 32)), 2, 0)));
}

TEST_F(SamplePointExtractionUtilsTest,
       ExtractSamplePointsFromBasicPackageWithDelayEstimates) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kBasicPackageCode));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<SamplePoint> points,
      ExtractSamplePoints(*package, op_models_, &delay_estimator_));
  EXPECT_THAT(
      points,
      ElementsAre(
          FieldsAre("kShll", EqualsProto(CreateParams(32, 32, 4)), 1, 56),
          FieldsAre("kSub", EqualsProto(CreateParams(4, 4, 4)), 1, 123),
          FieldsAre("kSub", EqualsProto(CreateParams(32, 32, 32)), 2, 345)));
}

TEST_F(SamplePointExtractionUtilsTest, ExtractSamplePointsWithArrayIndex) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package,
                           ParsePackage(kPackageCodeWithArrayIndex));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SamplePoint> points,
                           ExtractSamplePoints(*package, op_models_));
  EXPECT_THAT(points, ElementsAre(FieldsAre("kArrayIndex", EqualsProto(R"pb(
                                              result_width: 32
                                              operand_widths: 32
                                              operand_widths: 32
                                              operand_element_counts {
                                                element_counts: 2
                                                operand_number: 0
                                              }
                                            )pb"),
                                            1, 0)));
}

TEST_F(SamplePointExtractionUtilsTest, ConvertToListWithOpAttributes) {
  std::vector<SamplePoint> points = {
      SamplePoint{.op_name = "kSignExt", .params = CreateParams(64, 32)}};
  EXPECT_THAT(ConvertToOpSamplesList(points), EqualsProto(R"pb(
                op_samples {
                  op: "kIdentity"
                  samples { result_width: 1 operand_widths: 1 }
                }
                op_samples {
                  op: "kSignExt"
                  attributes: "new_bit_count=%r"
                  samples: { result_width: 64 operand_widths: 32 }
                }
              )pb"));
}

TEST_F(SamplePointExtractionUtilsTest, ConvertAllToOpSampleList) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kBasicPackageCode));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SamplePoint> points,
                           ExtractSamplePoints(*package, op_models_));
  EXPECT_THAT(
      ConvertToOpSamplesList(points), EqualsProto(R"pb(
        op_samples {
          op: "kIdentity"
          samples { result_width: 1 operand_widths: 1 }
        }
        op_samples {
          op: "kShll"
          samples { result_width: 32 operand_widths: 32 operand_widths: 4 }
        }
        op_samples {
          op: "kSub"
          samples: { result_width: 4 operand_widths: 4 operand_widths: 4 }
          samples: { result_width: 32 operand_widths: 32 operand_widths: 32 }
        }
      )pb"));
}

TEST_F(SamplePointExtractionUtilsTest, ConvertNToOpSampleList) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kBasicPackageCode));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SamplePoint> points,
                           ExtractSamplePoints(*package, op_models_));
  // Reorder so the most frequent ones per op come first.
  absl::c_sort(points, [](const SamplePoint& x, const SamplePoint& y) {
    if (x.op_name != y.op_name) {
      return x.op_name < y.op_name;
    }
    return x.frequency > y.frequency;
  });
  // Drop the lower frequency subtraction.
  EXPECT_THAT(
      ConvertToOpSamplesList(points, 2), EqualsProto(R"pb(
        op_samples {
          op: "kIdentity"
          samples { result_width: 1 operand_widths: 1 }
        }
        op_samples {
          op: "kShll"
          samples { result_width: 32 operand_widths: 32 operand_widths: 4 }
        }
        op_samples {
          op: "kSub"
          samples: { result_width: 32 operand_widths: 32 operand_widths: 32 }
        }
      )pb"));
}

TEST_F(SamplePointExtractionUtilsTest, ConvertLessThanNSamplesToListOfN) {
  XLS_ASSERT_OK_AND_ASSIGN(auto package, ParsePackage(kBasicPackageCode));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SamplePoint> points,
                           ExtractSamplePoints(*package, op_models_));
  OpSamplesList without_n = ConvertToOpSamplesList(points);
  OpSamplesList with_n = ConvertToOpSamplesList(points, 100);
  EXPECT_THAT(with_n, EqualsProto(without_n));
}

}  // namespace
}  // namespace xls::delay_model
