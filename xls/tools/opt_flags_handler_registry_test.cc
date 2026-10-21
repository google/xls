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

#include "xls/tools/opt_flags_handler_registry.h"

#include <memory>
#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/tools/opt.h"
#include "xls/tools/opt_flags.h"
#include "xls/tools/opt_flags.pb.h"
#include "xls/tools/test_utils/test_opt_flag_handler.h"
#include "xls/tools/test_utils/test_opt_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;

class CaptureOptionsPass : public OptimizationPass {
 public:
  explicit CaptureOptionsPass(std::optional<std::string>* captured_value)
      : OptimizationPass("capture_options", "Capture Options"),
        captured_value_(captured_value) {}

 protected:
  absl::StatusOr<bool> RunInternal(
      Package* ir, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext& context) const override {
    if (auto ext = options.GetExtension<TestOptOptionExtension>();
        ext.has_value()) {
      *captured_value_ = (*ext)->value;
    }
    return false;
  }

 private:
  std::optional<std::string>* captured_value_;
};

class CaptureOptionsDecorator : public tools::PassRegistryDecorator {
 public:
  explicit CaptureOptionsDecorator(std::optional<std::string>* captured_value)
      : captured_value_(captured_value) {}

  absl::StatusOr<std::unique_ptr<OptimizationPassRegistry>> Decorate(
      const OptimizationPassRegistry& registry) override {
    auto decorated =
        std::make_unique<OptimizationPassRegistry>(registry.OverridableClone());
    XLS_RETURN_IF_ERROR(decorated->Register(
        "capture_options",
        optimization_registry::internal::Pass<CaptureOptionsPass>(
            *decorated, captured_value_)));
    return decorated;
  }

 private:
  std::optional<std::string>* captured_value_;
};

TEST(OptFlagsHandlerRegistryTest, DefaultFlagValue) {
  OptFlagsProto proto;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::ParseFlags(proto));
  EXPECT_FALSE(proto.HasExtension(test_opt_flag));

  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));
  EXPECT_EQ(options.GetExtension<TestOptOptionExtension>(), std::nullopt);
}

TEST(OptFlagsHandlerRegistryTest, CustomFlagValue) {
  std::string original = absl::GetFlag(FLAGS_test_opt_flag);
  absl::SetFlag(&FLAGS_test_opt_flag, "custom_value");
  absl::Cleanup cleanup = [&]() {
    absl::SetFlag(&FLAGS_test_opt_flag, original);
  };

  OptFlagsProto proto;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::ParseFlags(proto));
  EXPECT_TRUE(proto.HasExtension(test_opt_flag));
  EXPECT_EQ(proto.GetExtension(test_opt_flag), "custom_value");

  OptimizationPassOptions options;
  XLS_ASSERT_OK(OptFlagsHandlerRegistry::Process(proto, options));
  std::optional<const TestOptOptionExtension*> ext =
      options.GetExtension<TestOptOptionExtension>();
  ASSERT_TRUE(ext.has_value());
  EXPECT_EQ((*ext)->value, "custom_value");
}

TEST(OptFlagsHandlerRegistryTest, EndToEndViaOptFlagsAndOptions) {
  std::string original = absl::GetFlag(FLAGS_test_opt_flag);
  absl::SetFlag(&FLAGS_test_opt_flag, "end_to_end_value");
  absl::Cleanup cleanup = [&]() {
    absl::SetFlag(&FLAGS_test_opt_flag, original);
  };

  XLS_ASSERT_OK_AND_ASSIGN(OptFlagsProto proto, GetOptFlags(std::nullopt));
  proto.set_area_model("");
  proto.mutable_pipeline()->mutable_top()->set_pass_name("capture_options");

  Package package("test_package");
  FunctionBuilder fb("main", &package);
  fb.Literal(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK(package.SetTop(f));

  std::optional<std::string> captured_value;
  CaptureOptionsDecorator decorator(&captured_value);
  XLS_ASSERT_OK(tools::OptimizeIrForTop(
      &package, proto, /*metadata=*/nullptr,
      tools::OptModifiers{.registry_decorator = &decorator}));
  EXPECT_EQ(captured_value, "end_to_end_value");
}

TEST(OptFlagsHandlerRegistryTest, ParseFlagsError) {
  std::string original = absl::GetFlag(FLAGS_test_opt_flag);
  absl::SetFlag(&FLAGS_test_opt_flag, "parse_error");
  absl::Cleanup cleanup = [&]() {
    absl::SetFlag(&FLAGS_test_opt_flag, original);
  };

  OptFlagsProto proto;
  EXPECT_THAT(OptFlagsHandlerRegistry::ParseFlags(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OptFlagsHandlerRegistryTest, ProcessError) {
  OptFlagsProto proto;
  proto.SetExtension(test_opt_flag, "invalid");

  OptimizationPassOptions options;
  EXPECT_THAT(OptFlagsHandlerRegistry::Process(proto, options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xls
