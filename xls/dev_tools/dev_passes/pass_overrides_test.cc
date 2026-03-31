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

#include "xls/dev_tools/dev_passes/pass_overrides.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/tools/opt.h"

namespace m = ::xls::op_matchers;

namespace xls::tools {
namespace {

class OptimizationPassOverridesTest : public IrTestBase {};
class TestPass : public OptimizationPass {
 public:
  explicit TestPass(std::string_view name) : OptimizationPass(name, name) {}
  absl::StatusOr<bool> RunInternal(
      Package* ir, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext& context) const override {
    return false;
  }
};

class TestOverrides : public OptimizationPassOverrides {
 public:
  explicit TestOverrides(std::string_view override_name)
      : override_name_(override_name) {}

  absl::StatusOr<std::unique_ptr<OptimizationPass>> OverridePass(
      const OptimizationPassGenerator& generator,
      const OptimizationPassRegistry& base_registry) override {
    override_called_ = true;
    return std::make_unique<TestPass>(override_name_);
  }

  bool override_called() const { return override_called_; }

 private:
  std::string override_name_;
  bool override_called_ = false;
};

TEST_F(OptimizationPassOverridesTest, OverridePass) {
  OptimizationPassRegistry registry;
  XLS_ASSERT_OK(registry.Register(
      "test_pass",
      optimization_registry::internal::Pass<TestPass>(registry, "test_pass")));

  TestOverrides overrides("overridden_pass");
  XLS_ASSERT_OK_AND_ASSIGN(auto decorated,
                           overrides.decorator().Decorate(registry));

  XLS_ASSERT_OK_AND_ASSIGN(auto* generator, decorated->Generator("test_pass"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OptimizationPass> pass,
                           generator->Generate());

  EXPECT_TRUE(overrides.override_called());
  EXPECT_EQ(pass->short_name(), "overridden_pass");
}

class DoubleRet : public OptimizationFunctionBasePass {
 public:
  explicit DoubleRet()
      : OptimizationFunctionBasePass("double_ret", "double_ret") {}

  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    Literal* ret = f->AsFunctionOrDie()->return_value()->As<Literal>();
    XLS_RETURN_IF_ERROR(
        ret->ReplaceUsesWithNew<Literal>(
               Value(bits_ops::UMul(ret->value().bits(),
                                    UBits(2, ret->value().bits().bit_count()))
                         .Slice(0, 32)))
            .status());
    return true;
  }
};
class IncrRet : public OptimizationFunctionBasePass {
 public:
  explicit IncrRet() : OptimizationFunctionBasePass("incr_ret", "incr_ret") {}

  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    Literal* ret = f->AsFunctionOrDie()->return_value()->As<Literal>();
    XLS_RETURN_IF_ERROR(ret->ReplaceUsesWithNew<Literal>(
                               Value(bits_ops::Increment(ret->value().bits())))
                            .status());
    return true;
  }
};

class TripleOverrides : public OptimizationPassOverrides {
 public:
  absl::StatusOr<std::unique_ptr<OptimizationPass>> OverridePass(
      const OptimizationPassGenerator& generator,
      const OptimizationPassRegistry& base_registry) override {
    auto compound = std::make_unique<OptimizationCompoundPass>(
        "triple_override", "triple_override");
    for (int64_t i = 0; i < 3; ++i) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> res,
                           generator.Generate());
      compound->AddOwned(std::move(res));
    }
    return compound;
  }
};

TEST_F(OptimizationPassOverridesTest, ReplaceWithMultiple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 32));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  OptimizationPassRegistry registry;
  XLS_ASSERT_OK(registry.Register(
      "incr", optimization_registry::internal::Pass<IncrRet>(registry)));
  XLS_ASSERT_OK(registry.Register(
      "double", optimization_registry::internal::Pass<DoubleRet>(registry)));

  TripleOverrides overrides;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OptimizationPassRegistry> decorated,
                           overrides.decorator().Decorate(registry));
  XLS_ASSERT_OK_AND_ASSIGN(auto pipeline,
                           GetOptimizationPipelineGenerator(*decorated)
                               .GeneratePipeline("incr double"));
  PassResults results;
  OptimizationContext context;
  XLS_ASSERT_OK(
      pipeline->Run(p.get(), OptimizationPassOptions(), &results, context));

  auto do_incr = [](Bits b) -> Bits { return bits_ops::Increment(b); };
  auto do_double = [](Bits b) -> Bits {
    return bits_ops::UMul(b, UBits(2, b.bit_count())).Slice(0, 32);
  };
  auto do_three_times = [&](Bits b) -> Bits {
    return do_double(do_double(do_double(do_incr(do_incr(do_incr(b))))));
  };
  RecordProperty("result", f->return_value()->ToString());
  EXPECT_THAT(f->return_value(), m::Literal(do_three_times(UBits(0, 32))));
}

TEST_F(OptimizationPassOverridesTest, OverrideCompoundPasses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 32));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  OptimizationPassRegistry registry;
  XLS_ASSERT_OK(registry.Register(
      "incr", optimization_registry::internal::Pass<IncrRet>(registry)));
  XLS_ASSERT_OK(registry.Register(
      "double", optimization_registry::internal::Pass<DoubleRet>(registry)));

  OptimizationPipelineProto pipeline_proto;
  XLS_ASSERT_OK(ParseTextProto(R"pb(
                                 compound_passes {
                                   short_name: "compound"
                                   long_name: "compound"
                                   passes: "incr"
                                   passes: "double"
                                 }
                               )pb",
                               "/dev/null", &pipeline_proto));
  XLS_ASSERT_OK(
      registry.RegisterPipelineProto(pipeline_proto, "custom-registry"));

  TripleOverrides overrides;
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OptimizationPassRegistry> decorated,
                           overrides.decorator().Decorate(registry));
  XLS_ASSERT_OK_AND_ASSIGN(auto gen, decorated->Generator("compound"));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OptimizationPass> pipeline,
                           gen->Generate());
  PassResults results;
  OptimizationContext context;
  XLS_ASSERT_OK(
      pipeline->Run(p.get(), OptimizationPassOptions(), &results, context));

  auto do_incr = [](Bits b) -> Bits { return bits_ops::Increment(b); };
  auto do_double = [](Bits b) -> Bits {
    return bits_ops::UMul(b, UBits(2, b.bit_count())).Slice(0, 32);
  };
  auto do_three_times = [&](Bits b) -> Bits {
    for (int64_t i = 0; i < 3; ++i) {
      b = do_incr(b);
    }
    for (int64_t i = 0; i < 3; ++i) {
      b = do_double(b);
    }
    return b;
  };
  auto do_compund_three_times = [&](Bits b) -> Bits {
    for (int64_t i = 0; i < 3; ++i) {
      b = do_three_times(b);
    }
    return b;
  };
  RecordProperty("result", f->return_value()->ToString());
  EXPECT_THAT(f->return_value(),
              m::Literal(do_compund_three_times(UBits(0, 32))));
}

}  // namespace
}  // namespace xls::tools
