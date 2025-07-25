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

#ifndef XLS_PASSES_OPTIMIZATION_PASS_REGISTRY_H_
#define XLS_PASSES_OPTIMIZATION_PASS_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/module_initializer.h"
#include "xls/common/source_location.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/pipeline_generator.h"

namespace xls {

// The name of the default pipeline in the optimization pipeline proto.
inline constexpr std::string_view kDefaultPassPipelineName = "default_pipeline";

// Helper for pass_registry which allows one to create a copy that can be
// overridden and to register the pipeline proto.
class OptimizationPassRegistry : public OptimizationPassRegistryBase {
 public:
  using OptimizationPassRegistryBase::OptimizationPassRegistryBase;
  // Register the compound passes described in the pipeline.
  absl::Status RegisterPipelineProto(const OptimizationPipelineProto& pipeline,
                                     std::string_view file);
  // Create a copy of this registry where existing pass names can be overwritten
  // without errors. Lifetime is the same as the source registry.
  OptimizationPassRegistry OverridableClone() const;
};

// Get the singleton pass registry for optimization passes.
OptimizationPassRegistry& GetOptimizationRegistry();

// Add wrappers that check for common options (opt-level, resource sharing, etc)
// and apply them.
absl::StatusOr<std::unique_ptr<OptimizationPass>> WrapPassWithOptions(
    std::unique_ptr<OptimizationPass>&& cur,
    const BasicPipelineOptions& options);

// Helpers to handle creating and configuring passes in a somewhat reasonable
// way.
//
// TODO(https://github.com/google/xls/issues/274): All this TMP stuff is
// required to support passes which take non-trivial configuration. We really
// should refactor the pass system so every pass only takes a single argument,
// the opt level, and nothing else.
namespace optimization_registry::internal {

// A templated class to hold the argument specification for creating a pass.
//
// This class holds a tuple of all the arguments needed to construct a PassClass
// instance (with some of them replaced with markers for cases where they depend
// on opt-level). When a new instance is requested this saved tuple is used to
// construct a new instance.
template <typename PassClass, typename... Args>
class Adder final : public OptimizationPassGenerator {
 public:
  explicit Adder(Args... args) : args_(std::forward_as_tuple(args...)) {}
  absl::StatusOr<std::unique_ptr<OptimizationPass>> Generate() const final {
    // Actually construct and insert the PassClass instance
    auto function = [&](auto... args) {
      std::unique_ptr<OptimizationPass> base =
          std::make_unique<PassClass>(std::forward<decltype(args)>(args)...);
      return base;
    };
    // Unpack the argument tuple.
    return std::apply(function, args_);
  }

 private:
  // All the arguments needed to construct a PassClass instance
  std::tuple<Args...> args_;
};
template <typename PassType, typename... Args>
std::unique_ptr<OptimizationPassGenerator> Pass(Args... args) {
  return std::make_unique<Adder<PassType, Args...>>(
      std::forward<Args>(args)...);
}

inline std::string ToHeader(std::string_view file_name) {
  if (file_name.ends_with(".cc")) {
    return absl::StrFormat("%s.h", file_name.substr(0, file_name.size() - 3));
  }
  return std::string(file_name);
}

}  // namespace optimization_registry::internal

template <typename PassT, typename... Args>
absl::Status RegisterOptimizationPass(std::string_view name, Args... args) {
  using optimization_registry::internal::Pass;
  return GetOptimizationRegistry().Register(
      name, Pass<PassT>(std::forward<Args>(args)...));
}

inline void RegisterOptimizationPassInfo(std::string_view name,
                                         std::string_view class_name,
                                         std::string_view header_file) {
  return GetOptimizationRegistry().AddRegistrationInfo(name, class_name,
                                                       header_file);
}

// Add source info for this name. This is for documentation generation.
inline void RegisterOptimizationPassInfoFor(
    absl::Span<std::string_view const> names,
    std::string_view class_name = "UNKNOWN",
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  for (auto name : names) {
    RegisterOptimizationPassInfo(
        name, class_name,
        optimization_registry::internal::ToHeader(loc.file_name()));
  }
}

absl::Status RegisterOptimizationPipelineProtoData(
    absl::Span<uint8_t const> data, std::string_view file);

#define REGISTER_OPT_PASS(ty, ...)                                            \
  XLS_REGISTER_MODULE_INITIALIZER(initializer_##ty##_register, {              \
    CHECK_OK(RegisterOptimizationPass<ty>(ty::kName, ##__VA_ARGS__));         \
    RegisterOptimizationPassInfo(                                             \
        ty::kName, #ty, optimization_registry::internal::ToHeader(__FILE__)); \
  })

#define REGISTER_NAMED_OPT_PASS(name, ty, ...)                            \
  XLS_REGISTER_MODULE_INITIALIZER(initializer_##name##_##ty##_register, { \
    CHECK_OK(RegisterOptimizationPass<ty>(name, ##__VA_ARGS__));          \
    RegisterOptimizationPassInfo(                                         \
        name, #ty, optimization_registry::internal::ToHeader(__FILE__));  \
  })

}  // namespace xls

#endif  // XLS_PASSES_OPTIMIZATION_PASS_REGISTRY_H_
