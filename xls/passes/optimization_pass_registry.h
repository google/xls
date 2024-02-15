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

#include <algorithm>
#include <memory>
#include <string_view>
#include <tuple>
#include <variant>

#include "absl/status/status.h"
#include "xls/common/logging/logging.h"
#include "xls/common/module_initializer.h"
#include "xls/passes/optimization_pass.h"

namespace xls {

// Get the singleton pass registry for optimization passes.
OptimizationPassRegistry& GetOptimizationRegistry();

namespace pass_config {
struct OptLevel : public std::monostate {};

// Token to replace with opt level but capped to some given value.
struct CappedOptLevel {
  // Maximum opt level allowed.
  decltype(kMaxOptLevel) cap;
};

static constexpr OptLevel kOptLevel{};
}  // namespace pass_config

// Helpers to handle creating and configuring passes in a somewhat reasonable
// way.
//
// TODO(https://github.com/google/xls/issues/274): All this TMP stuff is
// required to support passes which take non-trivial configuration. We really
// should refactor the pass system so every pass only takes a single argument,
// the opt level, and nothing else.
namespace optimization_registry::internal {
// Transform the input given the config. Does nothing by default since the
// opt-level does not need to be bound to anything.
template <typename Input>
inline auto Transform(Input input,
                      const OptimizationPassStandardConfig& config) {
  return input;
}

// Transform the input given the config. Returns the opt level capped to the
// given value.
//
// This binds the real capped opt level to the marker struct.
template <>
inline auto Transform(pass_config::CappedOptLevel input,
                      const OptimizationPassStandardConfig& config) {
  return std::min(input.cap, config);
}

// Transform the input given the config. Returns the opt level.
//
// This binds the real opt-level to the opt-level marker token.
template <>
inline auto Transform(pass_config::OptLevel input,
                      const OptimizationPassStandardConfig& config) {
  return config;
}

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
  absl::Status AddToPipeline(
      OptimizationCompoundPass* pass,
      const OptimizationPassStandardConfig& config) const final {
    // Function to bind the actual value of the opt-level into any arguments
    // that need it (eg arguments that are the opt level or a bounded
    // opt-level).
    auto transform = [&](auto arg) { return Transform(arg, config); };
    // Actually construct and insert the PassClass instance
    auto function = [&](auto... args) {
      pass->Add<PassClass>(transform(std::forward<decltype(args)>(args))...);
    };
    // Unpack the argument tuple.
    std::apply(function, args_);
    return absl::OkStatus();
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

}  // namespace optimization_registry::internal

template <typename PassT, typename... Args>
absl::Status RegisterOptimizationPass(std::string_view name, Args... args) {
  using optimization_registry::internal::Pass;
  return GetOptimizationRegistry().Register(
      name, Pass<PassT>(std::forward<Args>(args)...));
}

template <typename PassT>
absl::Status RegisterOptimizationPass() {
  return RegisterOptimizationPass<PassT>(PassT::kName);
}

#define REGISTER_OPT_PASS(ty, ...)                                        \
  XLS_REGISTER_MODULE_INITIALIZER(initializer_##ty##_register, {          \
    XLS_CHECK_OK(RegisterOptimizationPass<ty>(ty::kName, ##__VA_ARGS__)); \
  })

}  // namespace xls

#endif  // XLS_PASSES_OPTIMIZATION_PASS_REGISTRY_H_
