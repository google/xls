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

#ifndef XLS_CODEGEN_V_1_5_CODEGEN_PASS_REGISTRY_H_
#define XLS_CODEGEN_V_1_5_CODEGEN_PASS_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/codegen_pass_pipeline.pb.h"
#include "xls/common/module_initializer.h"
#include "xls/passes/pass_registry.h"

namespace xls::codegen {

// Instantiate the generic template for Codegen options
using CodegenPassRegistryBase =
    PassRegistry<BlockConversionPassOptions, BlockConversionContext>;
using CodegenPassGenerator =
    PassGenerator<BlockConversionPassOptions, BlockConversionContext>;

class CodegenPassRegistry : public CodegenPassRegistryBase {
 public:
  using CodegenPassRegistryBase::CodegenPassRegistryBase;

  CodegenPassRegistry(CodegenPassRegistry&&) = default;
  CodegenPassRegistry& operator=(CodegenPassRegistry&&) = default;
  CodegenPassRegistry(const CodegenPassRegistry&) = default;
  CodegenPassRegistry& operator=(const CodegenPassRegistry&) = default;

  // Register compound passes described in the pipeline proto
  absl::Status RegisterPipelineProto(const CodegenPipelineProto& pipeline,
                                     std::string_view file);

  // Create a copy of this registry where we are allowed to override pass
  // names without error. Lifetime is same as source registry.
  CodegenPassRegistry OverridableClone() const;

 protected:
  void InitBeforeAccessLocked() const override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(registry_lock_);

 private:
  mutable bool opt_passes_registered_ = false;
};

// Singleton accessor
CodegenPassRegistry& GetCodegenPassRegistry();

namespace codegen_registry::internal {

template <typename PassClass, typename... Args>
class Adder final : public CodegenPassGenerator {
 public:
  explicit Adder(const CodegenPassRegistryBase& registry,
                 std::tuple<Args...> args)
      : CodegenPassGenerator(registry), args_(std::move(args)) {}
  absl::StatusOr<std::unique_ptr<BlockConversionPass>> Generate() const final {
    auto function = [&](auto... args) {
      return std::make_unique<PassClass>(std::forward<decltype(args)>(args)...);
    };
    return std::apply(function, args_);
  }
  std::unique_ptr<CodegenPassGenerator> Clone(
      const CodegenPassRegistryBase& registry) const final {
    return std::make_unique<Adder<PassClass, Args...>>(registry, args_);
  }

 private:
  std::tuple<Args...> args_;
};

template <typename PassType, typename... Args>
std::unique_ptr<CodegenPassGenerator> Pass(
    const CodegenPassRegistryBase& registry, Args... args) {
  return std::make_unique<Adder<PassType, Args...>>(
      registry, std::forward_as_tuple(args...));
}

}  // namespace codegen_registry::internal

template <typename PassT, typename... Args>
absl::Status RegisterCodegenPass(std::string_view name, Args... args) {
  using codegen_registry::internal::Pass;
  return GetCodegenPassRegistry().Register(
      name, Pass<PassT>(GetCodegenPassRegistry(), std::forward<Args>(args)...));
}

// Macro to register passes mechanically
#define REGISTER_CODEGEN_PASS(ty, ...)                                      \
  XLS_REGISTER_MODULE_INITIALIZER(initializer_##ty##_codegen_register, {    \
    CHECK_OK(                                                               \
        ::xls::codegen::RegisterCodegenPass<ty>(ty::kName, ##__VA_ARGS__)); \
  })

#define REGISTER_NAMED_CODEGEN_PASS(name, ty, ...)                         \
  XLS_REGISTER_MODULE_INITIALIZER(                                         \
      initializer_##name##_##ty##_codegen_register, {                      \
        CHECK_OK(                                                          \
            ::xls::codegen::RegisterCodegenPass<ty>(name, ##__VA_ARGS__)); \
      })

absl::Status RegisterCodegenPipelineProtoData(absl::Span<uint8_t const> data,
                                              std::string_view file);

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_CODEGEN_PASS_REGISTRY_H_
