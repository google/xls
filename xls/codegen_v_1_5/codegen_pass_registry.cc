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

#include "xls/codegen_v_1_5/codegen_pass_registry.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_wrapper_pass.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"

namespace xls::codegen {

namespace {

class OptPassWrapperGenerator final : public CodegenPassGenerator {
 public:
  explicit OptPassWrapperGenerator(const CodegenPassRegistryBase& registry,
                                   OptimizationPassGenerator* opt_gen)
      : CodegenPassGenerator(registry), opt_gen_(opt_gen) {}

  absl::StatusOr<std::unique_ptr<BlockConversionPass>> Generate() const final {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> opt_pass,
                         opt_gen_->Generate());
    return std::make_unique<BlockConversionWrapperPass>(std::move(opt_pass));
  }

  std::unique_ptr<CodegenPassGenerator> Clone(
      const CodegenPassRegistryBase& registry) const final {
    return std::make_unique<OptPassWrapperGenerator>(registry, opt_gen_);
  }

 private:
  OptimizationPassGenerator* opt_gen_;
};

}  // namespace

void CodegenPassRegistry::InitBeforeAccessLocked() const {
  if (opt_passes_registered_) {
    return;
  }
  opt_passes_registered_ = true;

  for (const auto& info : GetOptimizationRegistry().GetRegisteredInfos()) {
    std::string codegen_name = absl::StrFormat("opt_pass<%s>", info.short_name);
    absl::StatusOr<OptimizationPassGenerator*> opt_gen =
        GetOptimizationRegistry().Generator(info.short_name);
    if (opt_gen.ok()) {
      CHECK_OK(RegisterLocked(
          codegen_name,
          std::make_unique<OptPassWrapperGenerator>(*this, *opt_gen)));
      AddRegistrationInfoLocked(
          codegen_name, absl::StrFormat("opt_pass<%s>", info.class_name),
          info.header_file);
    }
  }
}

CodegenPassRegistry& GetCodegenPassRegistry() {
  static CodegenPassRegistry kSingleCodegenPassRegistry;
  return kSingleCodegenPassRegistry;
}

CodegenPassRegistry CodegenPassRegistry::OverridableClone() const {
  // Just copy then call the protected overwrite function.
  CodegenPassRegistry cpy = *this;
  cpy.set_allow_overwrite(true);
  return cpy;
}

namespace {

class CompoundPassAdder final : public CodegenPassGenerator {
 public:
  explicit CompoundPassAdder(const CodegenPassRegistryBase& registry,
                             CodegenPipelineProto::CompoundPass compound)
      : CodegenPassGenerator(registry), compound_(std::move(compound)) {}
  std::unique_ptr<CodegenPassGenerator> Clone(
      const CodegenPassRegistryBase& registry) const final {
    return std::make_unique<CompoundPassAdder>(registry, compound_);
  }
  absl::StatusOr<std::unique_ptr<BlockConversionPass>> Generate() const final {
    auto res = std::make_unique<BlockConversionCompoundPass>(
        compound_.short_name(), compound_.long_name());
    for (const auto& pass : compound_.passes()) {
      XLS_ASSIGN_OR_RETURN(auto* generator, registry().Generator(pass));
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<BlockConversionPass> pass_instance,
                           generator->Generate());
      res->AddOwned(std::move(pass_instance));
    }
    return res;
  }

 private:
  CodegenPipelineProto::CompoundPass compound_;
};

}  // namespace

absl::Status CodegenPassRegistry::RegisterPipelineProto(
    const CodegenPipelineProto& pipeline, std::string_view file) {
  for (const auto& compound : pipeline.compound_passes()) {
    XLS_RETURN_IF_ERROR(
        Register(compound.short_name(),
                 std::make_unique<CompoundPassAdder>(*this, compound)))
        << "Failed to register compound pass " << compound.short_name();
    AddRegistrationInfo(compound.short_name(),
                        "CodegenPipelineProto.CompoundPass", file);
  }

  // Now add the default pipeline.
  CodegenPipelineProto::CompoundPass compound;
  compound.set_short_name("default_pipeline");
  compound.set_long_name("Default codegen pipeline");
  for (const auto& pass : pipeline.default_pipeline()) {
    compound.add_passes(pass);
  }
  AddRegistrationInfo(compound.short_name(), "CodegenPipelineProto", file);
  XLS_RETURN_IF_ERROR(
      Register(compound.short_name(),
               std::make_unique<CompoundPassAdder>(*this, compound)))
      << "Failed to register compound pass " << compound.short_name();
  return absl::OkStatus();
}

absl::Status RegisterCodegenPipelineProtoData(absl::Span<uint8_t const> data,
                                              std::string_view file) {
  CodegenPipelineProto proto;
  if (!proto.ParseFromArray(data.data(), data.size())) {
    return absl::InternalError("Failed to parse CodegenPipelineProto");
  }
  return GetCodegenPassRegistry().RegisterPipelineProto(proto, file);
}

}  // namespace xls::codegen
