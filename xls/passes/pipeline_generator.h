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

#ifndef XLS_PASSES_PIPELINE_GENERATOR_H_
#define XLS_PASSES_PIPELINE_GENERATOR_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"

namespace xls {

// A base class for generating pipelines based on a text program.
//
// Each generator of a pipeline should define a subclass which overrides the
// AddPassToPipeline function as appropriate to construct the requested passes.
template <typename IrT, typename OptionsT, typename ResultsT = PassResults>
class PipelineGeneratorBase {
 public:
  virtual ~PipelineGeneratorBase() = default;
  constexpr PipelineGeneratorBase(std::string_view short_name,
                                  std::string_view long_name)
      : short_name_(short_name), long_name_(long_name) {}

  absl::StatusOr<std::unique_ptr<CompoundPassBase<IrT, OptionsT, ResultsT>>>
  GeneratePipeline(const PassPipelineProto& pipeline) const {
    XLS_RET_CHECK(pipeline.has_top()) << "Empty pipeline";
    auto top = std::make_unique<CompoundPassBase<IrT, OptionsT, ResultsT>>(
        short_name_, long_name_);
    XLS_RETURN_IF_ERROR(GeneratePipelineElement(top.get(), pipeline.top()));
    return std::move(top);
  }

  // Very simple grammar:
  // <PIPELINE> ::= <NAME> | <NAME> <PIPELINE> | <FIXEDPOINT>
  // <FIXEDPOINT> := '[' <PIPELINE> ']'
  // <NAME> := [a-zA-Z0-9{}!@#$%^&*-_=+;:'",.<>/?\\*+`~()]+
  // <NAME> tokens are delimited by whitespace **ONLY**.
  absl::StatusOr<std::unique_ptr<CompoundPassBase<IrT, OptionsT, ResultsT>>>
  GeneratePipeline(std::string_view pipeline) const {
    std::string passes = absl::StrReplaceAll(
        pipeline, {{"\n", " "}, {"\t", " "}, {"[", " [ "}, {"]", " ] "}});
    auto toks = absl::StrSplit(passes, ' ');
    std::vector<std::unique_ptr<CompoundPassBase<IrT, OptionsT, ResultsT>>>
        stack;
    stack.emplace_back(
        std::make_unique<CompoundPassBase<IrT, OptionsT, ResultsT>>(
            short_name_, long_name_));
    int fp_cnt = 0;
    for (auto v : toks) {
      XLS_RET_CHECK_GE(stack.size(), 1);
      if (v.empty()) {
        continue;
      }
      if (v == "]") {
        XLS_RET_CHECK_GE(stack.size(), 2)
            << "Invalid pipeline definition. Unmatched ']' in pipeline.";
        std::unique_ptr<CompoundPassBase<IrT, OptionsT, ResultsT>> last =
            std::move(stack.back());
        stack.pop_back();
        XLS_ASSIGN_OR_RETURN(auto finalized,
                             FinalizeWithOptions(std::move(last), {}));
        stack.back()->AddOwned(std::move(finalized));
        continue;
      }
      if (v == "[") {
        stack.emplace_back(std::make_unique<
                           FixedPointCompoundPassBase<IrT, OptionsT, ResultsT>>(
            absl::StrFormat("fp-%s-%d", short_name_, fp_cnt),
            absl::StrFormat("fixed-point-%s-%d", long_name_, fp_cnt)));
        fp_cnt++;
        continue;
      }
      XLS_RETURN_IF_ERROR(
          AddPassToPipeline(stack.back().get(), v, /*options=*/{}))
          << "Unable to add pass '" << v << "' to pipeline";
    }
    XLS_RET_CHECK_EQ(stack.size(), 1)
        << "Invalid pipeline definition. Unmatched '[' in pipeline.";
    return std::move(stack.back());
  }

 protected:
  virtual absl::Status AddPassToPipeline(
      CompoundPassBase<IrT, OptionsT, ResultsT>* holder_pass,
      std::string_view pass_name,
      const PassPipelineProto::PassOptions& options) const = 0;

  // Apply any options given to the 'cur' pipeline and return it.
  virtual absl::StatusOr<std::unique_ptr<PassBase<IrT, OptionsT, ResultsT>>>
  FinalizeWithOptions(
      std::unique_ptr<CompoundPassBase<IrT, OptionsT, ResultsT>>&& cur,
      const PassPipelineProto::PassOptions& options) const = 0;

  absl::Status GeneratePipelineElement(
      CompoundPassBase<IrT, OptionsT, ResultsT>* current_pipeline,
      const PassPipelineProto::Element& element) const {
    if (element.has_pass_name()) {
      return AddPassToPipeline(current_pipeline, element.pass_name(),
                               element.options());
    }
    if (element.has_fixedpoint()) {
      auto fp_proto = element.fixedpoint();
      auto fixedpoint =
          std::make_unique<FixedPointCompoundPassBase<IrT, OptionsT, ResultsT>>(
              fp_proto.has_short_name() ? fp_proto.short_name() : "fixedpoint",
              fp_proto.has_long_name() ? fp_proto.long_name() : "fixedpoint");
      for (const PassPipelineProto::Element& e : fp_proto.elements()) {
        XLS_RETURN_IF_ERROR(GeneratePipelineElement(fixedpoint.get(), e));
      }
      XLS_ASSIGN_OR_RETURN(
          auto finalized,
          FinalizeWithOptions(std::move(fixedpoint), element.options()));
      current_pipeline->AddOwned(std::move(finalized));
      return absl::OkStatus();
    }
    if (element.has_pipeline()) {
      auto p_proto = element.pipeline();
      auto pipeline =
          std::make_unique<CompoundPassBase<IrT, OptionsT, ResultsT>>(
              p_proto.has_short_name() ? p_proto.short_name() : "pipeline",
              p_proto.has_long_name() ? p_proto.long_name() : "pipeline");
      for (const PassPipelineProto::Element& e : p_proto.elements()) {
        XLS_RETURN_IF_ERROR(GeneratePipelineElement(pipeline.get(), e));
      }
      XLS_ASSIGN_OR_RETURN(
          auto finalized,
          FinalizeWithOptions(std::move(pipeline), element.options()));
      current_pipeline->AddOwned(std::move(finalized));
      return absl::OkStatus();
    }
    LOG(WARNING) << "Pipeline element " << element << " has no passes.";
    return absl::OkStatus();
  }

 private:
  std::string short_name_;
  std::string long_name_;
};

}  // namespace xls

#endif  // XLS_PASSES_PIPELINE_GENERATOR_H_
