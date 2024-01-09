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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/pass_base.h"

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
        stack.back()->AddOwned(std::move(last));
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
      XLS_RETURN_IF_ERROR(AddPassToPipeline(stack.back().get(), v))
          << "Unable to add pass '" << v << "' to pipeline";
    }
    XLS_RET_CHECK_EQ(stack.size(), 1)
        << "Invalid pipeline definition. Unmatched '[' in pipeline.";
    return std::move(stack.back());
  }

 protected:
  virtual absl::Status AddPassToPipeline(
      CompoundPassBase<IrT, OptionsT, ResultsT>* pass,
      std::string_view pass_name) const = 0;

 private:
  std::string short_name_;
  std::string long_name_;
};

}  // namespace xls

#endif  // XLS_PASSES_PIPELINE_GENERATOR_H_
