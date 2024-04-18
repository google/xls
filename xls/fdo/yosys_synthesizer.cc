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

#include "xls/fdo/yosys_synthesizer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/status_macros.h"
#include "xls/fdo/extract_nodes.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace synthesis {

absl::StatusOr<int64_t> YosysSynthesizer::SynthesizeVerilogAndGetDelay(
    std::string_view verilog_text, std::string_view top_module_name) const {
  synthesis::CompileRequest request;
  request.set_module_text(verilog_text);
  request.set_top_module_name(top_module_name);
  request.set_target_frequency_hz(kFrequencyHz);

  synthesis::CompileResponse response;
  XLS_RETURN_IF_ERROR(service_.RunSynthesis(&request, &response));
  return response.slack_ps() == 0 ? 0 : kClockPeriodPs - response.slack_ps();
}

absl::StatusOr<std::unique_ptr<Synthesizer>>
YosysSynthesizerFactory::CreateSynthesizer(
    const SynthesizerParameters &parameters) {
  const auto& yosys_synthesizer_parameters =
      down_cast<const YosysSynthesizerParameters &>(parameters);
  return std::make_unique<YosysSynthesizer>(
      yosys_synthesizer_parameters.yosys_path(),
      yosys_synthesizer_parameters.sta_path(),
      yosys_synthesizer_parameters.synthesis_libraries());
}

absl::StatusOr<std::unique_ptr<Synthesizer>>
YosysSynthesizerFactory::CreateSynthesizer(
    const SchedulingOptions &scheduling_options) {
  if (scheduling_options.fdo_yosys_path().empty() ||
      scheduling_options.fdo_sta_path().empty() ||
      scheduling_options.fdo_synthesis_libraries().empty()) {
    return absl::InternalError(
        "yosys_path, sta_path, and synthesis_libraries must not be empty");
  }
  YosysSynthesizerParameters yosys_synthesizer_parameters(
      scheduling_options.fdo_yosys_path(), scheduling_options.fdo_sta_path(),
      scheduling_options.fdo_synthesis_libraries());
  return CreateSynthesizer(yosys_synthesizer_parameters);
}

XLS_REGISTER_MODULE_INITIALIZER(yosys, {
  CHECK_OK(GetSynthesizerManagerSingleton().RegisterSynthesizer(
      std::make_unique<YosysSynthesizerFactory>()));
});

}  // namespace synthesis
}  // namespace xls
