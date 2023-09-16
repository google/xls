// Copyright 2023 The XLS Authors
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

#include "xls/fdo/synthesizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/fdo/extract_nodes.h"
#include "xls/ir/node.h"
#include "xls/synthesis/synthesis.pb.h"

namespace xls {
namespace synthesis {

absl::StatusOr<std::vector<int64_t>>
Synthesizer::SynthesizeNodesConcurrentlyAndGetDelays(
    absl::Span<const absl::flat_hash_set<Node *>> nodes_list) const {
  // Launches multi-threading delay estimation.
  std::vector<absl::StatusOr<int64_t>> results;
  std::vector<std::unique_ptr<Thread>> threads;
  results.reserve(nodes_list.size());
  absl::StatusOr<int64_t> init_val = 0;
  for (const absl::flat_hash_set<Node *> &nodes : nodes_list) {
    results.push_back(init_val);
    absl::StatusOr<int64_t> &dest = results.back();
    // TODO(hanchenye): 2023-08-14 Use a thread pool structure that we can
    // schedule on.
    threads.push_back(std::make_unique<Thread>(
        [&]() { dest = SynthesizeNodesAndGetDelay(nodes); }));
  }

  // Records the estimated delays.
  for (auto &t : threads) {
    t->Join();
  }
  std::vector<int64_t> delay_list;
  delay_list.reserve(results.size());
  for (absl::StatusOr<int64_t> result : results) {
    XLS_RETURN_IF_ERROR(result.status());
    delay_list.push_back(result.value());
  }
  return delay_list;
}

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

absl::StatusOr<int64_t> YosysSynthesizer::SynthesizeNodesAndGetDelay(
    const absl::flat_hash_set<Node *> &nodes) const {
  std::string top_name = "tmp_module";
  XLS_ASSIGN_OR_RETURN(
      std::string verilog_text,
      ExtractNodesAndGetVerilog(nodes, top_name, /*flop_inputs_outputs=*/true));
  XLS_ASSIGN_OR_RETURN(int64_t nodes_delay,
                       SynthesizeVerilogAndGetDelay(verilog_text, top_name));
  return nodes_delay;
}

}  // namespace synthesis
}  // namespace xls
