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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/fdo/extract_nodes.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"
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

absl::StatusOr<int64_t> Synthesizer::SynthesizeNodesAndGetDelay(
    const absl::flat_hash_set<Node *> &nodes) const {
  std::string top_name = "tmp_module";
  XLS_ASSIGN_OR_RETURN(
      std::optional<std::string> verilog_text,
      ExtractNodesAndGetVerilog(nodes, top_name, /*flop_inputs_outputs=*/true));
  if (!verilog_text.has_value()) {
    return 0;
  }
  XLS_ASSIGN_OR_RETURN(
      int64_t nodes_delay,
      SynthesizeVerilogAndGetDelay(verilog_text.value(), top_name));
  return nodes_delay;
}

absl::StatusOr<SynthesizerFactory *> SynthesizerManager::GetSynthesizerFactory(
    std::string_view name) {
  if (!synthesizers_.contains(name)) {
    if (synthesizer_names_.empty()) {
      return absl::NotFoundError(
          absl::StrFormat("No synthesizer found named \"%s\". No "
                          "synthesizer are registered. Was InitXls called?",
                          name));
    }
    return absl::NotFoundError(absl::StrFormat(
        "No synthesizer found named \"%s\". Available synthesizers: %s", name,
        absl::StrJoin(synthesizer_names_, ", ")));
  }

  return synthesizers_.at(name).get();
}

absl::StatusOr<std::unique_ptr<Synthesizer>>
SynthesizerManager::MakeSynthesizer(std::string_view name,
                                    SynthesizerParameters &parameters) {
  XLS_ASSIGN_OR_RETURN(SynthesizerFactory * factory,
                       GetSynthesizerFactory(name));
  return factory->CreateSynthesizer(parameters);
};

absl::StatusOr<std::unique_ptr<Synthesizer>>
SynthesizerManager::MakeSynthesizer(
    std::string_view name, const SchedulingOptions &scheduling_options) {
  XLS_ASSIGN_OR_RETURN(SynthesizerFactory * factory,
                       GetSynthesizerFactory(name));
  return factory->CreateSynthesizer(scheduling_options);
};

absl::Status SynthesizerManager::RegisterSynthesizer(
    std::unique_ptr<SynthesizerFactory> synthesizer_factory) {
  std::string name = synthesizer_factory->name();
  if (synthesizers_.contains(name)) {
    return absl::InternalError(
        absl::StrFormat("SynthesizerFactory named %s already exists", name));
  }
  synthesizers_[name] = std::move(synthesizer_factory);
  synthesizer_names_.push_back(name);
  return absl::OkStatus();
}

SynthesizerManager &GetSynthesizerManagerSingleton() {
  static absl::NoDestructor<SynthesizerManager> manager;
  return *manager;
}

}  // namespace synthesis
}  // namespace xls
