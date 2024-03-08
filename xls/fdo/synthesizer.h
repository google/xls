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

#ifndef XLS_FDO_SYNTHESIZER_H_
#define XLS_FDO_SYNTHESIZER_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/synthesis/yosys/yosys_synthesis_service.h"

namespace xls {
namespace synthesis {

// An abstract class of a synthesis service.
class Synthesizer {
 public:
  explicit Synthesizer(std::string_view name) : name_(name) {}
  virtual ~Synthesizer() = default;

  const std::string &name() const { return name_; }

  // Synthesizes the given Verilog module with a synthesis tool and return its
  // overall delay.
  virtual absl::StatusOr<int64_t> SynthesizeVerilogAndGetDelay(
      std::string_view verilog_text,
      std::string_view top_module_name) const = 0;

  // Wraps the given set of nodes into a module, synthesize the module with a
  // synthesis tool, and return its overall delay. The nodes set can be an
  // arbitrary subgraph or multiple disjointed subgraphs from a function or
  // proc.
  virtual absl::StatusOr<int64_t> SynthesizeNodesAndGetDelay(
      const absl::flat_hash_set<Node *> &nodes) const = 0;

  // Launches "SynthesizeNodesAndGetDelay" concurrently for each set of nodes
  // listed in "nodes_list" and get their delays.
  absl::StatusOr<std::vector<int64_t>> SynthesizeNodesConcurrentlyAndGetDelays(
      absl::Span<const absl::flat_hash_set<Node *>> nodes_list) const;

 private:
  // Records the name of the concreate synthesizer, e.g., yosys, for management
  // and debugging purpose.
  std::string name_;
};

// A derived Synthesizer class for Yosys-OpenSTA-based synthesis and static
// timing analysis.
class YosysSynthesizer : public Synthesizer {
  static constexpr int64_t kFrequencyHz = 1e9;
  static constexpr int64_t kClockPeriodPs = 1e12 / kFrequencyHz;

 public:
  explicit YosysSynthesizer(std::string_view yosys_path,
                            std::string_view sta_path,
                            std::string_view synthesis_libraries)
      : Synthesizer("yosys"),
        service_(yosys_path, /*nextpnr_path=*/"", /*synthesis_target=*/"",
                 sta_path, synthesis_libraries, synthesis_libraries,
                 /*save_temps=*/false, /*return_netlist=*/false,
                 /*synthesis_only=*/false) {}

  absl::StatusOr<int64_t> SynthesizeVerilogAndGetDelay(
      std::string_view verilog_text,
      std::string_view top_module_name) const override;

  absl::StatusOr<int64_t> SynthesizeNodesAndGetDelay(
      const absl::flat_hash_set<Node *> &nodes) const override;

 private:
  YosysSynthesisServiceImpl service_;
};

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_SYNTHESIZER_H_
