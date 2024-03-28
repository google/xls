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

#ifndef XLS_FDO_YOSYS_SYNTHESIZER_H_
#define XLS_FDO_YOSYS_SYNTHESIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/synthesis/yosys/yosys_synthesis_service.h"

namespace xls {
namespace synthesis {

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

// An abstract class of a synthesis service.
class YosysSynthesizerParameters : public SynthesizerParameters {
 public:
  explicit YosysSynthesizerParameters(std::string_view yosys_path,
                                      std::string_view sta_path,
                                      std::string_view synthesis_libraries)
      : SynthesizerParameters("yosys"),
        yosys_path_(yosys_path),
        sta_path_(sta_path),
        synthesis_libraries_(synthesis_libraries) {}
  virtual ~YosysSynthesizerParameters() = default;

  std::string yosys_path() const { return yosys_path_; }
  std::string sta_path() const { return sta_path_; }
  std::string synthesis_libraries() const { return synthesis_libraries_; }

 private:
  std::string yosys_path_;
  std::string sta_path_;
  std::string synthesis_libraries_;
};

// An abstract class that can construct synthesizers. Meant to be passed to the
// manager at application init by synthesizers.
class YosysSynthesizerFactory : public SynthesizerFactory {
 public:
  explicit YosysSynthesizerFactory() : SynthesizerFactory("yosys") {}
  virtual ~YosysSynthesizerFactory() = default;
  absl::StatusOr<std::unique_ptr<Synthesizer>> CreateSynthesizer(
      const SynthesizerParameters &parameters) override;
  absl::StatusOr<std::unique_ptr<Synthesizer>> CreateSynthesizer(
      const SchedulingOptions &scheduling_options) override;
};

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_YOSYS_SYNTHESIZER_H_
