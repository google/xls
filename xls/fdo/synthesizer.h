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
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"
#include "xls/scheduling/scheduling_options.h"

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
  // proc. The default implementation generates intermediate Verilog and calls
  // `SynthesizeVerilogAndGetDelay`; a subclass may override this if desired,
  // for example, to change how the intermediate Verilog is structured.
  virtual absl::StatusOr<int64_t> SynthesizeNodesAndGetDelay(
      const absl::flat_hash_set<Node *> &nodes) const;

  // Launches "SynthesizeNodesAndGetDelay" concurrently for each set of nodes
  // listed in "nodes_list" and get their delays.
  absl::StatusOr<std::vector<int64_t>> SynthesizeNodesConcurrentlyAndGetDelays(
      absl::Span<const absl::flat_hash_set<Node *>> nodes_list) const;

 private:
  // Records the name of the concreate synthesizer, e.g., yosys, for management
  // and debugging purpose.
  std::string name_;
};

// An abstract class of a synthesis service.
class SynthesizerParameters {
 public:
  explicit SynthesizerParameters(std::string name) : name_(std::move(name)) {}
  SynthesizerParameters() = default;
  virtual ~SynthesizerParameters() = default;
  const std::string &name() const { return name_; }

 private:
  // Records the name of the concreate synthesizer, e.g., yosys, for
  // management and debugging purpose.
  std::string name_;
};

// An abstract class that can construct synthesizers. Meant to be passed to the
// manager at application init by synthesizers.
class SynthesizerFactory {
 public:
  explicit SynthesizerFactory(std::string_view name) : name_(name) {}
  virtual ~SynthesizerFactory() = default;

  // Creates a Synthesizer directly from its
  virtual absl::StatusOr<std::unique_ptr<Synthesizer>> CreateSynthesizer(
      const SynthesizerParameters &parameters) = 0;

  // Creates a SynthesizerParameters object from the given scheduling options.
  virtual absl::StatusOr<std::unique_ptr<Synthesizer>> CreateSynthesizer(
      const SchedulingOptions &scheduling_options) {
    return absl::UnimplementedError(
        absl::StrFormat("Not implemented for %s synthesizer", name_));
  };

  const std::string &name() const { return name_; }

 private:
  // Records the name of the concreate synthesizer, e.g., yosys, for management
  // and debugging purpose.
  std::string name_;
};

// An abstraction which holds multiple SynthesizerManager objects organized by
// name.
class SynthesizerManager {
 public:
  // Returns a Synthesizer object associated with the given name. User must
  // provide a SynthesizerParameters object that matches the type. For example
  // "yosys" must be a YosysSynthesizerParameters.
  absl::StatusOr<std::unique_ptr<Synthesizer>> MakeSynthesizer(
      std::string_view, SynthesizerParameters &parameters);

  // Make synthesizer using SchedulingOptions proto
  absl::StatusOr<std::unique_ptr<Synthesizer>> MakeSynthesizer(
      std::string_view, const SchedulingOptions &scheduling_options);

  // Adds a Synthesizer to the manager and associates it with the given name.
  absl::Status RegisterSynthesizer(
      std::unique_ptr<SynthesizerFactory> synthesizer);

  // Returns a list of the names of available models in this manager.
  absl::Span<const std::string> synthesizer_names() const {
    return synthesizer_names_;
  }

 private:
  absl::StatusOr<SynthesizerFactory *> GetSynthesizerFactory(
      std::string_view name);

  absl::flat_hash_map<std::string, std::unique_ptr<SynthesizerFactory>>
      synthesizers_;
  std::vector<std::string> synthesizer_names_;
};

// Returns the singleton SynthesizerManager manager where synthesizer are
// registered.
SynthesizerManager &GetSynthesizerManagerSingleton();

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_FDO_SYNTHESIZER_H_
