// Copyright 2021 The XLS Authors
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

#ifndef XLS_NOC_DRIVERS_EXPERIMENT_FACTORY_H_
#define XLS_NOC_DRIVERS_EXPERIMENT_FACTORY_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/noc/drivers/experiment.h"

namespace xls::noc {

// A factory object that can register a set of ExperimentBuilderBase* objects
// for later retrieval via a tag.
//
class ExperimentFactory {
 public:
  // Returns list of all pre-defined experiment tags.
  std::vector<std::string> ListExperimentTags() const {
    std::vector<std::string> tags;

    for (const auto& [tag, builder] : experiments_) {
      tags.push_back(tag);
    }

    return tags;
  }

  // Given a tag, returns a new experiment object.
  absl::StatusOr<Experiment> BuildExperiment(std::string_view tag) const {
    if (!experiments_.contains(tag)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s tag has not been registered", tag));
    }

    return experiments_.at(tag)->BuildExperiment();
  }

  // Register new experiment builder and returns pointer to the underlying
  // builder.
  //
  // Note: The returned pointer points to the newly created object and can
  //       be used to provide additional parameters to the builder.
  template <class ExperimentBuilder>
  absl::StatusOr<ExperimentBuilder*> RegisterNewBuilder(std::string_view tag) {
    if (experiments_.contains(tag)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s tag already registered", tag));
    }

    experiments_[tag] = std::make_unique<ExperimentBuilder>();
    return down_cast<ExperimentBuilder*>(experiments_[tag].get());
  }

 private:
  absl::btree_map<std::string, std::unique_ptr<ExperimentBuilderBase>>
      experiments_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_DRIVERS_EXPERIMENT_FACTORY_H_
