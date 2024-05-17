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

#include "xls/noc/drivers/sample_experiments.h"

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/noc/drivers/experiment_factory.h"
#include "xls/noc/drivers/samples/aggregate_tree_experiment.h"
#include "xls/noc/drivers/samples/simple_vc_experiment.h"

namespace xls::noc {

// Adds the sample experiments to the factory.
absl::Status RegisterSampleExperiments(ExperimentFactory& factory) {
  XLS_RET_CHECK_OK(RegisterAggregateTreeExperiment(factory));
  XLS_RET_CHECK_OK(RegisterSimpleVCExperiment(factory));
  return absl::OkStatus();
}

}  // namespace xls::noc
