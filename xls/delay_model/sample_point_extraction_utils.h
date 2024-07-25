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

#ifndef XLS_DELAY_MODEL_SAMPLE_POINT_EXTRACTION_UTILS_H_
#define XLS_DELAY_MODEL_SAMPLE_POINT_EXTRACTION_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_model.pb.h"
#include "xls/ir/package.h"

namespace xls::delay_model {

// A flattened rendition of an `OpSamples` message for one specific
// `Parameterization`, annotated with additional info.
struct SamplePoint {
  // The operation name, as it would appear in an `OpSamples` message.
  std::string op_name;

  // The inputs and outputs.
  Parameterization params;

  // How many times the operation occurs in the corpus.
  int64_t frequency = 0;

  // The estimated delay in picoseconds, according to some existing delay model.
  int64_t delay_estimate_in_ps = 0;
};

// Creates `SamplePoint` objects for the operations actually performed by the IR
// in `package`. Only operations having a regression estimator, according to
// `op_models`, are included, since only that estimator needs sample points. If
// a `delay_estimator` is specified, then the returned samples have a
// `delay_estimate_in_ps` populated; otherwise, it is zero. The returned vector
// is in a canonical order by op name and sizes of the parameters.
absl::StatusOr<std::vector<SamplePoint>> ExtractSamplePoints(
    const Package& package, const OpModels& op_models,
    std::optional<DelayEstimator*> delay_estimator = std::nullopt);

// Converts the given `samples` into an `OpSamplesList` proto suitable for
// downstream use. The output is limited to the first `n` elements, if `samples`
// contains more than that. The resulting list always has a `kIdentity` sample,
// which does not count towards `n`, regardless of whether it even appears in
// `samples`.
OpSamplesList ConvertToOpSamplesList(
    absl::Span<const SamplePoint> samples,
    size_t n = std::numeric_limits<size_t>::max());

}  // namespace xls::delay_model

#endif  // XLS_DELAY_MODEL_SAMPLE_POINT_EXTRACTION_UTILS_H_
