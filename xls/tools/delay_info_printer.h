// Copyright 2025 The XLS Authors
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

#ifndef XLS_TOOLS_DELAY_INFO_PRINTER_H_
#define XLS_TOOLS_DELAY_INFO_PRINTER_H_

#include <memory>

#include "absl/status/status.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/fdo/synthesizer.h"
#include "xls/tools/delay_info_flags.pb.h"

namespace xls::tools {

// A utility for generating delay info for an IR package that is associated with
// it at init time.
class DelayInfoPrinter {
 public:
  virtual ~DelayInfoPrinter() = default;

  // Initializes this helper using only the command line flags.
  virtual absl::Status Init(DelayInfoFlagsProto flags) = 0;

  // Initializes this helper using the given command line flags, but with the
  // caller providing the delay estimator and synthesizer. This overload will
  // not automatically create a delay estimator or synthesizer based on the
  // flags.
  virtual absl::Status Init(
      DelayInfoFlagsProto flags, DelayEstimator* delay_estimator,
      std::unique_ptr<synthesis::Synthesizer> synthesizer) = 0;

  // Generates the applicable delay info based on the flags that were provided
  // at init time. The primary output destination is stdout, but if a proto file
  // is specified in the flags provided at init time, then proto output is also
  // written to that file.
  virtual absl::Status GenerateApplicableInfo() = 0;
};

// Creates a `DelayInfoPrinter` implementation.
std::unique_ptr<DelayInfoPrinter> CreateDelayInfoPrinter();

}  // namespace xls::tools

#endif  // XLS_TOOLS_DELAY_INFO_PRINTER_H_
