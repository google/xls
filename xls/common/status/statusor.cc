// Copyright 2020 Google LLC
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

#include "xls/common/status/statusor.h"

#include <utility>

#include "absl/status/status.h"
#include "xls/common/logging/logging.h"

namespace xabsl {
namespace internal_statusor {

void Helper::HandleInvalidStatusCtorArg(absl::Status* status) {
  const char* kMessage =
      "An OK status is not a valid constructor argument to StatusOr<T>";
  XLS_LOG(DFATAL) << kMessage;
  // In optimized builds, we will fall back to absl::StatusCode::kInternal.
  *status = absl::InternalError(kMessage);
}

void Helper::Crash(const absl::Status& status) {
  XLS_LOG(FATAL) << "Attempting to fetch value instead of handling error "
                 << status;
}

void CrashBecauseOfBadAccess(absl::Status status) {
  ABSL_RAW_LOG(FATAL, "Attempting to fetch value instead of handling error %s",
               status.ToString().c_str());
}

}  // namespace internal_statusor
}  // namespace xabsl
