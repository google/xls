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

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/source_location.h"
#include "xls/common/status/status_builder.h"

namespace xabsl {

/* static */ void StatusBuilder::AddSourceLocation(absl::Status& status,
                                                 xabsl::SourceLocation loc) {
  // Not supported in OSS absl::Status at the moment.
}
/* static */ absl::Span<const SourceLocation> StatusBuilder::GetSourceLocations(
    const absl::Status& status) {
  // Not supported in OSS absl::Status at the moment.
  return {};
}

}  // namespace xabsl
