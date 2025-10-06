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

#ifndef XLS_DSLX_DSLX_PAYLOADS_H_
#define XLS_DSLX_DSLX_PAYLOADS_H_

#include <optional>

#include "absl/status/status.h"
#include "xls/dslx/type_system/type_info.pb.h"

namespace xls::dslx {

// Attaches `payload` to `status`.
void SetStatusPayload(absl::Status& status, const StatusPayloadProto& payload);

// Retrieves a `StatusPayloadProto` from `status`, if one is attached.
std::optional<StatusPayloadProto> GetStatusPayload(const absl::Status& status);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DSLX_PAYLOADS_H_
