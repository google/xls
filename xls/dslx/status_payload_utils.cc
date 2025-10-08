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

#include "xls/dslx/status_payload_utils.h"

#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "xls/dslx/status_payload.pb.h"

namespace xls::dslx {
namespace {

constexpr std::string_view kStatusPayloadName = "status_payload";

}  // namespace

void SetStatusPayload(absl::Status& status, const StatusPayloadProto& payload) {
  status.SetPayload(kStatusPayloadName, payload.SerializeAsCord());
}

std::optional<StatusPayloadProto> GetStatusPayload(const absl::Status& status) {
  std::optional<absl::Cord> payload = status.GetPayload(kStatusPayloadName);
  if (payload.has_value()) {
    StatusPayloadProto proto;
    if (proto.ParseFromString(payload->Flatten())) {
      return proto;
    }
  }
  return std::nullopt;
}

}  // namespace xls::dslx
