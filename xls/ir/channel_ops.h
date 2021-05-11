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

#ifndef XLS_IR_CHANNEL_OPS_H_
#define XLS_IR_CHANNEL_OPS_H_

#include <ostream>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xls {

// Indicates the type(s) of operations permitted on the channel. Send-only
// channels can only have send operations (not receive) associated with the
// channel as might be used for communicated to a component outside of
// XLS. Receive-only channels are similarly defined. Send-receive channels can
// have both send and receive operations and can be used for communicated
// between procs.
enum class ChannelOps { kSendOnly, kReceiveOnly, kSendReceive };

std::string ChannelOpsToString(ChannelOps ops);
absl::StatusOr<ChannelOps> StringToChannelOps(absl::string_view str);
std::ostream& operator<<(std::ostream& os, ChannelOps ops);

}  // namespace xls

#endif  // XLS_IR_CHANNEL_OPS_H_
