// Copyright 2020 The XLS Authors
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

#include "xls/noc/config/arbiter_scheme_config_proto_builder.h"

#include <string_view>

#include "absl/types/span.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder_utils.h"

namespace xls::noc {

ArbiterSchemeConfigProtoBuilder&
ArbiterSchemeConfigProtoBuilder::WithPriorityEntry(
    std::string_view output_port_name,
    absl::Span<const PortVirtualChannelTuple> priority_list) {
  RouterConfigProto::ArbiterPriorityEntryConfig* arbiter_priority_entry_config =
      proto_->mutable_priority()->add_entries();
  arbiter_priority_entry_config->set_output_port_name(
      xls::ToProtoString(output_port_name));
  for (const PortVirtualChannelTuple& priority_entry : priority_list) {
    RouterConfigProto::PortVirtualChannelTupleConfig* tuple_config =
        arbiter_priority_entry_config->add_input_port_tuples();
    tuple_config->set_port_name(priority_entry.port_name);
    tuple_config->set_virtual_channel_name(priority_entry.virtual_channel_name);
  }
  return *this;
}

}  // namespace xls::noc
