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

#include "xls/noc/config/custom_network_config_builder_options_proto_builder.h"

namespace xls::noc {

UnidirectionalTreeNetworkConfigOptionsProtoBuilder&
UnidirectionalTreeNetworkConfigOptionsProtoBuilder::WithRadix(int64_t radix) {
  proto_->set_radix(radix);
  return *this;
}

UnidirectionalTreeNetworkConfigOptionsProtoBuilder&
UnidirectionalTreeNetworkConfigOptionsProtoBuilder::AsDistributionTree() {
  proto_->set_type(UnidirectionalTreeNetworkConfigOptionsProto::DISTRIBUTION);
  return *this;
}

UnidirectionalTreeNetworkConfigOptionsProtoBuilder&
UnidirectionalTreeNetworkConfigOptionsProtoBuilder::AsAggregationTree() {
  proto_->set_type(UnidirectionalTreeNetworkConfigOptionsProto::AGGREGATION);
  return *this;
}

}  // namespace xls::noc
