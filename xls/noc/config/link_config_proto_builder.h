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

#ifndef XLS_NOC_CONFIG_LINK_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_LINK_CONFIG_PROTO_BUILDER_H_

#include <cstdint>
#include <string_view>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

// A builder for constructing a link configuration proto.
class LinkConfigProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit LinkConfigProtoBuilder(LinkConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the name of the link.
  LinkConfigProtoBuilder& WithName(std::string_view name);

  // Adds the source port name configuration to the link.
  LinkConfigProtoBuilder& WithSourcePort(std::string_view port_name);

  // Adds the sink port name configuration to the link.
  LinkConfigProtoBuilder& WithSinkPort(std::string_view port_name);

  // Adds the phit bit width configuration to the link.
  LinkConfigProtoBuilder& WithPhitBitWidth(int64_t phit_bit_width);

  // Adds the pipeline stage configuration for source to sink on the link.
  LinkConfigProtoBuilder& WithSourceSinkPipelineStage(int64_t pipeline_stage);

  // Adds the pipeline stage configuration for sink to source on the link.
  LinkConfigProtoBuilder& WithSinkSourcePipelineStage(int64_t pipeline_stage);

  // When building the link configuration proto, the last flow control mechanism
  // triggered by the user is enabled.

  // Enables the peek flow control configuration to the link, and disables any
  // flow control mechanism scheme enabled.
  LinkConfigProtoBuilder& WithPeekFlowControl();

  // Enables the token credit-based flow control configuration to the link, and
  // disables any flow control mechanism scheme enabled.
  LinkConfigProtoBuilder& WithTokenCreditBasedFlowControl();

  // Enables the total available credit-based flow control configuration to the
  // link, and disables any flow control mechanism scheme enabled.
  LinkConfigProtoBuilder& WithTotalCreditBasedFlowControl(
      int64_t credit_bit_width);

 private:
  LinkConfigProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_LINK_CONFIG_PROTO_BUILDER_H_
