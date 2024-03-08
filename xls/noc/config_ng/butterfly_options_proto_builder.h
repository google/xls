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

#ifndef XLS_NOC_CONFIG_NG_BUTTERFLY_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_NG_BUTTERFLY_OPTIONS_PROTO_BUILDER_H_

#include "xls/noc/config_ng/bidirectional_butterfly_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"
#include "xls/noc/config_ng/unidirectional_butterfly_options_proto_builder.h"

namespace xls::noc {

// A builder to aid in constructing a butterfly options proto.
class ButterflyOptionsProtoBuilder {
 public:
  // Constructor storing the proto pointer as a class member.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  explicit ButterflyOptionsProtoBuilder(ButterflyOptionsProto* proto_ptr);

  // Constructor storing the proto pointer as a class member and sets the fields
  // of the proto to default_proto.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  ButterflyOptionsProtoBuilder(ButterflyOptionsProto* proto_ptr,
                               const ButterflyOptionsProto& default_proto);

  // Copy the field values of the proto message object from another builder
  // to the builder.
  ButterflyOptionsProtoBuilder& CopyFrom(
      const ButterflyOptionsProtoBuilder& builder);

  // Sets the radix per stage of the butterfly options.
  ButterflyOptionsProtoBuilder& SetRadixPerStage(int64_t radix_per_stage);

  // Sets the stage count of the butterfly options.
  ButterflyOptionsProtoBuilder& SetStageCount(int64_t stage_count);

  // Sets the flatten flag of the butterfly options.
  ButterflyOptionsProtoBuilder& SetFlatten(bool flatten);

  // Enable the unidirectional type of the butterfly options.
  ButterflyOptionsProtoBuilder& EnableUnidirectional(
      const UnidirectionalButterflyOptionsProtoBuilder& builder);
  UnidirectionalButterflyOptionsProtoBuilder EnableUnidirectional();

  // Enable the bidirectional type of the butterfly options.
  ButterflyOptionsProtoBuilder& EnableBidirectional(
      const BidirectionalButterflyOptionsProtoBuilder& builder);
  BidirectionalButterflyOptionsProtoBuilder EnableBidirectional();

  // A unidirectional butterfly topology with a user-defined number of stages,
  // the number radix per stage, and the direction of the communication flow.
  // The topology can be described as a conventional butterfly topology or
  // flattened butterfly topology.
  ButterflyOptionsProtoBuilder& UnidirectionalButterflyTopology(
      int64_t stage_count, int64_t radix_per_stage,
      UnidirectionalButterflyOptionsProto::Flow flow, bool flatten = false);

  // A bidirectional butterfly topology with a user-defined number of stages,
  // the number radix per stage, and the location of the endpoint nodes are
  // connected to. The topology can be described as a conventional butterfly
  // topology or flattened butterfly topology.
  ButterflyOptionsProtoBuilder& BidirectionalButterflyTopology(
      int64_t stage_count, int64_t radix_per_stage,
      BidirectionalButterflyOptionsProto::EndpointConnection
          endpoint_connection,
      bool flatten = false);

 private:
  ButterflyOptionsProto* proto_ptr_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_BUTTERFLY_OPTIONS_PROTO_BUILDER_H_
