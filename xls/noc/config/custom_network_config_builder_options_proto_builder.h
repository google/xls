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

// Builder definitions for the custom network config options proto.

#ifndef XLS_NOC_CONFIG_CUSTOM_NETWORK_CONFIG_BUILDER_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_CUSTOM_NETWORK_CONFIG_BUILDER_OPTIONS_PROTO_BUILDER_H_

#include <cstdint>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/network_config_builder_options.pb.h"

namespace xls::noc {

// A fluent builder for constructing a grid network config options proto.
class GridNetworkConfigOptionsProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit GridNetworkConfigOptionsProtoBuilder(
      GridNetworkConfigOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the number of rows.
  GridNetworkConfigOptionsProtoBuilder& WithNumRows(int64_t num_rows);

  // Adds the number of columns.
  GridNetworkConfigOptionsProtoBuilder& WithNumColumns(int64_t num_columns);

  // Adds the row loopback.
  GridNetworkConfigOptionsProtoBuilder& WithRowLoopback(bool row_loopback);

  // Adds the column loopback.
  GridNetworkConfigOptionsProtoBuilder& WithColumnLoopback(
      bool column_loopback);

 private:
  GridNetworkConfigOptionsProto* proto_;
};

// A fluent builder for constructing a unidirectional tree network config
// options proto.
class UnidirectionalTreeNetworkConfigOptionsProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit UnidirectionalTreeNetworkConfigOptionsProtoBuilder(
      UnidirectionalTreeNetworkConfigOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the maximum number of input ports for each router.
  UnidirectionalTreeNetworkConfigOptionsProtoBuilder& WithRadix(int64_t radix);

  // Sets the tree type to distribution.
  UnidirectionalTreeNetworkConfigOptionsProtoBuilder& AsDistributionTree();

  // Sets the tree type to aggregation.
  UnidirectionalTreeNetworkConfigOptionsProtoBuilder& AsAggregationTree();

 private:
  UnidirectionalTreeNetworkConfigOptionsProto* proto_;
};

// A fluent builder for constructing a bidirectional tree network config options
// proto.
class BidirectionalTreeNetworkConfigOptionsProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit BidirectionalTreeNetworkConfigOptionsProtoBuilder(
      BidirectionalTreeNetworkConfigOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the maximum number of input ports for each router.
  BidirectionalTreeNetworkConfigOptionsProtoBuilder& WithRadix(int64_t radix);

  // Adds the maximum number of send ports connected at the root.
  BidirectionalTreeNetworkConfigOptionsProtoBuilder& WithNumSendPortsAtRoot(
      int64_t num_send_ports_at_root);

  // Adds the maximum number of receive ports connected at the root.
  BidirectionalTreeNetworkConfigOptionsProtoBuilder& WithNumRecvPortsAtRoot(
      int64_t num_recv_ports_at_root);

 private:
  BidirectionalTreeNetworkConfigOptionsProto* proto_;
};

// A fluent builder for constructing a fully connected network config options
// proto.
class FullyConnectedNetworkConfigOptionsProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit FullyConnectedNetworkConfigOptionsProtoBuilder(
      FullyConnectedNetworkConfigOptionsProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

 private:
  FullyConnectedNetworkConfigOptionsProto* proto_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_CUSTOM_NETWORK_CONFIG_BUILDER_OPTIONS_PROTO_BUILDER_H_
