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

#ifndef XLS_NOC_CONFIG_NG_TREE_OPTIONS_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_NG_TREE_OPTIONS_PROTO_BUILDER_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xls/noc/config_ng/bidirectional_tree_options_proto_builder.h"
#include "xls/noc/config_ng/level_options_proto_builder.h"
#include "xls/noc/config_ng/topology_options_network_config_builder.pb.h"
#include "xls/noc/config_ng/unidirectional_tree_options_proto_builder.h"

namespace xls::noc {

// A builder to aid in constructing a tree options proto.
class TreeOptionsProtoBuilder {
 public:
  // Constructor storing the proto pointer as a class member.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  explicit TreeOptionsProtoBuilder(TreeOptionsProto* proto_ptr);

  // Constructor storing the proto pointer as a class member and sets the fields
  // of the proto to default_proto.
  // proto_ptr cannot be nullptr. Does not take ownership of the proto_ptr. The
  // proto_ptr must refer to a valid object that outlives this object.
  TreeOptionsProtoBuilder(TreeOptionsProto* proto_ptr,
                          const TreeOptionsProto& default_proto);

  // Copy the field values of the proto message object from another builder
  // to the builder.
  TreeOptionsProtoBuilder& CopyFrom(const TreeOptionsProtoBuilder& builder);

  // Add a level to the tree options.
  LevelOptionsProtoBuilder AddLevel();

  // Add a level to the tree options.
  TreeOptionsProtoBuilder& AddLevel(const LevelOptionsProtoBuilder& builder);

  // Add the levels to the tree options.
  // levels: The list of levels to append. Each element of the list
  // corresponds to a level. The index of element in the list represents the
  // index of the level. The value of the element at a given index
  // represents the number of nodes at the given level.
  TreeOptionsProtoBuilder& AddLevels(absl::Span<const int64_t> levels);

  // Enable the unidirectional type of the tree options.
  TreeOptionsProtoBuilder& EnableUnidirectional(
      const UnidirectionalTreeOptionsProtoBuilder& builder);
  UnidirectionalTreeOptionsProtoBuilder EnableUnidirectional();

  // Enable the bidirectional type of the tree options.
  TreeOptionsProtoBuilder& EnableBidirectional(
      const BidirectionalTreeOptionsProtoBuilder& builder);
  BidirectionalTreeOptionsProtoBuilder EnableBidirectional();

  // TODO(vmirian) 03-30-21 add bidirectional tree with radix list

  // A bidirectional binary tree topology with a user-defined number of levels,
  // the number of send ports and receive ports at the root of the tree. The
  // number of levels, the number of send ports and receive ports at the root of
  // the tree must be greater than zero. The number of endpoints at the final
  // level of the tree is equal to 2 to the power of the number of levels.
  TreeOptionsProtoBuilder& BidirectionalBinaryTreeTopology(
      int64_t level_count, int64_t send_port_count_at_root,
      int64_t recv_port_count_at_root);

  // A unidirectional binary tree topology with a user-defined number of levels,
  // the number of ports at the root of the tree, and the flow of the
  // communication. The number of levels, and the number of ports at the root
  // of the tree must be greater than zero. The number of endpoints at the final
  // level of the tree is equal to 2 to the power of the number of levels.
  TreeOptionsProtoBuilder& UnidirectionalBinaryTreeTopology(
      int64_t level_count, int64_t port_count_at_root,
      UnidirectionalTreeOptionsProto::Type type);

  // An aggregation tree with the radix of each level described within a list.
  // The value at the index i of the list defines the radix at level i. The
  // radix value and the number of ports at the root must be greater than zero.
  TreeOptionsProtoBuilder& AggregationTreeTopology(
      absl::Span<const int64_t> radix_list, int64_t port_count_at_root);

  // An distribution tree with the radix of each level described within a list.
  // The value at the index i of the list defines the radix at level i. The
  // radix value and the number of ports at the root must be greater than zero.
  TreeOptionsProtoBuilder& DistributionTreeTopology(
      absl::Span<const int64_t> radix_list, int64_t port_count_at_root);

 private:
  TreeOptionsProto* proto_ptr_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_TREE_OPTIONS_PROTO_BUILDER_H_
