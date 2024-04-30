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

#include "xls/noc/config_ng/tree_options_proto_builder.h"

#include <cstdint>
#include <vector>

#include "absl/log/die_if_null.h"

namespace xls::noc {

TreeOptionsProtoBuilder::TreeOptionsProtoBuilder(TreeOptionsProto* proto_ptr)
    : proto_ptr_(ABSL_DIE_IF_NULL(proto_ptr)) {}

TreeOptionsProtoBuilder::TreeOptionsProtoBuilder(
    TreeOptionsProto* proto_ptr, const TreeOptionsProto& default_proto)
    : TreeOptionsProtoBuilder(proto_ptr) {
  *proto_ptr_ = default_proto;
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::CopyFrom(
    const TreeOptionsProtoBuilder& builder) {
  *proto_ptr_ = *builder.proto_ptr_;
  return *this;
}

LevelOptionsProtoBuilder TreeOptionsProtoBuilder::AddLevel() {
  return LevelOptionsProtoBuilder(proto_ptr_->add_levels());
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::AddLevel(
    const LevelOptionsProtoBuilder& builder) {
  LevelOptionsProtoBuilder(proto_ptr_->add_levels()).CopyFrom(builder);
  return *this;
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::AddLevels(
    absl::Span<const int64_t> levels) {
  for (int64_t index = 0; index < levels.size(); index++) {
    this->AddLevel().SetIndex(index).SetNodeCount(levels[index]);
  }
  return *this;
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::EnableUnidirectional(
    const UnidirectionalTreeOptionsProtoBuilder& builder) {
  UnidirectionalTreeOptionsProtoBuilder(
      proto_ptr_->mutable_unidirectional_tree())
      .CopyFrom(builder);
  return *this;
}

UnidirectionalTreeOptionsProtoBuilder
TreeOptionsProtoBuilder::EnableUnidirectional() {
  return UnidirectionalTreeOptionsProtoBuilder(
      proto_ptr_->mutable_unidirectional_tree());
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::EnableBidirectional(
    const BidirectionalTreeOptionsProtoBuilder& builder) {
  BidirectionalTreeOptionsProtoBuilder(proto_ptr_->mutable_bidirectional_tree())
      .CopyFrom(builder);
  return *this;
}

BidirectionalTreeOptionsProtoBuilder
TreeOptionsProtoBuilder::EnableBidirectional() {
  return BidirectionalTreeOptionsProtoBuilder(
      proto_ptr_->mutable_bidirectional_tree());
}

TreeOptionsProtoBuilder&
TreeOptionsProtoBuilder::BidirectionalBinaryTreeTopology(
    const int64_t level_count, const int64_t send_port_count_at_root,
    const int64_t recv_port_count_at_root) {
  this->AddLevels(std::vector<int64_t>(level_count, 2));
  this->EnableBidirectional()
      .SetSendPortCountAtRoot(send_port_count_at_root)
      .SetRecvPortCountAtRoot(recv_port_count_at_root);
  return *this;
}

TreeOptionsProtoBuilder&
TreeOptionsProtoBuilder::UnidirectionalBinaryTreeTopology(
    const int64_t level_count, const int64_t port_count_at_root,
    const UnidirectionalTreeOptionsProto::Type type) {
  this->AddLevels(std::vector<int64_t>(level_count, 2));
  this->EnableUnidirectional().SetType(type).SetPortCountAtRoot(
      port_count_at_root);
  return *this;
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::AggregationTreeTopology(
    absl::Span<const int64_t> radix_list, const int64_t port_count_at_root) {
  this->AddLevels(radix_list);
  this->EnableUnidirectional().AsAggregation().SetPortCountAtRoot(
      port_count_at_root);
  return *this;
}

TreeOptionsProtoBuilder& TreeOptionsProtoBuilder::DistributionTreeTopology(
    absl::Span<const int64_t> radix_list, const int64_t port_count_at_root) {
  this->AddLevels(radix_list);
  this->EnableUnidirectional().AsDistribution().SetPortCountAtRoot(
      port_count_at_root);
  return *this;
}

}  // namespace xls::noc
