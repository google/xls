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

#include "xls/passes/pass_base.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"

namespace xls {

std::string_view RamKindToString(RamKind kind) {
  switch (kind) {
    case RamKind::kAbstract:
      return "abstract";
    case RamKind::k1RW:
      return "1rw";
    case RamKind::k1R1W:
      return "1r1w";
    case RamKind::k2RW:
      return "2rw";
  }
}

int64_t RamConfig::addr_width() const {
  XLS_CHECK_GE(depth, 0);
  return CeilOfLog2(static_cast<uint64_t>(depth));
}

std::optional<int64_t> RamConfig::mask_width(int64_t data_width) const {
  if (!word_partition_size.has_value()) {
    return std::nullopt;
  }

  XLS_CHECK_GT(word_partition_size.value(), 0);
  return (data_width + word_partition_size.value() - 1) /
         word_partition_size.value();
}

absl::StatusOr<RamKind> RamKindFromProto(RamKindProto proto) {
  switch (proto) {
    case RamKindProto::RAM_ABSTRACT:
      return RamKind::kAbstract;
    case RamKindProto::RAM_1RW:
      return RamKind::k1RW;
    case RamKindProto::RAM_1R1W:
      return RamKind::k1R1W;
    default:
      return absl::InvalidArgumentError("Invalid RamKind");
  }
}

/*static*/ absl::StatusOr<RamConfig> RamConfig::FromProto(
    const RamConfigProto& proto) {
  XLS_ASSIGN_OR_RETURN(RamKind kind, RamKindFromProto(proto.kind()));
  return RamConfig{
      .kind = kind,
      .depth = proto.depth(),
      .word_partition_size = proto.has_word_partition_size()
                                 ? std::optional(proto.word_partition_size())
                                 : std::nullopt,
      // TODO(google/xls#861): Add support for initialization info in proto.
      .initial_value = std::nullopt,
  };
}

/*static*/ absl::StatusOr<RamRewrite> RamRewrite::FromProto(
    const RamRewriteProto& proto) {
  if (proto.has_model_builder()) {
    return absl::UnimplementedError("Model builders not yet implemented.");
  }
  XLS_ASSIGN_OR_RETURN(RamConfig from_config,
                       RamConfig::FromProto(proto.from_config()));
  XLS_ASSIGN_OR_RETURN(RamConfig to_config,
                       RamConfig::FromProto(proto.to_config()));
  return RamRewrite{
      .from_config = from_config,
      .from_channels_logical_to_physical =
          absl::flat_hash_map<std::string, std::string>(
              proto.from_channels_logical_to_physical().begin(),
              proto.from_channels_logical_to_physical().end()),
      .to_config = to_config,
      .to_name_prefix = proto.to_name_prefix(),
      .model_builder = std::nullopt,
  };
}

absl::StatusOr<std::vector<RamRewrite>> RamRewritesFromProto(
    const RamRewritesProto& proto) {
  std::vector<RamRewrite> rewrites;
  rewrites.reserve(proto.rewrites_size());
  for (const auto& rewrite_proto : proto.rewrites()) {
    XLS_ASSIGN_OR_RETURN(RamRewrite rewrite,
                         RamRewrite::FromProto(rewrite_proto));
    rewrites.push_back(rewrite);
  }
  return rewrites;
}

}  // namespace xls
