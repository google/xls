// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system/channel_manager.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"

namespace xls::dslx {

const StructMemberNode* ImplicitChannelDecl::GetNamedMember() const {
  const auto* sink_member = absl::down_cast<const StructMemberNode*>(
      GetChannelOrArrayDefiner(out_end));
  const auto* source_member = absl::down_cast<const StructMemberNode*>(
      GetChannelOrArrayDefiner(in_end));
  // Prefer `sink_member`, unless only `source_member` belongs to `caller`.
  if (caller->GetMemberByName(sink_member->name()) != sink_member &&
      caller->GetMemberByName(source_member->name()) == source_member) {
    return source_member;
  }
  return sink_member;
}

const InterpValue& ForwardedChannel::GetCallerEnd() const {
  return GetChannelOrArrayDirection(from_end) == ChannelDirection::kIn
             ? from_end
             : to_end;
}

absl::Status ChannelManager::CheckEndpointAvailable(
    const ProcDef* caller, int64_t id, const Expr* expr,
    const Invocation* origin, const FileTable& file_table) const {
  if (auto it = endpoint_to_decl_.find(std::make_pair(caller, id));
      it != endpoint_to_decl_.end()) {
    return TypeInferenceErrorStatus(
        origin->span(), nullptr,
        absl::Substitute("`$0` is already bound by `$1`.", expr->ToString(),
                         it->second->origin->ToString()),
        file_table);
  }
  if (auto it = endpoint_to_forward_.find(std::make_pair(caller, id));
      it != endpoint_to_forward_.end()) {
    return TypeInferenceErrorStatus(
        origin->span(), nullptr,
        absl::Substitute("`$0` is already forwarded by `$1`.", expr->ToString(),
                         it->second->origin->ToString()),
        file_table);
  }
  return absl::OkStatus();
}

absl::Status ChannelManager::Bind(const ProcDef* caller,
                                  const InterpValue& out_end,
                                  const InterpValue& in_end,
                                  const Expr* out_expr, const Expr* in_expr,
                                  const Invocation* origin,
                                  const FileTable& file_table) {
  int64_t out_id = GetChannelOrArrayId(out_end);
  int64_t in_id = GetChannelOrArrayId(in_end);

  // Error if either channel is already bound or forwarded.
  for (const auto& [id, expr] :
       {std::make_pair(out_id, out_expr), std::make_pair(in_id, in_expr)}) {
    XLS_RETURN_IF_ERROR(
        CheckEndpointAvailable(caller, id, expr, origin, file_table));
  }

  auto decl = std::make_shared<ImplicitChannelDecl>(
      ImplicitChannelDecl{.out_end = out_end,
                          .in_end = in_end,
                          .caller = caller,
                          .origin = origin});
  endpoint_to_decl_.emplace(std::make_pair(caller, out_id), decl);
  endpoint_to_decl_.emplace(std::make_pair(caller, in_id), std::move(decl));
  return absl::OkStatus();
}

absl::Status ChannelManager::Forward(const ProcDef* caller,
                                     const InterpValue& from_end,
                                     const InterpValue& to_end,
                                     const Expr* from_expr, const Expr* to_expr,
                                     const Invocation* origin,
                                     const FileTable& file_table) {
  int64_t from_id = GetChannelOrArrayId(from_end);
  int64_t to_id = GetChannelOrArrayId(to_end);

  const auto* from_member = absl::down_cast<const StructMemberNode*>(
      GetChannelOrArrayDefiner(from_end));
  const auto* to_member = absl::down_cast<const StructMemberNode*>(
      GetChannelOrArrayDefiner(to_end));
  bool from_is_caller_member =
      caller->GetMemberByName(from_member->name()) == from_member;
  bool to_is_caller_member =
      caller->GetMemberByName(to_member->name()) == to_member;

  const bool is_source =
      GetChannelOrArrayDirection(from_end) == ChannelDirection::kIn;
  if (is_source) {
    if (!from_is_caller_member || to_is_caller_member) {
      return TypeInferenceErrorStatus(
          origin->span(), nullptr,
          absl::Substitute("`forward` on a `Source` must forward from a member "
                           "of `$0` to a child proc's `Source`; got `$1` to "
                           "`$2`.",
                           caller->identifier(), from_expr->ToString(),
                           to_expr->ToString()),
          file_table);
    }
  } else if (from_is_caller_member || !to_is_caller_member) {
    return TypeInferenceErrorStatus(
        origin->span(), nullptr,
        absl::Substitute("`forward` on a `Sink` must forward from a child "
                         "proc's `Sink` to a member of `$0`; got `$1` to "
                         "`$2`.",
                         caller->identifier(), from_expr->ToString(),
                         to_expr->ToString()),
        file_table);
  }

  for (const auto& [id, expr] :
       {std::make_pair(from_id, from_expr), std::make_pair(to_id, to_expr)}) {
    XLS_RETURN_IF_ERROR(
        CheckEndpointAvailable(caller, id, expr, origin, file_table));
  }

  auto forwarded = std::make_shared<ForwardedChannel>(ForwardedChannel{
      .from_end = from_end, .to_end = to_end, .origin = origin});
  endpoint_to_forward_.emplace(std::make_pair(caller, from_id), forwarded);
  endpoint_to_forward_.emplace(std::make_pair(caller, to_id),
                               std::move(forwarded));
  return absl::OkStatus();
}

std::optional<const ImplicitChannelDecl*>
ChannelManager::GetImplicitChannelDecl(const ProcDef* caller,
                                       const InterpValue& endpoint) const {
  if (!endpoint.IsChannelReference()) {
    return std::nullopt;
  }
  int64_t channel_id = GetChannelOrArrayId(endpoint);
  const auto it = endpoint_to_decl_.find(std::make_pair(caller, channel_id));
  if (it == endpoint_to_decl_.end()) {
    return std::nullopt;
  }
  return it->second.get();
}

std::optional<const ForwardedChannel*> ChannelManager::GetForwardedChannel(
    const ProcDef* caller, const InterpValue& endpoint) const {
  if (!endpoint.IsChannelReference()) {
    return std::nullopt;
  }
  int64_t channel_id = GetChannelOrArrayId(endpoint);
  const auto it = endpoint_to_forward_.find(std::make_pair(caller, channel_id));
  if (it == endpoint_to_forward_.end()) {
    return std::nullopt;
  }
  return it->second.get();
}

}  // namespace xls::dslx
