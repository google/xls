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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

absl::Status ChannelManager::Bind(const ProcDef* caller,
                                  const InterpValue& out_end,
                                  const InterpValue& in_end,
                                  const Expr* out_expr, const Expr* in_expr,
                                  const Invocation* origin,
                                  const FileTable& file_table) {
  int64_t out_id = *out_end.GetChannelReferenceOrDie().GetChannelId();
  int64_t in_id = *in_end.GetChannelReferenceOrDie().GetChannelId();

  auto it = endpoint_to_decl_.find(std::make_pair(caller, out_id));
  if (it != endpoint_to_decl_.end()) {
    const ImplicitChannelDecl& existing =
        decls_by_caller_.at(caller)[it->second];
    if (*existing.out_end.GetChannelReferenceOrDie().GetChannelId() == out_id &&
        *existing.in_end.GetChannelReferenceOrDie().GetChannelId() == in_id) {
      return absl::OkStatus();
    }
    return TypeInferenceErrorStatus(
        origin->span(), nullptr,
        absl::Substitute("`$0` is already bound by `$1`.", out_expr->ToString(),
                         existing.origin->ToString()),
        file_table);
  }

  it = endpoint_to_decl_.find(std::make_pair(caller, in_id));
  if (it != endpoint_to_decl_.end()) {
    const ImplicitChannelDecl& existing =
        decls_by_caller_.at(caller)[it->second];
    return TypeInferenceErrorStatus(
        origin->span(), nullptr,
        absl::Substitute("`$0` is already bound by `$1`.", in_expr->ToString(),
                         existing.origin->ToString()),
        file_table);
  }

  std::vector<ImplicitChannelDecl>& decls = decls_by_caller_[caller];
  size_t index = decls.size();
  decls.push_back(ImplicitChannelDecl{
      .out_end = out_end, .in_end = in_end, .origin = origin});
  endpoint_to_decl_.emplace(std::make_pair(caller, out_id), index);
  endpoint_to_decl_.emplace(std::make_pair(caller, in_id), index);
  return absl::OkStatus();
}

std::optional<const ImplicitChannelDecl*>
ChannelManager::GetImplicitChannelDecl(const ProcDef* caller,
                                       int64_t channel_id) const {
  const auto it = endpoint_to_decl_.find(std::make_pair(caller, channel_id));
  if (it == endpoint_to_decl_.end()) {
    return std::nullopt;
  }
  return &decls_by_caller_.at(caller)[it->second];
}

absl::Span<const ImplicitChannelDecl>
ChannelManager::GetImplicitChannelDeclsFrom(const ProcDef* caller) const {
  const auto it = decls_by_caller_.find(caller);
  if (it == decls_by_caller_.end()) {
    return {};
  }
  return it->second;
}

}  // namespace xls::dslx
