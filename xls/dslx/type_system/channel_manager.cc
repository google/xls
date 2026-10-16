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

#include "absl/status/status.h"
#include "absl/strings/substitute.h"
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

  // Error if either channel is already bound.
  for (const auto& [id, expr] :
       {std::make_pair(out_id, out_expr), std::make_pair(in_id, in_expr)}) {
    if (auto it = endpoint_to_decl_.find(std::make_pair(caller, id));
        it != endpoint_to_decl_.end()) {
      return TypeInferenceErrorStatus(
          origin->span(), nullptr,
          absl::Substitute("`$0` is already bound by `$1`.", expr->ToString(),
                           it->second->origin->ToString()),
          file_table);
    }
  }

  auto decl = std::make_shared<ImplicitChannelDecl>(ImplicitChannelDecl{
      .out_end = out_end, .in_end = in_end, .origin = origin});
  endpoint_to_decl_.emplace(std::make_pair(caller, out_id), decl);
  endpoint_to_decl_.emplace(std::make_pair(caller, in_id), std::move(decl));
  return absl::OkStatus();
}

std::optional<const ImplicitChannelDecl*>
ChannelManager::GetImplicitChannelDecl(const ProcDef* caller,
                                       const InterpValue& endpoint) const {
  if (!endpoint.IsChannelReference()) {
    return std::nullopt;
  }
  int64_t channel_id = *endpoint.GetChannelReferenceOrDie().GetChannelId();
  const auto it = endpoint_to_decl_.find(std::make_pair(caller, channel_id));
  if (it == endpoint_to_decl_.end()) {
    return std::nullopt;
  }
  return it->second.get();
}

}  // namespace xls::dslx
