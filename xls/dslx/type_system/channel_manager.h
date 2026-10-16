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

#ifndef XLS_DSLX_TYPE_SYSTEM_CHANNEL_MANAGER_H_
#define XLS_DSLX_TYPE_SYSTEM_CHANNEL_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// A channel declared implicitly in a proc constructor, rather than written
// as `chan<T>(...)`. For example, `sink.bind(source)` declares one channel
// whose out end is `sink` and whose in end is `source`, storing both
// endpoint `ChannelReference` InterpValues like `HandleChannelDecl`'s
// `(out_end, in_end)` tuple.
struct ImplicitChannelDecl {
  InterpValue out_end;
  InterpValue in_end;
  // The call that declared the channel, e.g. the `bind` invocation.
  const Invocation* origin;
};

class ChannelManager {
 public:
  // Records `sink.bind(source)` inside `caller`.
  // - No-op if the exact `(out_end, in_end)` pair is already recorded.
  // - Returns a `TypeInferenceErrorStatus` if either endpoint is already
  //   bound to a different peer.
  absl::Status Bind(const ProcDef* caller, const InterpValue& out_end,
                    const InterpValue& in_end, const Expr* out_expr,
                    const Expr* in_expr, const Invocation* origin,
                    const FileTable& file_table);

  // Returns the `ImplicitChannelDecl` that `channel_id` belongs to (as
  // either `out_end` or `in_end`) within `caller`, or `std::nullopt` if
  // unbound.
  std::optional<const ImplicitChannelDecl*> GetImplicitChannelDecl(
      const ProcDef* caller, int64_t channel_id) const;

  // Returns all implicit channels declared in `caller`, in declaration
  // order.
  absl::Span<const ImplicitChannelDecl> GetImplicitChannelDeclsFrom(
      const ProcDef* caller) const;

 private:
  absl::flat_hash_map<const ProcDef*, std::vector<ImplicitChannelDecl>>
      decls_by_caller_;
  // Maps (caller, endpoint channel_id) -> index in decls_by_caller_[caller].
  absl::flat_hash_map<std::pair<const ProcDef*, int64_t>, size_t>
      endpoint_to_decl_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_CHANNEL_MANAGER_H_
