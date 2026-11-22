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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// A channel declared implicitly, stores both endpoint ChannelReference
// InterpValues like HandleChannelDecl's (out_end, in_end) tuple.
struct ImplicitChannelDecl {
  InterpValue out_end;
  InterpValue in_end;
  // The proc that bound these endpoints and owns this channel.
  const ProcDef* caller;
  // The `bind` call that created this channel.
  const Invocation* origin;

  // Returns the endpoint member used to name this channel, preferring the
  // owning proc's member and defaulting to the sink.
  const StructMemberNode* GetNamedMember() const;
};

// A `from.forward(to)` call aliasing a parent proc's boundary `Source` or
// `Sink` with a spawned child's (`parent -> child` for `Source`, `child ->
// parent` for `Sink`).
struct ForwardedChannel {
  InterpValue from_end;
  InterpValue to_end;
  const Invocation* origin;

  // Returns the parent proc's endpoint (`from_end` for `Source`, `to_end` for
  // `Sink`).
  const InterpValue& GetCallerEnd() const;
};

class ChannelManager {
 public:
  // Records `sink.bind(source)` inside `caller`. Returns a
  // `TypeInferenceErrorStatus` if either endpoint is already bound or
  // forwarded.
  absl::Status Bind(const ProcDef* caller, const InterpValue& out_end,
                    const InterpValue& in_end, const Expr* out_expr,
                    const Expr* in_expr, const Invocation* origin,
                    const FileTable& file_table);

  // Records `from.forward(to)` inside `caller`. Returns a
  // `TypeInferenceErrorStatus` if the endpoints do not follow data-flow
  // direction (`parent.source -> child.source` or `child.sink -> parent.sink`)
  // or if either endpoint is already bound or forwarded.
  absl::Status Forward(const ProcDef* caller, const InterpValue& from_end,
                       const InterpValue& to_end, const Expr* from_expr,
                       const Expr* to_expr, const Invocation* origin,
                       const FileTable& file_table);

  // Returns the `ImplicitChannelDecl` that `endpoint` belongs to (as
  // either `out_end` or `in_end`) within `caller`, or `std::nullopt` if
  // unbound.
  std::optional<const ImplicitChannelDecl*> GetImplicitChannelDecl(
      const ProcDef* caller, const InterpValue& endpoint) const;

  // Returns the `ForwardedChannel` that `endpoint` belongs to (as either
  // `from_end` or `to_end`) within `caller`.
  std::optional<const ForwardedChannel*> GetForwardedChannel(
      const ProcDef* caller, const InterpValue& endpoint) const;

 private:
  absl::Status CheckEndpointAvailable(const ProcDef* caller, int64_t id,
                                      const Expr* expr,
                                      const Invocation* origin,
                                      const FileTable& file_table) const;

  absl::flat_hash_map<std::pair<const ProcDef*, int64_t>,
                      std::shared_ptr<ImplicitChannelDecl>>
      endpoint_to_decl_;
  absl::flat_hash_map<std::pair<const ProcDef*, int64_t>,
                      std::shared_ptr<ForwardedChannel>>
      endpoint_to_forward_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_CHANNEL_MANAGER_H_
