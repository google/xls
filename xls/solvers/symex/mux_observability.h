// Copyright 2026 The XLS Authors
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

#ifndef XLS_SOLVERS_SYMEX_MUX_OBSERVABILITY_H_
#define XLS_SOLVERS_SYMEX_MUX_OBSERVABILITY_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"

namespace xls::solvers::symex {

// Tracks which multiplexers can still influence a function's observable
// outputs, given the arm choices made so far along one symbolic execution path.
//
// `Z3EncodingVisitor` encodes every multiplexer as an opaque, unconstrained SSA
// variable, so the Z3 expression of a node mentions the variable of multiplexer
// `m` exactly when `m` is reachable from that node without passing through
// another multiplexer. Those are the node's "boundary" multiplexers, and a
// multiplexer matters only if it sits on the boundary of a value that is itself
// observable. Deciding that is graph reachability, so tracking observability
// costs no solver queries.
//
// Observability is seeded from the function's return value and flows
// backwards: choosing arm `k` of multiplexer `m` makes `m`'s selector and arm
// `k` observable, but says nothing about the arms that were not chosen.
// Tracking is therefore *arm sensitive*: the same multiplexer can be
// observable under one arm choice of its consumer and a don't care under
// another.
//
// A multiplexer that is not observable is an *observability don't care*: its
// selector cannot affect the function result on this path, so the engine may
// leave the selector unconstrained instead of enumerating its arms. This
// collapses the exponential blowup from exploring functional units whose
// outputs are discarded on the current path.
//
// `muxes()` is in reverse topological order (consumers before producers), so
// every consumer of a multiplexer has already been decided by the time the
// multiplexer is visited. One sweep is therefore exact and no fixpoint
// iteration is needed. A depth-first traversal follows that order:
//
//   for (Node* mux : observability.muxes()) {
//     if (!observability.IsObservable(mux)) {
//       // Don't care: leave the selector unconstrained and move on.
//       continue;
//     }
//     for (Node* arm : arms_of(mux)) {
//       int64_t token = observability.MarkArmChosen(mux, arm);
//       ...  // recurse
//       observability.Rollback(token);
//     }
//   }
//
// Not thread safe: boundary sets are computed lazily and memoized on first use.
class MuxObservability {
 public:
  // Builds an observability tracker for the `sel` and `priority_sel` nodes of
  // `fn`, seeded with the multiplexers that are observable on every path.
  static absl::StatusOr<MuxObservability> Create(Function* fn);

  // Multiplexers of `fn` in reverse topological order, i.e. consumers before
  // the producers they consume.
  absl::Span<Node* const> muxes() const { return muxes_; }

  // Returns true if `mux` can still affect an observable output given the arm
  // choices recorded so far.
  bool IsObservable(const Node* mux) const { return observable_.contains(mux); }

  // Records that `mux` selects `arm`, which makes `mux`'s selector and `arm`
  // observable. Returns a token to pass to `Rollback` when backtracking out of
  // this choice.
  int64_t MarkArmChosen(Node* mux, Node* arm);

  // Undoes every observability marking made since `token` was handed out.
  void Rollback(int64_t token);

 private:
  MuxObservability() = default;

  // Marks the boundary multiplexers of `node` observable, recording each newly
  // marked multiplexer on the undo trail.
  void MarkObservable(Node* node);

  // Returns the multiplexers reachable from `node` without passing through
  // another multiplexer; `node` itself if it is a multiplexer. Memoized.
  const std::vector<Node*>& BoundaryMuxes(Node* node);

  // Current depth of the undo trail. This doubles as the rollback token, since
  // `Rollback` unwinds the trail back down to the depth it was handed.
  int64_t TrailSize() const { return static_cast<int64_t>(undo_trail_.size()); }

  std::vector<Node*> muxes_;

  // Multiplexers currently known to affect an observable output, plus the
  // stack of markings that `Rollback` undoes.
  absl::flat_hash_set<const Node*> observable_;
  std::vector<const Node*> undo_trail_;

  absl::flat_hash_map<const Node*, std::vector<Node*>> boundary_cache_;
};

}  // namespace xls::solvers::symex

#endif  // XLS_SOLVERS_SYMEX_MUX_OBSERVABILITY_H_
