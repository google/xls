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

#include "xls/data_structures/transitive_closure.h"

#include <bit>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {
namespace {

InlineBitmap ReachableFromInternal(InlineBitmap reached,
                                   std::vector<int64_t> worklist,
                                   absl::Span<const InlineBitmap> adj) {
  while (!worklist.empty()) {
    int64_t i = worklist.back();
    worklist.pop_back();
    if (i >= adj.size()) {
      continue;
    }
    const InlineBitmap& neighbors = adj[i];
    CHECK_EQ(neighbors.bit_count(), reached.bit_count());
    for (int64_t word_idx = 0; word_idx < neighbors.word_count(); ++word_idx) {
      uint64_t word = neighbors.GetWord(word_idx);
      uint64_t current_reached = reached.GetWord(word_idx);
      word &= ~current_reached;
      if (word != 0) {
        reached.SetWord(word_idx, current_reached | word);
      }
      while (word != 0) {
        int64_t bit_idx = std::countr_zero(word);
        worklist.push_back(word_idx * 64 + bit_idx);
        word &= (word - 1);
      }
    }
  }
  return reached;
}

}  // namespace

InlineBitmap ReachableFrom(absl::Span<const int64_t> starting_nodes,
                           absl::Span<const InlineBitmap> adj) {
  for (const InlineBitmap& bitmaps : adj) {
    CHECK_EQ(bitmaps.bit_count(), adj.size());
  }
  InlineBitmap reached(adj.size());
  std::vector<int64_t> worklist;
  worklist.reserve(starting_nodes.size());
  for (int64_t node : starting_nodes) {
    if (node < adj.size() && !reached.Get(node)) {
      reached.Set(node);
      worklist.push_back(node);
    }
  }
  return ReachableFromInternal(std::move(reached), std::move(worklist), adj);
}

InlineBitmap ReachableFrom(const absl::flat_hash_set<int64_t>& starting_nodes,
                           absl::Span<const InlineBitmap> adj) {
  for (const InlineBitmap& bitmaps : adj) {
    CHECK_EQ(bitmaps.bit_count(), adj.size());
  }
  InlineBitmap reached(adj.size());
  std::vector<int64_t> worklist;
  worklist.reserve(starting_nodes.size());
  for (int64_t node : starting_nodes) {
    if (node < adj.size() && !reached.Get(node)) {
      reached.Set(node);
      worklist.push_back(node);
    }
  }
  return ReachableFromInternal(std::move(reached), std::move(worklist), adj);
}

InlineBitmap ReachableFrom(const InlineBitmap& starting_nodes,
                           absl::Span<const InlineBitmap> adj) {
  for (const InlineBitmap& bitmaps : adj) {
    CHECK_EQ(bitmaps.bit_count(), adj.size());
  }
  CHECK_EQ(starting_nodes.bit_count(), adj.size());
  InlineBitmap reached = starting_nodes;
  std::vector<int64_t> worklist;
  for (int64_t word_idx = 0; word_idx < reached.word_count(); ++word_idx) {
    uint64_t word = reached.GetWord(word_idx);
    while (word != 0) {
      int64_t bit_idx = std::countr_zero(word);
      worklist.push_back(word_idx * 64 + bit_idx);
      word &= (word - 1);
    }
  }
  return ReachableFromInternal(std::move(reached), std::move(worklist), adj);
}

}  // namespace xls
