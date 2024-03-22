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

#include "xls/codegen/register_chaining_analysis.h"

#include <deque>
#include <iterator>
#include <list>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"

namespace xls::verilog {

namespace {
bool IsChainable(const RegisterData& first, const RegisterData& next) {
  if (first.read_stage != next.write_stage) {
    return false;
  }
  if (next.write_stage >= next.read_stage) {
    // next is a state loopback register. These may only be at the start of a
    // chain.
    return false;
  }
  if (next.write->data() != first.read) {
    return false;
  }
  return true;
}

// Would a write at the end of the chain happen after the start of the chain is
// already updated?
bool IsClobbered(const RegisterData& chain_start,
                 const RegisterData& chain_end) {
  if (chain_start.read_stage <= chain_start.write_stage) {
    // The chain starts with a loopback (write is in a later stage then read).
    return chain_end.write_stage >= chain_start.write_stage;
  }
  // Chain does not start with a loopback so no way to clobber the register
  // value.
  return false;
}

// Insert the register into a chain, return the chain that was inserted into
std::list<std::deque<RegisterData>>::iterator InsertIntoChain(
    std::list<std::deque<RegisterData>>& chains, const RegisterData& reg) {
  for (auto it = chains.begin(); it != chains.end(); ++it) {
    if (IsChainable(reg, it->front()) && !IsClobbered(reg, it->back())) {
      // Can put it at the front of this.
      it->push_front(reg);
      return it;
    }
    if (IsChainable(it->back(), reg) && !IsClobbered(it->front(), reg)) {
      // Can put it at the back of this chain.
      it->push_back(reg);
      return it;
    }
  }
  // No compatible chain found.
  chains.emplace_front().push_back(reg);
  return chains.begin();
}

void ReduceChains(std::list<std::deque<RegisterData>>& chains,
                  std::list<std::deque<RegisterData>>::iterator modified_entry,
                  bool is_front_modified) {
  // TODO(allight): b/c of the way we add stuff to chains we can skip to
  // modified entry in the iteration probably. Needs some more thought. Rather
  // small optimization in any case.
  for (auto it = chains.begin(); it != chains.end(); ++it) {
    if (it == modified_entry) {
      // can't merge with ourself.
      continue;
    }
    if (is_front_modified) {
      // Want to perform `(merge it modified_entry)`
      if (IsChainable(it->back(), modified_entry->front()) &&
          !IsClobbered(it->front(), modified_entry->back())) {
        absl::c_copy(*modified_entry, std::back_inserter(*it));
        chains.erase(modified_entry);
        VLOG(2) << "Merged chain now (len: " << it->size()
                << "): " << it->front() << " -> " << it->back();
        return;
      }
    } else {
      // Want to perform `(merge modified_entry it)`
      if (IsChainable(modified_entry->back(), it->front()) &&
          !IsClobbered(modified_entry->front(), it->back())) {
        absl::c_copy(*it, std::back_inserter(*modified_entry));
        chains.erase(it);
        VLOG(2) << "Merged chain now(len: " << modified_entry->size()
                << "): " << modified_entry->front() << " -> "
                << modified_entry->back();
        return;
      }
    }
  }
}

// Splits a chain of registers at the locations required by the register r/w and
// concurrent stage information. This returns a list of all the (non-singleton)
// chains that make up the original chain and are internally mutually exclusive,
// meaning the registers could all be merged.
std::vector<std::vector<RegisterData>> SplitOneChain(
    const std::deque<RegisterData>& chain,
    const ConcurrentStageGroups& groups) {
  std::vector<std::vector<RegisterData>> results;
  // Was this register written from a stage which is not in the mutual exclusive
  // zone?
  //
  // We need to keep the register which crosses into mutual exclusive zone
  // separate since if the pipeline pauses in a stage after the read of this
  // register but still within the mutual exclusive zone then other stages will
  // erroneously see the register as empty and attempt to write into it
  // asynchronously.
  auto is_read_write_concurrent = [&](const RegisterData& cur) {
    return groups.IsConcurrent(cur.write_stage, cur.read_stage);
  };

  // Do cleanup of a potential singleton to start a new chain.
  auto maybe_remove_singleton = [&]() {
    if (!results.empty() && results.back().size() == 1) {
      results.pop_back();
    }
  };

  for (const auto& cur : chain) {
    auto is_mutex_with_current = [&](const RegisterData& v) {
      return groups.IsMutuallyExclusive(v.read_stage, cur.read_stage) &&
             groups.IsMutuallyExclusive(v.write_stage, cur.write_stage);
    };
    // Can't merge up if (1) there's nothing to merge with, (2) the register
    // might be written concurrently, (3) there's some stage in the current
    // group we're not mutex with.
    if (results.empty() || is_read_write_concurrent(cur) ||
        !absl::c_all_of(results.back(), is_mutex_with_current)) {
      maybe_remove_singleton();
      results.push_back({cur});
    } else {
      results.back().push_back(cur);
    }
  }
  maybe_remove_singleton();
  return results;
}

}  // namespace

void RegisterChains::InsertAndReduce(const RegisterData& data) {
  VLOG(2) << "Adding to chain " << data;
  auto modified_entry = InsertIntoChain(chains_, data);
  if (modified_entry->size() == 1) {
    VLOG(2) << "Chain is singleton.";
    // Left as a singleton so nothing to merge (If a merge was possible the
    // singleton would have been put onto that chain instead).
    return;
  }
  VLOG(2) << "Chain now (len: " << modified_entry->size()
          << "): " << modified_entry->front() << " -> "
          << modified_entry->back();

  ReduceChains(chains_, modified_entry,
               /*is_front_modified=*/modified_entry->front() == data);
}

absl::StatusOr<std::vector<std::vector<RegisterData>>>
RegisterChains::SplitBetweenMutexRegions(
    const ConcurrentStageGroups& groups,
    const CodegenPassOptions& options) const {
  VLOG(2) << "Concurrent Groups\n" << groups;
  std::vector<std::vector<RegisterData>> results;
  results.reserve(chains_.size());
  for (const auto& chain : chains_) {
    absl::c_move(SplitOneChain(chain, groups), std::back_inserter(results));
  }
  // NB Technically we could try to reduce the chains further since by removing
  // concurrent elements loopback entries might be gone meaning chains that
  // couldn't merge previously now are able to. However since the loopbacks
  // define the mutex regions in practice no loopback will ever be removed by a
  // mutex region so there's no point in trying to do this.
  return results;
}
}  // namespace xls::verilog
