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

#include "absl/algorithm/container.h"

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
        return;
      }
    } else {
      // Want to perform `(merge modified_entry it)`
      if (IsChainable(modified_entry->back(), it->front()) &&
          !IsClobbered(modified_entry->front(), it->back())) {
        absl::c_copy(*it, std::back_inserter(*modified_entry));
        chains.erase(it);
        return;
      }
    }
  }
}

}  // namespace

void RegisterChains::InsertAndReduce(const RegisterData& data) {
  auto modified_entry = InsertIntoChain(chains_, data);
  if (modified_entry->size() == 1) {
    // Left as a singleton so nothing to merge (If a merge was possible the
    // singleton would have been put onto that chain instead).
    return;
  }

  ReduceChains(chains_, modified_entry,
               /*is_front_modified=*/modified_entry->front() == data);
}

}  // namespace xls::verilog
