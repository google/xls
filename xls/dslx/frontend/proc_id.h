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

#ifndef XLS_DSLX_FRONTEND_PROC_ID_H_
#define XLS_DSLX_FRONTEND_PROC_ID_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc.h"

namespace xls::dslx {

// ProcId is used to represent a unique instantiation of a Proc, that is, to
// differentiate the instance of Proc Foo spawned from Proc Bar from the one
// spawned from Proc Baz. Each instance can have different member data:
// different constants or channels, so we need to be able to identify each
// separately.
struct ProcId {
  // Contains the "spawn chain": the series of Procs through which this Proc was
  // spawned, with the oldest/"root" proc as element 0.  Contains the current
  // proc, as well. The second element of each pair is a zero-based spawn index
  // of that same proc by the spawning proc. For example, with a spawn chain
  // like:
  //     A |-> D |-> B -> C
  //       |-> B -> C
  //       |-> B -> C
  //       |-> E |-> B -> C
  //
  // the `proc_instance_stack` for each `C` would look like:
  //    [{A, 0}, {D, 0}, {B, 0}, {C, 0}]
  //    [{A, 0}, {B, 0}, {C, 0}]
  //    [{A, 0}, {B, 1}, {C, 0}]
  //    [{A, 0}, {E, 0}, {B, 0}, {C, 0}]
  std::vector<std::pair<Proc*, int>> proc_instance_stack;

  std::string ToString() const {
    if (proc_instance_stack.empty()) {
      return "";
    }
    // The first proc in a chain never needs an instance count. Leaving it out
    // specifically when the chain length is more than 1 gets us the historical
    // output in most cases (where there was only an instance count at the end
    // of the chain).
    CHECK_EQ(proc_instance_stack[0].second, 0);
    const bool omit_first_instance_count = proc_instance_stack.size() > 1;
    std::string part_with_instance_counts = absl::StrJoin(
        proc_instance_stack.begin() + (omit_first_instance_count ? 1 : 0),
        proc_instance_stack.end(), "->",
        [](std::string* out, const std::pair<Proc*, int> p) {
          absl::StrAppendFormat(out, "%s:%d", p.first->identifier(), p.second);
        });
    return omit_first_instance_count
               ? absl::StrCat(proc_instance_stack[0].first->identifier(), "->",
                              part_with_instance_counts)
               : part_with_instance_counts;
  }

  bool operator==(const ProcId& other) const {
    return proc_instance_stack == other.proc_instance_stack;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ProcId& pid) {
    return H::combine(std::move(h), pid.proc_instance_stack);
  }
};

// An object that deals out `ProcId` instances.
class ProcIdFactory {
 public:
  // Creates a `ProcId` representing the given `spawnee` spawned by the given
  // `parent` context. If `parent` is `nullopt` then the `spawnee` is the root
  // of the proc network. If `count_as_new_instance` is true, then subsequent
  // calls with the same `parent` and `spawnee` will get a new instance count
  // value. Otherwise, subsequent calls will get an equivalent `ProcId` to the
  // one returned by this call.
  ProcId CreateProcId(const std::optional<ProcId>& parent, Proc* spawnee,
                      bool count_as_new_instance = true);

  // Returns whether any `spawnee` ever passed to `CreateProcId` in this
  // factory has been passed more than once with the `count_as_new_instance`
  // flag.
  bool HasMultipleInstancesOfAnyProc() const {
    return has_multiple_instances_of_any_proc_;
  }

 private:
  // Maps each `parent` and `spawnee` identifier passed to `CreateProcId` to the
  // number of instances of that pairing, i.e., the number of times that
  // `parent` and `spawnee` have been passed in with `true` for
  // `count_as_new_instance`.
  absl::flat_hash_map<std::pair<ProcId, std::string>, int> instance_counts_;
  bool has_multiple_instances_of_any_proc_ = false;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_PROC_ID_H_
