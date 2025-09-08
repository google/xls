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

#include "xls/ir/clone_package.h"

#include <memory>
#include <optional>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/call_graph.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

absl::StatusOr<std::unique_ptr<Package>> ClonePackage(
    const Package* p, std::optional<std::string_view> name) {
  std::unique_ptr<Package> clone =
      std::make_unique<Package>(name.value_or(p->name()));
  absl::flat_hash_map<Channel*, Channel*> chan_map;
  absl::flat_hash_map<const Function*, Function*> func_map;
  absl::flat_hash_map<const Block*, Block*> block_map;
  absl::flat_hash_map<const FunctionBase*, FunctionBase*> fb_map;

  for (Channel* c : p->channels()) {
    XLS_ASSIGN_OR_RETURN(Channel * new_chan, clone->CloneChannel(c, c->name()));
    chan_map[c] = new_chan;
  }
  for (FunctionBase* fb : FunctionsInPostOrder(p)) {
    if (fb->IsFunction()) {
      Function* f = fb->AsFunctionOrDie();
      XLS_ASSIGN_OR_RETURN(Function * nf,
                           f->Clone(f->name(), clone.get(), func_map));
      func_map[f] = nf;
      fb_map[f] = nf;
    } else if (fb->IsProc()) {
      Proc* op = fb->AsProcOrDie();
      XLS_ASSIGN_OR_RETURN(
          Proc * np,
          op->Clone(op->name(), clone.get(), /*channel_remapping=*/{}, fb_map,
                    /*state_name_remapping=*/{}));
      fb_map[op] = np;
    } else {
      XLS_RET_CHECK(fb->IsBlock());
      Block* b = fb->AsBlockOrDie();
      XLS_ASSIGN_OR_RETURN(Block * nb,
                           b->Clone(b->name(), clone.get(), /*reg_name_map=*/{},
                                    /*block_instantiation_map=*/block_map));
      block_map[b] = nb;
      fb_map[b] = nb;
    }
  }
  return clone;
}

}  // namespace xls
