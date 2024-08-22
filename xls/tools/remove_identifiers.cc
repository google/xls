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

#include "xls/tools/remove_identifiers.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/call_graph.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"

namespace xls {

absl::StatusOr<std::unique_ptr<Package>> StripPackage(
    Package* source, const StripOptions& options) {
  std::unique_ptr<Package> result =
      std::make_unique<Package>(options.new_package_name);
  int function_count = 0;
  auto next_name = [&](FunctionBase* fb) -> std::string {
    if (options.strip_function_names) {
      return absl::StrFormat("%s_%d",
                             fb->IsFunction() ? "function"
                             : fb->IsProc()   ? "proc"
                                              : "block",
                             function_count++);
    }
    return std::string(fb->name());
  };
  if (!options.strip_location_info) {
    for (const auto& [fileno, name] : source->fileno_to_name()) {
      result->SetFileno(fileno, name);
    }
  }

  absl::flat_hash_map<std::string, std::string> chan_map;
  int64_t chan_cnt = 0;
  for (Channel* c : source->channels()) {
    auto new_name = options.strip_chan_names
                        ? absl::StrFormat("chan_%d", chan_cnt++)
                        : std::string(c->name());
    chan_map[c->name()] = new_name;
    XLS_RETURN_IF_ERROR(result->CloneChannel(c, new_name).status());
  }
  absl::flat_hash_map<const Function*, Function*> func_map;
  absl::flat_hash_map<const FunctionBase*, FunctionBase*> func_base_map;
  for (FunctionBase* fb : FunctionsInPostOrder(source)) {
    auto new_name = next_name(fb);
    if (fb->IsFunction()) {
      XLS_ASSIGN_OR_RETURN(
          Function * new_func,
          fb->AsFunctionOrDie()->Clone(new_name, result.get(), func_map));
      func_map[fb->AsFunctionOrDie()] = new_func;
      func_base_map[fb] = new_func;
    } else if (fb->IsProc()) {
      if (fb->AsProcOrDie()->is_new_style_proc()) {
        for (auto c : fb->AsProcOrDie()->channels()) {
          // New style procs
          auto new_chan_name = options.strip_chan_names
                                   ? absl::StrFormat("chan_%d", chan_cnt++)
                                   : std::string(c->name());
          chan_map[c->name()] = new_chan_name;
        }
      }
      XLS_ASSIGN_OR_RETURN(Proc * new_proc,
                           fb->AsProcOrDie()->Clone(new_name, result.get(),
                                                    chan_map, func_base_map));
      func_base_map[fb] = new_proc;
    } else {
      absl::flat_hash_map<std::string, std::string> register_map;
      if (options.strip_reg_names) {
        int64_t reg_cnt = 0;
        for (auto* r : fb->AsBlockOrDie()->GetRegisters()) {
          register_map[r->name()] = absl::StrFormat("register_%d", reg_cnt++);
        }
        if (fb->AsBlockOrDie()->GetClockPort()) {
          register_map[fb->AsBlockOrDie()->GetClockPort()->name] = "clock";
        }
      }
      XLS_ASSIGN_OR_RETURN(
          Block * new_block,
          fb->AsBlockOrDie()->Clone(new_name, result.get(), register_map));
      func_base_map[fb] = new_block;
    }
  }
  for (FunctionBase* fb : result->GetFunctionBases()) {
    for (Node* n : fb->nodes()) {
      if (options.strip_node_names) {
        if (n->Is<Param>()) {
          n->SetName("param");
        } else {
          n->ClearName();
        }
      }
      if (options.strip_location_info) {
        n->SetLoc(SourceInfo());
      }
    }
  }
  if (source->GetTop()) {
    XLS_RETURN_IF_ERROR(result->SetTop(func_base_map[*source->GetTop()]));
  }
  return result;
}

}  // namespace xls
