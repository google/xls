// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_TRAIT_DERIVER_DISPATCHER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_TRAIT_DERIVER_DISPATCHER_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/trait_deriver.h"

namespace xls::dslx {

// An object that facilitates building a single `TraitDeriver` out of
// separate per-target-function implementations.
class TraitDeriverDispatcher : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& actual_struct_def,
      const StructType& concrete_struct_type, const Function& function) final {
    const auto it = handlers_.find(
        std::make_pair(trait.identifier(), function.identifier()));
    if (it == handlers_.end()) {
      return absl::UnimplementedError(
          absl::Substitute("No handler set for function `$0` of trait `$1`.",
                           function.identifier(), trait.identifier()));
    }
    return it->second->DeriveFunctionBody(module, trait, actual_struct_def,
                                          concrete_struct_type, function);
  }

  void SetHandler(std::string_view trait_name, std::string_view function_name,
                  std::unique_ptr<TraitDeriver> handler) {
    handlers_[std::make_pair(std::string(trait_name),
                             std::string(function_name))] = std::move(handler);
  }

 private:
  absl::flat_hash_map<std::pair<std::string, std::string>,
                      std::unique_ptr<TraitDeriver>>
      handlers_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_TRAIT_DERIVER_DISPATCHER_H_
