// Copyright 2022 The XLS Authors
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
#include "xls/dslx/bytecode/bytecode_cache.h"

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

absl::StatusOr<BytecodeFunction*> BytecodeCache::GetOrCreateBytecodeFunction(
    ImportData& import_data, const Function& f, const TypeInfo* type_info,
    const std::optional<ParametricEnv>& caller_bindings) {
  XLS_RET_CHECK(type_info != nullptr);
  Key key = std::make_tuple(&f, type_info, caller_bindings);
  if (!cache_.contains(key)) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<BytecodeFunction> bf,
        BytecodeEmitter::Emit(&import_data, type_info, f, caller_bindings));
    cache_.emplace(key, std::move(bf));
  }

  return cache_.at(key).get();
}

}  // namespace xls::dslx
