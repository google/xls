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
#include "xls/dslx/bytecode_cache.h"

#include "xls/dslx/bytecode_emitter.h"

namespace xls::dslx {

BytecodeCache::BytecodeCache(ImportData* import_data)
    : import_data_(import_data) {}

absl::StatusOr<BytecodeFunction*> BytecodeCache::GetOrCreateBytecodeFunction(
    const Function* f, const TypeInfo* type_info,
    const std::optional<SymbolicBindings>& caller_bindings) {
  Key key = std::make_tuple(f, type_info, caller_bindings);
  if (!cache_.contains(key)) {
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<BytecodeFunction> bf,
        BytecodeEmitter::Emit(import_data_, type_info, f, caller_bindings));
    cache_.emplace(key, std::move(bf));
  }

  return cache_.at(key).get();
}

}  // namespace xls::dslx
