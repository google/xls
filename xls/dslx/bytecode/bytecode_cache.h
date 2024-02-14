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
#ifndef XLS_DSLX_BYTECODE_BYTECODE_CACHE_H_
#define XLS_DSLX_BYTECODE_BYTECODE_CACHE_H_

#include <memory>
#include <optional>
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_cache_interface.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

class BytecodeCache : public BytecodeCacheInterface {
 public:
  explicit BytecodeCache(ImportData* import_data);

  absl::StatusOr<BytecodeFunction*> GetOrCreateBytecodeFunction(
      const Function* f, const TypeInfo* type_info,
      const std::optional<ParametricEnv>& caller_bindings) override;

 private:
  using Key = std::tuple<const Function*, const TypeInfo*,
                         std::optional<ParametricEnv>>;

  ImportData* import_data_;
  absl::flat_hash_map<Key, std::unique_ptr<BytecodeFunction>> cache_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_BYTECODE_CACHE_H_
