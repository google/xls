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

#ifndef XLS_DSLX_TYPE_SYSTEM_V2_SUM_INSTANCE_CANONICALIZER_H_
#define XLS_DSLX_TYPE_SYSTEM_V2_SUM_INSTANCE_CANONICALIZER_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"

namespace xls::dslx {

// Clones `module` with resolved sum-constructor expressions rewritten to the
// dedicated `SumInstance` expression form, including nested payloads, in one
// clone traversal that preserves lexical definition identity. Returns `nullopt`
// when no constructor expressions need canonicalization. Imports must already
// be available in `import_data`. Normalize parsed modules before preparation or
// type inference publishes their nodes.
absl::StatusOr<std::optional<std::unique_ptr<Module>>> CanonicalizeSumInstances(
    const Module& module, const ImportData& import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_V2_SUM_INSTANCE_CANONICALIZER_H_
