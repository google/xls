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

#ifndef XLS_DSLX_FRONTEND_FUZZ_DOMAIN_REWRITER_H_
#define XLS_DSLX_FRONTEND_FUZZ_DOMAIN_REWRITER_H_

#include "absl/status/status.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"

namespace xls::dslx {

// Walks the module and populates all empty fuzz domain structs with members
// derived from their original structs.
absl::Status RewriteDomainStructs(Module& module, ImportData& import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_FUZZ_DOMAIN_REWRITER_H_
