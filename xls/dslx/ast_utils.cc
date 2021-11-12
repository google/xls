// Copyright 2021 The XLS Authors
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
#include "xls/dslx/ast_utils.h"

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

absl::StatusOr<Proc*> ResolveProc(Expr* callee, TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetProcOrError(name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  absl::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  absl::optional<const ImportedInfo*> imported_info =
      type_info->GetImported(*import);
  return imported_info.value()->module->GetProcOrError(colon_ref->attr());
}

}  // namespace xls::dslx
