// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_LSP_FIND_DEFINITION_H_
#define XLS_DSLX_LSP_FIND_DEFINITION_H_

#include <optional>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Looks up what (if any) reference is present in the module at position
// "selected", and, if there is one present, resolve it to a defining construct,
// and returns the span of that name definition.
//
// For example:
//
//    fn f() -> u32 { u32:42 }
//    ---^ found definition
//
//    fn main() -> u32 { f() }
//    -------------------^ selected pos
//
// Note that this currently only supports resolution in a single file, e.g. a
// colon-reference to a construct in another module will return nullopt.
std::optional<const NameDef*> FindDefinition(const Module& m, const Pos& selected,
                                   const TypeInfo& type_info,
                                   ImportData& import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_FIND_DEFINITION_H_
