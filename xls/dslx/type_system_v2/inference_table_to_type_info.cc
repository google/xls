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

#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"

#include <memory>
#include <optional>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {
namespace {

absl::StatusOr<std::unique_ptr<Type>> Concretize(
    const TypeAnnotation* annotation, const InferenceTable& table,
    const FileTable& file_table) {
  if (const auto* builtin_annotation =
          dynamic_cast<const BuiltinTypeAnnotation*>(annotation);
      builtin_annotation != nullptr) {
    return ConcretizeBuiltinTypeAnnotation(*builtin_annotation, file_table);
  }
  return absl::UnimplementedError(absl::Substitute(
      "Type inference version 2 is a work in progress and cannot yet handle "
      "type annotation `$0`.",
      annotation->ToString()));
}

}  // namespace

absl::StatusOr<TypeInfo*> InferenceTableToTypeInfo(
    const InferenceTable& table, Module& module, TypeInfoOwner& type_info_owner,
    const FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * ti, type_info_owner.New(&module));
  for (const AstNode* node : table.GetNodes()) {
    std::optional<const TypeAnnotation*> annotation =
        table.GetTypeAnnotation(node);
    if (!annotation.has_value()) {
      return absl::UnimplementedError(absl::Substitute(
          "Type inference version 2 is a work in progress and cannot yet "
          "handle `$0` because it has no type annotation.",
          node->ToString()));
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type,
                         Concretize(*annotation, table, file_table));
    ti->SetItem(node, *type);
  }
  return ti;
}

}  // namespace xls::dslx
