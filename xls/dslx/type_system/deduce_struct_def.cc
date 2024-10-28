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

#include "xls/dslx/type_system/deduce_struct_def.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/deduce_struct_def_base_utils.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<Type>> DeduceStructDef(const StructDef* node,
                                                      DeduceCtx* ctx) {
  XLS_RETURN_IF_ERROR(TypecheckStructDefBase(node, ctx));
  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Type>> members,
                       DeduceStructDefBaseMembers(node, ctx));
  auto wrapped = std::make_unique<StructType>(std::move(members), *node);
  auto result = std::make_unique<MetaType>(std::move(wrapped));
  ctx->type_info()->SetItem(node->name_def(), *result);
  VLOG(5) << absl::StreamFormat("Deduced type for struct %s => %s",
                                node->ToString(), result->ToString());
  return result;
}

}  // namespace xls::dslx
