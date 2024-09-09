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

#include "xls/dslx/type_system/unwrap_meta_type.h"

#include <memory>
#include <string_view>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

absl::StatusOr<std::unique_ptr<Type>> UnwrapMetaType(
    std::unique_ptr<Type> t, const Span& span, std::string_view context,
    const FileTable& file_table) {
  MetaType* metatype = dynamic_cast<MetaType*>(t.get());
  if (metatype == nullptr) {
    return TypeInferenceErrorStatus(
        span, t.get(), absl::StrCat("Expected a type in ", context),
        file_table);
  }
  return std::move(metatype->wrapped());
}

absl::StatusOr<const Type*> UnwrapMetaType(const Type& t) {
  const MetaType* metatype = dynamic_cast<const MetaType*>(&t);
  XLS_RET_CHECK(metatype != nullptr) << t << " was not a metatype.";
  return metatype->wrapped().get();
}

}  // namespace xls::dslx
