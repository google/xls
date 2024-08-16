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
#ifndef XLS_DSLX_IR_CONVERT_CHANNEL_SCOPE_H_
#define XLS_DSLX_IR_CONVERT_CHANNEL_SCOPE_H_

#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/package.h"

namespace xls::dslx {

// TODO: https://github.com/google/xls/issues/704 - Add an array representation.
using ChannelOrArray = std::variant<Channel*>;

// An object that manages definition and access to channels used in a proc.
class ChannelScope {
 public:
  ChannelScope(PackageConversionData* conversion_info, TypeInfo* type_info,
               ImportData* import_data, const ParametricEnv& bindings);

  // Creates the channel object, or array of channel objects, indicated by the
  // given `decl`, and returns it.
  absl::StatusOr<ChannelOrArray> DefineChannelOrArray(const ChannelDecl* decl);

  // Associates `name_def` as an alias of the channel or array defined by the
  // given `decl` that was previously passed to `DefineChannelOrArray`. This
  // should be used, for example, when a channel is passed into `spawn` and the
  // receiving proc associates it with a local argument name.
  absl::StatusOr<ChannelOrArray> AssociateWithExistingChannelOrArray(
      const NameDef* name_def, const ChannelDecl* decl);

  // Variant of `AssociateWithExistingChannelOrArray`, to be used when the
  // caller has the `channel` returned by `DefineChannelOrArray` on hand, rather
  // than the `decl` it was made from.
  absl::Status AssociateWithExistingChannel(const NameDef* name_def,
                                            Channel* channel);

 private:
  PackageConversionData* const conversion_info_;
  TypeInfo* const type_info_;
  ImportData* const import_data_;
  NameUniquer channel_name_uniquer_;
  const ParametricEnv& bindings_;

  absl::flat_hash_map<const ChannelDecl*, Channel*> decl_to_channel_;
  absl::flat_hash_map<const NameDef*, Channel*> name_def_to_channel_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_CHANNEL_SCOPE_H_
