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

#include <list>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

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
#include "xls/ir/type.h"

namespace xls::dslx {

// Represents an array of channels that has been flattened by a `ChannelScope`
// so that each element has a synthetic, flattened name. The data in a
// `ChannelArray` is opaque to all but the internals of the `ChannelScope`.
class ChannelArray {
  friend class ChannelScope;

 public:
  // For logging purposes.
  std::string ToString() const { return base_channel_name_; }

 private:
  explicit ChannelArray(std::string_view base_channel_name)
      : base_channel_name_(base_channel_name) {}

  std::string_view base_channel_name() const { return base_channel_name_; }

  void AddChannel(std::string_view flattened_name, Channel* channel) {
    flattened_name_to_channel_.emplace(flattened_name, channel);
  }

  std::optional<Channel*> FindChannel(std::string_view flattened_name) {
    const auto it = flattened_name_to_channel_.find(flattened_name);
    if (it == flattened_name_to_channel_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // The base channel name is the DSLX package name plus the channel name string
  // in the DSLX source code.
  const std::string base_channel_name_;

  // Each element in this map represents one element of the array. The flattened
  // channel name for an element is `base_channel_name_` plus a suffix produced
  // by `ChannelScope::CreateAllArrayElementSuffixes()`, with the base name and
  // suffix separated by a double underscore.
  absl::flat_hash_map<std::string, Channel*> flattened_name_to_channel_;
};

using ChannelOrArray = std::variant<Channel*, ChannelArray*>;

// An object that manages definition and access to channels used in a proc.
class ChannelScope {
 public:
  ChannelScope(PackageConversionData* conversion_info, ImportData* import_data,
               std::optional<FifoConfig> default_fifo_config = std::nullopt);

  // The owner (IR converter driving the overall procedure) should invoke this
  // with the `type_info` and `bindings` for the function, before converting
  // each function to IR. All other functions assume this has been done.
  void EnterFunctionContext(TypeInfo* type_info,
                            const ParametricEnv& bindings) {
    function_context_.emplace(type_info, bindings);
  }

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
  // caller has the channel or array returned by `DefineChannelOrArray` on hand,
  // rather than the `decl` it was made from.
  absl::Status AssociateWithExistingChannelOrArray(
      const NameDef* name_def, ChannelOrArray channel_or_array);

  // Retrieves the individual `Channel` that is referred to by the given `index`
  // operation. In order for this to succeed, `index` must meet the following
  // requirements:
  //   1. The array being indexed must have been previously defined via
  //      `DefineChannelOrArray`. Otherwise, a not-found error is returned.
  //   2. If the array being indexed is an alias, then that alias must have been
  //      associated with an existing array. Otherwise, a not-found error is
  //      returned.
  //   3. The expression(s) in `index` indicating the element in the array must
  //      be constexpr evaluatable.
  // A not-found error is the guaranteed result in cases where `index` is not
  // a channel array index operation at all.
  absl::StatusOr<Channel*> GetChannelForArrayIndex(const Index* index);

 private:
  absl::StatusOr<std::string_view> GetBaseNameForNameDef(
      const NameDef* name_def);

  std::string_view GetBaseNameForChannelOrArray(
      ChannelOrArray channel_or_array);

  absl::StatusOr<std::vector<std::string>> CreateAllArrayElementSuffixes(
      const std::vector<Expr*>& dims);

  absl::StatusOr<std::string> CreateBaseChannelName(const ChannelDecl* decl);

  absl::StatusOr<xls::Type*> GetChannelType(const ChannelDecl* decl) const;

  absl::StatusOr<std::optional<FifoConfig>> CreateFifoConfig(
      const ChannelDecl* decl) const;

  absl::StatusOr<Channel*> CreateChannel(std::string_view name, xls::Type* type,
                                         std::optional<FifoConfig> fifo_config);

  absl::StatusOr<Channel*> GetChannelArrayElement(
      const NameRef* name_ref, std::string_view flattened_name_suffix);

  PackageConversionData* const conversion_info_;
  ImportData* const import_data_;
  NameUniquer channel_name_uniquer_;

  // Set by the caller via `EnterContext()` before the conversion of each
  // function.
  struct FunctionContext {
    FunctionContext(TypeInfo* type_info_value,
                    const ParametricEnv& bindings_value)
        : type_info(type_info_value), bindings(bindings_value) {}

    TypeInfo* const type_info;
    const ParametricEnv& bindings;
  };
  std::optional<FunctionContext> function_context_;

  // Owns all arrays that are pointed to by ChannelOrArray objects dealt out by
  // this scope. A `list` is used for pointer stability.
  std::list<ChannelArray> arrays_;

  // If present, the default FIFO config to use for any FIFO that does not
  // specify a depth.
  std::optional<FifoConfig> default_fifo_config_;

  absl::flat_hash_map<const ChannelDecl*, ChannelOrArray>
      decl_to_channel_or_array_;
  absl::flat_hash_map<const NameDef*, ChannelOrArray>
      name_def_to_channel_or_array_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_CHANNEL_SCOPE_H_
