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

#include "xls/dslx/make_value_format_descriptor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {
namespace {

absl::StatusOr<ValueFormatDescriptor> MakeStructFormatDescriptor(
    const StructTypeBase& struct_type, FormatPreference field_preference) {
  std::vector<std::string> field_names;
  std::vector<ValueFormatDescriptor> field_formats;
  field_names.reserve(struct_type.size());
  field_formats.reserve(struct_type.size());
  for (size_t i = 0; i < struct_type.size(); ++i) {
    const Type& member_type = struct_type.GetMemberType(i);
    field_names.push_back(std::string{struct_type.GetMemberName(i)});
    XLS_ASSIGN_OR_RETURN(
        auto desc, MakeValueFormatDescriptor(member_type, field_preference));
    field_formats.push_back(std::move(desc));
  }
  return ValueFormatDescriptor::MakeStruct(
      struct_type.struct_def_base().identifier(), field_names, field_formats);
}

absl::StatusOr<ValueFormatDescriptor> MakeTupleFormatDescriptor(
    const TupleType& tuple_type, FormatPreference field_preference) {
  std::vector<ValueFormatDescriptor> elements;
  for (size_t i = 0; i < tuple_type.size(); ++i) {
    const Type& member_type = tuple_type.GetMemberType(i);
    XLS_ASSIGN_OR_RETURN(
        auto vfd, MakeValueFormatDescriptor(member_type, field_preference));
    elements.push_back(std::move(vfd));
  }
  return ValueFormatDescriptor::MakeTuple(elements);
}

absl::StatusOr<ValueFormatDescriptor> MakeArrayFormatDescriptor(
    const ArrayType& type, FormatPreference field_preference) {
  XLS_ASSIGN_OR_RETURN(int64_t size, type.size().GetAsInt64());
  XLS_ASSIGN_OR_RETURN(
      ValueFormatDescriptor element_type_descriptor,
      MakeValueFormatDescriptor(type.element_type(), field_preference));
  return ValueFormatDescriptor::MakeArray(element_type_descriptor, size);
}

absl::StatusOr<ValueFormatDescriptor> MakeEnumFormatDescriptor(
    const EnumType& type, FormatPreference field_preference) {
  absl::flat_hash_map<Bits, std::string> value_to_name;
  const EnumDef& enum_def = type.nominal_type();
  for (size_t i = 0; i < enum_def.values().size(); ++i) {
    const std::string& s = enum_def.GetMemberName(i);
    const InterpValue& v = type.members().at(i);
    XLS_RET_CHECK(v.IsEnum());
    value_to_name[v.GetBitsOrDie()] = s;
  }
  return ValueFormatDescriptor::MakeEnum(enum_def.identifier(),
                                         std::move(value_to_name));
}

}  // namespace

absl::StatusOr<ValueFormatDescriptor> MakeValueFormatDescriptor(
    const Type& type, FormatPreference field_preference) {
  class Visitor : public TypeVisitor {
   public:
    explicit Visitor(FormatPreference field_preference)
        : field_preference_(field_preference) {}

    absl::Status HandleArray(const ArrayType& t) final {
      if (IsBitsLike(t)) {
        result_ = ValueFormatDescriptor::MakeLeafValue(field_preference_);
        return absl::OkStatus();
      }
      XLS_ASSIGN_OR_RETURN(result_,
                           MakeArrayFormatDescriptor(t, field_preference_));
      return absl::OkStatus();
    }
    absl::Status HandleStruct(const StructType& t) final {
      XLS_ASSIGN_OR_RETURN(result_,
                           MakeStructFormatDescriptor(t, field_preference_));
      return absl::OkStatus();
    }
    absl::Status HandleProc(const ProcType& t) final {
      XLS_ASSIGN_OR_RETURN(result_,
                           MakeStructFormatDescriptor(t, field_preference_));
      return absl::OkStatus();
    }
    absl::Status HandleTuple(const TupleType& t) final {
      XLS_ASSIGN_OR_RETURN(result_,
                           MakeTupleFormatDescriptor(t, field_preference_));
      return absl::OkStatus();
    }
    absl::Status HandleEnum(const EnumType& t) final {
      XLS_ASSIGN_OR_RETURN(result_,
                           MakeEnumFormatDescriptor(t, field_preference_));
      return absl::OkStatus();
    }
    absl::Status HandleBits(const BitsType& t) final {
      result_ = ValueFormatDescriptor::MakeLeafValue(field_preference_);
      return absl::OkStatus();
    }
    absl::Status HandleFunction(const FunctionType& t) final {
      return absl::InvalidArgumentError("Cannot format a function type; got: " +
                                        t.ToString());
    }
    absl::Status HandleToken(const TokenType& t) final {
      return absl::InvalidArgumentError("Cannot format a token type; got: " +
                                        t.ToString());
    }
    absl::Status HandleChannel(const ChannelType& t) final {
      return absl::InvalidArgumentError("Cannot format a channel type; got: " +
                                        t.ToString());
    }
    absl::Status HandleMeta(const MetaType& t) final {
      return absl::InvalidArgumentError("Cannot format a metatype; got: " +
                                        t.ToString());
    }
    absl::Status HandleBitsConstructor(const BitsConstructorType& t) final {
      return absl::InvalidArgumentError(
          "Cannot format a bits constructor; got: " + t.ToString());
    }
    absl::Status HandleModule(const ModuleType& t) final {
      return absl::InvalidArgumentError("Cannot format a module type; got: " +
                                        t.ToString());
    }

    ValueFormatDescriptor& result() { return result_; }

   private:
    const FormatPreference field_preference_;
    ValueFormatDescriptor result_;
  };

  Visitor v(field_preference);
  XLS_RETURN_IF_ERROR(type.Accept(v));
  return std::move(v.result());
}

}  // namespace xls::dslx
