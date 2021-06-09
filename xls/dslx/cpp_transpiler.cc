// Copyright 2020 The XLS Authors
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

#include "xls/dslx/cpp_transpiler.h"

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "xls/common/case_converters.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {
namespace {

// Just a helper to avoid long signatures.
struct TranspileData {
  Module* module;
  TypeInfo* type_info;
  ImportData* import_data;
  absl::Span<const std::filesystem::path> addl_search_paths;
};

absl::StatusOr<std::string> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition);

absl::StatusOr<std::string> TranspileColonRef(const TranspileData& xpile_data,
                                              const ColonRef* colon_ref) {
  return absl::UnimplementedError("TranspileColonRef not yet implemented.");
}

absl::StatusOr<std::string> TranspileEnumDef(const TranspileData& xpile_data,
                                             const EnumDef* enum_def) {
  constexpr absl::string_view kTemplate = "enum class %s {\n%s\n};";
  constexpr absl::string_view kMemberTemplate = "  %s = %s,";

  std::vector<std::string> members;
  for (const EnumMember& member : enum_def->values()) {
    auto typecheck_fn = [&xpile_data](Module* module) {
      return CheckModule(module, xpile_data.import_data,
                         xpile_data.addl_search_paths);
    };

    // TODO(rspringer): 2021-06-09 Support parametric values.
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        Interpreter::InterpretExpr(xpile_data.module, xpile_data.type_info,
                                   typecheck_fn, xpile_data.addl_search_paths,
                                   xpile_data.import_data, {}, member.value));
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
    if (bit_count > 64) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "While transpiling enum %s: "
          "Only values up to 64b in size are currently supported. "
          "Member: %s (%d)",
          enum_def->identifier(), member.name_def->identifier(), bit_count));
    }

    std::string identifier =
        std::string(1, 'k') + Camelize(member.name_def->identifier());

    std::string val_str;
    if (value.IsSigned()) {
      XLS_ASSIGN_OR_RETURN(int64_t int_val, value.GetBitValueInt64());
      val_str = absl::StrCat(int_val);
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t int_val, value.GetBitValueUint64());
      val_str = absl::StrCat(int_val);
    }
    members.push_back(absl::StrFormat(kMemberTemplate, identifier, val_str));
  }

  return absl::StrFormat(kTemplate, enum_def->identifier(),
                         absl::StrJoin(members, "\n"));
}

absl::StatusOr<std::string> TranspileTypeDef(const TypeDef* type_def) {
  return absl::UnimplementedError("TranspileTypeDef not yet implemented.");
}

absl::StatusOr<std::string> TranspileStructDef(const StructDef* struct_def) {
  return absl::UnimplementedError("TranspileStructDef not yet implemented.");
}

absl::StatusOr<std::string> TranspileSingleToCpp(
    const TranspileData& xpile_data, const TypeDefinition& type_definition) {
  return absl::visit(Visitor{[](const TypeDef* type_def) {
                               return TranspileTypeDef(type_def);
                             },
                             [](const StructDef* struct_def) {
                               return TranspileStructDef(struct_def);
                             },
                             [&](const EnumDef* enum_def) {
                               return TranspileEnumDef(xpile_data, enum_def);
                             },
                             [&](const ColonRef* colon_ref) {
                               return TranspileColonRef(xpile_data, colon_ref);
                             }},
                     type_definition);
}

}  // namespace

absl::StatusOr<std::string> TranspileToCpp(
    Module* module, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  std::vector<std::string> results;
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->GetRootTypeInfo(module));
  struct TranspileData xpile_data {
    module, type_info, import_data, additional_search_paths
  };
  for (const TypeDefinition& def : module->GetTypeDefinitions()) {
    XLS_ASSIGN_OR_RETURN(std::string result,
                         TranspileSingleToCpp(xpile_data, def));
    results.push_back(result);
  }

  return absl::StrJoin(results, "\n");
}

}  // namespace xls::dslx
