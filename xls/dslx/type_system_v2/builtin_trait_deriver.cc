// Copyright 2025 The XLS Authors
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

#include "xls/dslx/type_system_v2/builtin_trait_deriver.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/type_to_type_annotation.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/trait_deriver_dispatcher.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

class SpawnDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(Module& module,
                                                     const Trait& trait,
                                                     const StructDefBase& def,
                                                     const StructTypeBase&,
                                                     const Function&) {
    if (def.kind() != AstNodeKind::kProcDef) {
      return TypeInferenceErrorStatus(
          def.span(), /*type=*/nullptr,
          absl::Substitute("Trait `$0` is only supported on procs, but struct "
                           "`$1` attempted to derive it.",
                           trait.identifier(), def.identifier()),
          *module.file_table());
    }

    return module.Make<StatementBlock>(Span::None(), std::vector<Statement*>{},
                                       /*trailing_semi=*/true);
  }
};

class ToBitsDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDefBase& def,
      const StructTypeBase& concrete_type, const Function& function) final {
    if (def.kind() != AstNodeKind::kStructDef) {
      return TypeInferenceErrorStatus(
          def.span(), /*type=*/nullptr,
          absl::Substitute("Trait `$0` is only supported on structs, but proc "
                           "`$1` attempted to derive it.",
                           trait.identifier(), def.identifier()),
          *module.file_table());
    }

    XLS_RET_CHECK(!function.params().empty());
    Param* self_param = function.params()[0];
    std::vector<StructMemberNode*> members = def.members();

    std::vector<Expr*> member_exprs;
    member_exprs.reserve(members.size());
    for (StructMemberNode* member : members) {
      const std::string member_name = member->name();
      Expr* self = module.Make<NameRef>(Span::None(), self_param->identifier(),
                                        self_param->name_def());
      member_exprs.push_back(
          module.Make<Attr>(Span::None(), self, member_name));
    }
    XLS_ASSIGN_OR_RETURN(
        Expr * result,
        Concat(module, absl::MakeSpan(member_exprs), concrete_type.members(),
               /*invalid_element_handler=*/
               [&](const Expr* expr, const Type& type) -> absl::Status {
                 return TypeInferenceErrorStatus(
                     def.span(), /*type=*/nullptr,
                     absl::Substitute(
                         "Derivation of `$0` for `$1` encountered element "
                         "that cannot be converted to bits: `$2` of type `$3`.",
                         trait.identifier(), def.identifier(), expr->ToString(),
                         type.ToString()),
                     *module.file_table());
               }));

    Statement* statement = module.Make<Statement>(result);
    return module.Make<StatementBlock>(Span::None(),
                                       std::vector<Statement*>{statement},
                                       /*trailing_semi=*/false);
  }

 private:
  using InvalidElementHandler =
      absl::FunctionRef<absl::Status(const Expr*, const Type&)>;

  // Generates a concatenation of all the given exprs converted to bits. Note
  // that `types` may contain either 1 element or `exprs.size()` elements,
  // depending on whether the exprs are of heterogeneous types.
  absl::StatusOr<Expr*> Concat(Module& module, absl::Span<Expr*> exprs,
                               const std::vector<std::unique_ptr<Type>>& types,
                               InvalidElementHandler invalid_element_handler) {
    XLS_RET_CHECK(types.size() == 1 || types.size() == exprs.size());
    if (exprs.empty()) {
      return module.Make<Number>(
          Span::None(), "0", NumberKind::kOther,
          CreateUnOrSnAnnotation(module, Span::None(), /*is_signed=*/false,
                                 static_cast<int64_t>(0)));
    }
    std::optional<Expr*> result;
    for (int i = 0; i < exprs.size(); i++) {
      const std::unique_ptr<Type>& type =
          types.size() == 1 ? types[0] : types[i];
      Expr* next = exprs[i];
      Expr* next_as_bits;
      if (const auto* bits_type = dynamic_cast<const BitsType*>(type.get())) {
        if (bits_type->is_signed()) {
          XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                               bits_type->size().GetAsInt64());
          next_as_bits = module.Make<Cast>(
              Span::None(), next,
              CreateUnOrSnAnnotation(module, Span::None(), /*is_signed=*/false,
                                     bit_count));
        } else {
          next_as_bits = next;
        }
      } else if (type->IsStruct()) {
        next_as_bits = module.Make<Invocation>(
            Span::None(), module.Make<Attr>(Span::None(), next, "to_bits"),
            /*args=*/std::vector<Expr*>{});
      } else if (type->IsTuple()) {
        const std::vector<std::unique_ptr<Type>>& element_types =
            types[i]->AsTuple().members();
        XLS_ASSIGN_OR_RETURN(
            next_as_bits, ArrayOrTupleToBits<TupleIndex>(
                              module, next, element_types.size(), element_types,
                              invalid_element_handler));
      } else if (type->IsArray()) {
        std::vector<std::unique_ptr<Type>> element_types;
        XLS_ASSIGN_OR_RETURN(int64_t element_count,
                             type->AsArray().size().GetAsInt64());
        element_types.push_back(type->AsArray().element_type().CloneToUnique());
        XLS_ASSIGN_OR_RETURN(
            next_as_bits,
            ArrayOrTupleToBits<Index>(module, next, element_count,
                                      element_types, invalid_element_handler));
      } else if (type->IsEnum()) {
        XLS_ASSIGN_OR_RETURN(int bit_count,
                             type->AsEnum().GetTotalBitCount()->GetAsInt64());
        next_as_bits = module.Make<Cast>(
            Span::None(), next,
            CreateUnOrSnAnnotation(module, Span::None(), /*is_signed=*/false,
                                   bit_count));
      } else {
        return invalid_element_handler(next, *type);
      }

      if (result.has_value()) {
        result = module.Make<Binop>(Span::None(), BinopKind::kConcat, *result,
                                    next_as_bits, Span::None(), false);
      } else {
        result = next_as_bits;
      }
    }
    return *result;
  }

  // Generates an expression to convert an array or tuple to bits. Note that
  // `element_types` only contains one type for an array, but it contains
  // `element_count` types for a tuple.
  template <typename IndexNodeType>
  absl::StatusOr<Expr*> ArrayOrTupleToBits(
      Module& module, Expr* container, int element_count,
      const std::vector<std::unique_ptr<Type>>& element_types,
      InvalidElementHandler invalid_element_handler) {
    XLS_RET_CHECK(element_types.size() == 1 ||
                  element_types.size() == element_count);
    std::vector<Expr*> elements;
    elements.reserve(element_count);
    for (int i = 0; i < element_count; i++) {
      elements.push_back(module.Make<IndexNodeType>(
          Span::None(), container,
          module.Make<Number>(Span::None(), absl::StrCat(i), NumberKind::kOther,
                              /*type_annotation=*/nullptr)));
    }
    return Concat(module, absl::MakeSpan(elements), element_types,
                  invalid_element_handler);
  }
};

class DefaultDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDefBase& def,
      const StructTypeBase& concrete_type, const Function& function) final {
    if (def.kind() == AstNodeKind::kProcDef) {
      return TypeInferenceErrorStatus(
          def.span(), /*type=*/nullptr,
          absl::Substitute("Trait `$0` is only supported on structs, but proc "
                           "`$1` attempted to derive it.",
                           trait.identifier(), def.identifier()),
          *module.file_table());
    }

    std::vector<std::pair<std::string, Expr*>> members;
    for (int64_t i = 0; i < concrete_type.size(); i++) {
      StructMemberNode* member = def.members()[i];
      XLS_ASSIGN_OR_RETURN(
          Expr * value,
          MakeDefault(module, def, *member, concrete_type.GetMemberType(i)));
      members.push_back({member->name(), value});
    }
    // Infer parametrics from `Self`.
    TypeAnnotation* struct_ref = module.Make<TypeRefTypeAnnotation>(
        Span::None(),
        module.Make<TypeRef>(Span::None(), const_cast<StructDefBase*>(&def)),
        std::vector<ExprOrType>(), std::nullopt);
    Statement* statement = module.Make<Statement>(
        module.Make<StructInstance>(Span::None(), struct_ref, members));
    return module.Make<StatementBlock>(Span::None(),
                                       std::vector<Statement*>{statement},
                                       /*trailing_semi=*/false);
  }

 private:
  absl::StatusOr<Expr*> MakeDefault(Module& module, const StructDefBase& def,
                                    const StructMemberNode& member,
                                    const Type& type) {
    if (!ContainsStruct(type)) {
      if (type.HasEnum()) {
        return MissingDefaultError(module, def, member, type);
      }
      XLS_ASSIGN_OR_RETURN(TypeAnnotation * zero_type,
                           CreateTypeAnnotation(module, type, Span::None()));
      return module.Make<ZeroMacro>(Span::None(), zero_type);
    }
    if (type.IsArray()) {
      XLS_ASSIGN_OR_RETURN(int64_t size, type.AsArray().size().GetAsInt64());
      if (size == 0) {
        return module.Make<Array>(Span::None(), std::vector<Expr*>{},
                                  /*has_ellipsis=*/false);
      }
      XLS_ASSIGN_OR_RETURN(
          Expr * element,
          MakeDefault(module, def, member, type.AsArray().element_type()));
      return module.Make<Array>(Span::None(), std::vector<Expr*>{element},
                                /*has_ellipsis=*/size > 1);
    }
    if (type.IsTuple()) {
      std::vector<Expr*> elements;
      for (int64_t i = 0; i < type.AsTuple().size(); i++) {
        XLS_ASSIGN_OR_RETURN(
            Expr * element,
            MakeDefault(module, def, member, type.AsTuple().GetMemberType(i)));
        elements.push_back(element);
      }
      return module.Make<XlsTuple>(Span::None(), elements,
                                   /*has_trailing_comma=*/false);
    }

    const StructType& struct_type = type.AsStruct();
    const StructDef& nested = struct_type.nominal_type();
    if (!HasDefault(nested)) {
      return MissingDefaultError(module, def, member, type);
    }
    std::vector<ExprOrType> parametrics;
    for (const ParametricBinding* binding : nested.parametric_bindings()) {
      const auto it = struct_type.nominal_type_dims_by_identifier().find(
          binding->identifier());
      if (it == struct_type.nominal_type_dims_by_identifier().end() ||
          !it->second.value().IsBits()) {
        break;
      }
      const InterpValue& value = it->second.value();
      XLS_ASSIGN_OR_RETURN(int64_t bit_count, value.GetBitCount());
      parametrics.push_back(module.Make<Number>(
          Span::None(), value.ToString(/*humanize=*/true), NumberKind::kOther,
          CreateUnOrSnAnnotation(module, Span::None(), value.IsSigned(),
                                 bit_count)));
    }
    TypeRefTypeAnnotation* subject = module.Make<TypeRefTypeAnnotation>(
        Span::None(),
        module.Make<TypeRef>(Span::None(), const_cast<StructDef*>(&nested)),
        std::move(parametrics), std::nullopt);
    return module.Make<Invocation>(
        Span::None(), module.Make<ColonRef>(Span::None(), subject, "default"),
        std::vector<Expr*>());
  }

  static bool ContainsStruct(const Type& type) {
    if (type.IsArray()) {
      return ContainsStruct(type.AsArray().element_type());
    }
    if (type.IsTuple()) {
      return absl::c_any_of(type.AsTuple().members(),
                            [](const auto& t) { return ContainsStruct(*t); });
    }
    return type.IsStruct();
  }

  static bool HasDefault(const StructDef& def) {
    if (def.impl().has_value() &&
        (*def.impl())->GetFunction("default").has_value()) {
      return true;
    }
    std::optional<const Attribute*> derive =
        GetAttribute(&def, AttributeKind::kDerive);
    return derive.has_value() &&
           absl::c_any_of((*derive)->args(), [](const auto& arg) {
             const auto* name = std::get_if<std::string>(&arg);
             return name != nullptr && *name == "Default";
           });
  }

  static absl::Status MissingDefaultError(Module& module,
                                          const StructDefBase& def,
                                          const StructMemberNode& member,
                                          const Type& type) {
    return TypeInferenceErrorStatus(
        def.span(), /*type=*/nullptr,
        absl::Substitute("Cannot derive `Default` for `$0`: field `$1` has "
                         "type `$2`, which does not implement `Default`.",
                         def.identifier(), member.name(),
                         type.ToInlayHintString()),
        *module.file_table());
  }
};

}  // namespace

std::unique_ptr<TraitDeriver> CreateBuiltinTraitDeriver() {
  auto result = std::make_unique<TraitDeriverDispatcher>();
  result->SetHandler("Spawn", "spawn", std::make_unique<SpawnDeriver>());
  result->SetHandler("ToBits", "to_bits", std::make_unique<ToBitsDeriver>());
  result->SetHandler("Default", "default", std::make_unique<DefaultDeriver>());
  return result;
}

}  // namespace xls::dslx
