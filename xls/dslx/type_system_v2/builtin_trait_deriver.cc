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
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/trait_deriver_dispatcher.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

class ToBitsDeriver : public TraitDeriver {
 public:
  absl::StatusOr<StatementBlock*> DeriveFunctionBody(
      Module& module, const Trait& trait, const StructDef& struct_def,
      const StructType& concrete_struct_type, const Function& function) final {
    XLS_RET_CHECK(!function.params().empty());
    Param* self_param = function.params()[0];
    std::vector<StructMemberNode*> members = struct_def.members();

    Expr* result = nullptr;
    if (members.empty()) {
      result = module.Make<Number>(
          Span::None(), "0", NumberKind::kOther,
          CreateUnOrSnAnnotation(module, Span::None(), false,
                                 static_cast<int64_t>(0)));
    } else {
      std::vector<Expr*> member_exprs;
      member_exprs.reserve(members.size());
      for (StructMemberNode* member : members) {
        const std::string member_name = member->name();
        Expr* self = module.Make<NameRef>(
            Span::None(), self_param->identifier(), self_param->name_def());
        member_exprs.push_back(
            module.Make<Attr>(Span::None(), self, member_name));
      }
      XLS_ASSIGN_OR_RETURN(
          result,
          Concat(
              module, absl::MakeSpan(member_exprs),
              concrete_struct_type.members(),
              /*invalid_element_handler=*/
              [&](const Expr* expr, const Type& type) -> absl::Status {
                return TypeInferenceErrorStatus(
                    struct_def.span(), /*type=*/nullptr,
                    absl::Substitute(
                        "Derivation of `$0` for `$1` encountered element "
                        "that cannot be converted to bits: `$2` of type `$3`.",
                        trait.identifier(), struct_def.identifier(),
                        expr->ToString(), type.ToString()),
                    *module.file_table());
              }));
    }

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

}  // namespace

std::unique_ptr<TraitDeriver> CreateBuiltinTraitDeriver() {
  auto result = std::make_unique<TraitDeriverDispatcher>();
  result->SetHandler("ToBits", "to_bits", std::make_unique<ToBitsDeriver>());
  return result;
}

}  // namespace xls::dslx
