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

#include "xls/dslx/type_system/deduce_utils.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token_utils.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_with_type.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"

namespace xls::dslx {
namespace {

// A validator for the type of an argument being formatted by a macro like
// `trace_fmt!` or `vtrace_fmt!`.
class FormatMacroArgumentValidator : public TypeVisitor {
 public:
  FormatMacroArgumentValidator(const FileTable& file_table, const Span& span)
      : file_table_(file_table), span_(span) {}

  absl::Status HandleArray(const ArrayType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleBits(const BitsType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleEnum(const EnumType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleToken(const TokenType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleStruct(const StructType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleProc(const ProcType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleTuple(const TupleType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleBitsConstructor(const BitsConstructorType& t) override {
    return absl::OkStatus();
  }
  absl::Status HandleFunction(const FunctionType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with function type",
        file_table_);
  }
  absl::Status HandleChannel(const ChannelType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with channel type",
        file_table_);
  }
  absl::Status HandleMeta(const MetaType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with meta type", file_table_);
  }
  absl::Status HandleModule(const ModuleType& t) override {
    return TypeInferenceErrorStatus(
        span_, &t, ": Cannot format an expression with module type",
        file_table_);
  }

 private:
  const FileTable& file_table_;
  const Span& span_;
};

}  // namespace

// Resolves a `TypeAlias` AST node to a `ColonRef` subject -- this requires us
// to traverse through aliases transitively to find a subject.
//
// Has to be an enum or builtin-type name, given the context we're in: looking
// for _values_ hanging off, e.g. in service of a `::` ref.
//
// Note: the returned AST node may not be from the same module that the
// original `type_alias` was from.
static absl::StatusOr<std::variant<
    EnumDef*, BuiltinNameDef*, ArrayTypeAnnotation*, TypeRefTypeAnnotation*>>
ResolveTypeAliasToDirectColonRefSubject(ImportData* import_data,
                                        const TypeInfo* type_info,
                                        TypeAlias* type_alias) {
  VLOG(5) << "ResolveTypeDefToDirectColonRefSubject; type_alias: `"
          << type_alias->ToString() << "`";

  // Resolve through all the transitive aliases.
  TypeDefinition current_type_definition = type_alias;
  TypeRefTypeAnnotation* current_type_ref;
  while (std::holds_alternative<TypeAlias*>(current_type_definition)) {
    TypeAlias* next_type_alias = std::get<TypeAlias*>(current_type_definition);
    VLOG(5) << " TypeAlias: `" << next_type_alias->ToString() << "`";
    TypeAnnotation& type = next_type_alias->type_annotation();
    VLOG(5) << " TypeAnnotation: `" << type.ToString() << "`";

    if (auto* bti = dynamic_cast<BuiltinTypeAnnotation*>(&type);
        bti != nullptr) {
      return bti->builtin_name_def();
    }
    if (auto* ata = dynamic_cast<ArrayTypeAnnotation*>(&type); ata != nullptr) {
      return ata;
    }

    TypeRefTypeAnnotation* type_ref_type =
        dynamic_cast<TypeRefTypeAnnotation*>(&type);
    // TODO(rspringer): We'll need to collect parametrics from type_ref_type to
    // support parametric TypeDefs.
    XLS_RET_CHECK(type_ref_type != nullptr)
        << type.ToString() << " :: " << type.GetNodeTypeName();
    VLOG(5) << " TypeRefTypeAnnotation: `" << type_ref_type->ToString() << "`";

    current_type_definition = type_ref_type->type_ref()->type_definition();
    current_type_ref = type_ref_type;
  }

  VLOG(5) << absl::StreamFormat(
      "ResolveTypeDefToDirectColonRefSubject; arrived at type definition: `%s`",
      ToAstNode(current_type_definition)->ToString());

  if (std::holds_alternative<ColonRef*>(current_type_definition)) {
    ColonRef* colon_ref = std::get<ColonRef*>(current_type_definition);
    type_info = import_data->GetRootTypeInfo(colon_ref->owner()).value();
    XLS_ASSIGN_OR_RETURN(ColonRefSubjectT subject,
                         ResolveColonRefSubjectForTypeChecking(
                             import_data, type_info, colon_ref));
    XLS_RET_CHECK(std::holds_alternative<Module*>(subject));
    Module* module = std::get<Module*>(subject);

    // Grab the type definition being referred to by the `ColonRef` -- this is
    // what we now have to traverse to (or we may have arrived).
    XLS_ASSIGN_OR_RETURN(current_type_definition,
                         module->GetTypeDefinition(colon_ref->attr()));

    if (std::holds_alternative<TypeAlias*>(current_type_definition)) {
      TypeAlias* new_alias = std::get<TypeAlias*>(current_type_definition);
      XLS_RET_CHECK_EQ(new_alias->owner(), module);
      // We need to get the right type info for the enum's containing module. We
      // can get the top-level module since [currently?] enums can't be
      // parameterized.
      type_info = import_data->GetRootTypeInfo(module).value();
      return ResolveTypeAliasToDirectColonRefSubject(import_data, type_info,
                                                     new_alias);
    }
  }

  if (std::holds_alternative<UseTreeEntry*>(current_type_definition)) {
    UseTreeEntry* use_tree_entry =
        std::get<UseTreeEntry*>(current_type_definition);
    std::string_view identifier =
        use_tree_entry->GetLeafNameDef().value()->identifier();
    // Get the imported module.
    XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported,
                         type_info->GetImportedOrError(use_tree_entry));
    const Module& imported_module = *imported->module;
    XLS_ASSIGN_OR_RETURN(current_type_definition,
                         imported_module.GetTypeDefinition(identifier));
  }

  // If struct type, return the `TypeRefTypeAnnotation` to preserve parametrics.
  if (std::holds_alternative<StructDef*>(current_type_definition)) {
    return current_type_ref;
  }

  if (!std::holds_alternative<EnumDef*>(current_type_definition)) {
    return absl::InternalError(absl::StrFormat(
        "ResolveTypeDefToDirectColonRefSubject() can only be called when the "
        "TypeAlias directly or indirectly refers to an EnumDef or StructDef; "
        "got: `%s` (kind: %s)",
        ToAstNode(current_type_definition)->ToString(),
        ToAstNode(current_type_definition)->GetNodeTypeName()));
  }

  return std::get<EnumDef*>(current_type_definition);
}

absl::Status TryEnsureFitsInType(const Number& number,
                                 const BitsLikeProperties& bits_like,
                                 const Type& type) {
  const FileTable& file_table = *number.owner()->file_table();
  VLOG(5) << "TryEnsureFitsInType; number: " << number.ToString() << " @ "
          << number.span().ToString(file_table);

  std::optional<bool> maybe_signed;
  if (!bits_like.is_signed.IsParametric()) {
    maybe_signed = bits_like.is_signed.GetAsBool().value();
  }

  // Characters have a `u8` type. They can support the dash (negation symbol).
  if (number.number_kind() != NumberKind::kCharacter &&
      number.text()[0] == '-' && maybe_signed.has_value() &&
      !maybe_signed.value()) {
    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Number %s invalid: "
                        "can't assign a negative value to an unsigned type.",
                        number.ToString()),
        file_table);
  }

  if (bits_like.size.IsParametric()) {
    // We have to wait for the dimension to be fully resolved before we can
    // check that the number is compliant.
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(const int64_t bit_count, bits_like.size.GetAsInt64());

  // Helper to give an informative error on the appropriate range when we
  // determine the numerical value given doesn't fit into the type.
  auto does_not_fit = [&]() -> absl::Status {
    std::string low;
    std::string high;
    XLS_RET_CHECK(maybe_signed.has_value());
    if (maybe_signed.value()) {
      low = BitsToString(Bits::MinSigned(bit_count),
                         FormatPreference::kSignedDecimal);
      high = BitsToString(Bits::MaxSigned(bit_count),
                          FormatPreference::kSignedDecimal);
    } else {
      low = BitsToString(Bits(bit_count), FormatPreference::kUnsignedDecimal);
      high = BitsToString(Bits::AllOnes(bit_count),
                          FormatPreference::kUnsignedDecimal);
    }

    return TypeInferenceErrorStatus(
        number.span(), &type,
        absl::StrFormat("Value '%s' does not fit in "
                        "the bitwidth of a %s (%d). "
                        "Valid values are [%s, %s].",
                        number.text(), type.ToString(), bit_count, low, high),
        file_table);
  };

  XLS_ASSIGN_OR_RETURN(
      bool fits_in_type,
      number.FitsInType(bit_count, maybe_signed.value_or(false)));
  if (!fits_in_type) {
    return does_not_fit();
  }
  return absl::OkStatus();
}

absl::Status TryEnsureFitsInType(const Number& number, const Type& type) {
  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      bits_like.has_value()) {
    return TryEnsureFitsInType(number, bits_like.value(), type);
  }
  return TypeInferenceErrorStatus(
      number.span(), &type,
      absl::StrFormat("Non-bits type (%s) used to define a numeric literal.",
                      type.GetDebugTypeName()),
      *number.owner()->file_table());
}

absl::Status TryEnsureFitsInBitsType(const Number& number,
                                     const BitsType& type) {
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  XLS_RET_CHECK(bits_like.has_value())
      << "bits type can always give bits-like properties";
  return TryEnsureFitsInType(number, bits_like.value(), type);
}

absl::Status ValidateArrayTypeForIndex(const Index& node, const Type& type,
                                       const FileTable& file_table) {
  if (const auto* tuple_type = dynamic_cast<const TupleType*>(&type)) {
    return TypeInferenceErrorStatus(
        node.span(), tuple_type,
        "Tuples should not be indexed with array-style syntax. "
        "Use `tuple.<number>` syntax instead.",
        file_table);
  }

  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  if (bits_like.has_value()) {
    return TypeInferenceErrorStatus(
        node.span(), &type,
        "Bits-like value cannot be indexed, value to index is not an array.",
        file_table);
  }

  const auto* array_type = dynamic_cast<const ArrayType*>(&type);
  if (array_type == nullptr) {
    return TypeInferenceErrorStatus(
        node.span(), &type, "Value to index is not an array.", file_table);
  }

  return absl::OkStatus();
}

absl::Status ValidateTupleTypeForIndex(const TupleIndex& node, const Type& type,
                                       const FileTable& file_table) {
  if (dynamic_cast<const TupleType*>(&type) != nullptr) {
    return absl::OkStatus();
  }
  return TypeInferenceErrorStatus(
      node.span(), &type,
      absl::StrCat("Attempted to use tuple indexing on a non-tuple: ",
                   node.ToString()),
      file_table);
}

absl::Status ValidateArrayIndex(const Index& node, const Type& array_type,
                                const Type& index_type, const TypeInfo& ti,
                                const FileTable& file_table) {
  std::optional<BitsLikeProperties> index_bits_like = GetBitsLike(index_type);
  if (!index_bits_like.has_value()) {
    return TypeInferenceErrorStatus(node.span(), &index_type,
                                    "Index is not bits typed.", file_table);
  }
  XLS_ASSIGN_OR_RETURN(bool is_signed, index_bits_like->is_signed.GetAsBool());
  if (is_signed) {
    return TypeInferenceErrorStatus(node.span(), &index_type,
                                    "Index is not unsigned-bits typed.",
                                    file_table);
  }

  const Expr* rhs = std::get<Expr*>(node.rhs());
  VLOG(10) << absl::StreamFormat("Index RHS: `%s` constexpr? %d",
                                 rhs->ToString(), ti.IsKnownConstExpr(rhs));

  // If we know the array size concretely and the index is a constexpr
  // expression, we can check it is in bounds. Note that in v2, the size will
  // never be parametric here, because v2 always resolves parametrics before
  // producing a `Type`.
  const auto& casted_array_type = dynamic_cast<const ArrayType&>(array_type);

  // Reject indexing into zero-sized arrays regardless of whether the index is
  // constexpr, since out-of-bounds semantics (return last element) are
  // undefined for empty arrays.
  if (!casted_array_type.size().IsParametric()) {
    XLS_ASSIGN_OR_RETURN(int64_t concrete_size,
                         casted_array_type.size().GetAsInt64());
    if (concrete_size == 0) {
      return TypeInferenceErrorStatus(node.span(), &array_type,
                                      "Zero-sized arrays cannot be indexed",
                                      file_table);
    }
  }
  if (casted_array_type.size().IsParametric() || !ti.IsKnownConstExpr(rhs)) {
    return absl::OkStatus();
  }

  VLOG(10) << "Array type: " << casted_array_type.ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue constexpr_value, ti.GetConstExpr(rhs));
  VLOG(10) << "Index RHS is known constexpr value: " << constexpr_value;
  XLS_ASSIGN_OR_RETURN(uint64_t constexpr_index,
                       constexpr_value.GetBitValueUnsigned());
  XLS_ASSIGN_OR_RETURN(int64_t array_size,
                       casted_array_type.size().GetAsInt64());
  if (constexpr_index >= array_size) {
    return TypeInferenceErrorStatus(
        node.span(), &array_type,
        absl::StrFormat("Index has a compile-time constant value %d that is "
                        "out of bounds of the array type.",
                        constexpr_index),
        file_table);
  }
  return absl::OkStatus();
}

absl::Status ValidateTupleIndex(const TupleIndex& node, const Type& tuple_type,
                                const Type& index_type, const TypeInfo& ti,
                                const FileTable& file_table) {
  // TupleIndex RHSs are always constexpr numbers.
  const auto& casted_tuple_type = dynamic_cast<const TupleType&>(tuple_type);
  XLS_ASSIGN_OR_RETURN(InterpValue index_value, ti.GetConstExpr(node.index()));
  XLS_ASSIGN_OR_RETURN(int64_t index, index_value.GetBitValueViaSign());
  if (index >= casted_tuple_type.size()) {
    return TypeInferenceErrorStatus(
        node.span(), &tuple_type,
        absl::StrCat("Out-of-bounds tuple index specified: ",
                     node.index()->ToString()),
        file_table);
  }
  return absl::OkStatus();
}

void UseImplicitToken(DeduceCtx* ctx) {
  CHECK(!ctx->fn_stack().empty());
  Function* caller = ctx->fn_stack().back().f();
  // Note: caller could be nullptr; e.g. when we're calling a function that
  // can fail!() from the top level of a module; e.g. in a module-level const
  // expression.
  if (caller != nullptr) {
    ctx->type_info()->NoteRequiresImplicitToken(*caller, true);
  }

  // TODO(rspringer): 2021-09-01: How to fail! from inside a proc?
}

bool IsNameRefTo(const Expr* e, const NameDef* name_def) {
  if (auto* name_ref = dynamic_cast<const NameRef*>(e)) {
    const AnyNameDef any_name_def = name_ref->name_def();
    return std::holds_alternative<const NameDef*>(any_name_def) &&
           std::get<const NameDef*>(any_name_def) == name_def;
  }
  return false;
}

absl::Status ValidateNumber(const Number& number, const Type& type) {
  VLOG(5) << "Validating " << number.ToString() << " vs " << type;

  if (std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
      bits_like.has_value()) {
    return TryEnsureFitsInType(number, bits_like.value(), type);
  }

  const FileTable& file_table = *number.owner()->file_table();
  return TypeInferenceErrorStatus(
      number.span(), &type,
      absl::StrFormat("Non-bits type (%s) used to define a numeric literal.",
                      type.GetDebugTypeName()),
      file_table);
}

absl::Status ValidateFormatMacroArgument(const Type& type, const Span& span,
                                         const FileTable& file_table) {
  FormatMacroArgumentValidator validator(file_table, span);
  return type.Accept(validator);
}

// When a ColonRef's subject is a NameRef, this resolves the entity referred to
// by that ColonRef. In a valid program that can only be a limited set of
// things, which is reflected in the return type provided.
//
// e.g.
//
//    A::B
//    ^
//    \- subject name_ref
//
// Args:
//  name_ref: The subject in the colon ref.
//
// Returns the entity the subject name_ref is referring to.
static absl::StatusOr<ColonRefSubjectT> ResolveColonRefNameRefSubject(
    NameRef* name_ref, ImportData* import_data, const TypeInfo* type_info) {
  VLOG(5) << "ResolveColonRefNameRefSubject for `" << name_ref->ToString()
          << "`";

  std::variant<const NameDef*, BuiltinNameDef*> any_name_def =
      name_ref->name_def();
  if (std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
    return std::get<BuiltinNameDef*>(any_name_def);
  }

  const NameDef* name_def = std::get<const NameDef*>(any_name_def);
  AstNode* definer = name_def->definer();

  auto make_subject_error = [&] {
    // We don't know how to colon-reference into this subject, so we return an
    // error.
    std::string type_str;
    if (definer != nullptr) {
      type_str =
          absl::StrFormat("; subject is a %s",
                          absl::AsciiStrToLower(definer->GetNodeTypeName()));
    }
    return TypeInferenceErrorStatus(
        name_ref->span(), nullptr,
        absl::StrFormat("Cannot resolve `::` subject `%s` -- subject must be a "
                        "module or enum definition%s",
                        name_ref->ToString(), type_str),
        import_data->file_table());
  };

  if (definer == nullptr) {
    return make_subject_error();
  }

  VLOG(5) << " ResolveColonRefNameRefSubject definer: `" << definer->ToString()
          << "` type: " << definer->GetNodeTypeName();

  // Now we have the AST node that defines the colon-ref subject -- we have to
  // turn that appropriately into the `ColonRefSubjectT`.

  // If the name is defined by an import statement we return the module that it
  // imports as the subject.
  if (Import* import = dynamic_cast<Import*>(definer); import != nullptr) {
    std::optional<const ImportedInfo*> imported =
        type_info->GetImported(import);
    if (!imported.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Could not find Module for Import: ", import->ToString()));
    }
    return imported.value()->module;
  }

  // If the LHS isn't an Import, then it should be an EnumDef (possibly via a
  // TypeAlias).
  if (EnumDef* enum_def = dynamic_cast<EnumDef*>(definer);
      enum_def != nullptr) {
    return enum_def;
  }

  if (StructDef* struct_def = dynamic_cast<StructDef*>(definer);
      struct_def != nullptr) {
    return struct_def;
  }

  if (UseTreeEntry* use_tree_entry = dynamic_cast<UseTreeEntry*>(definer);
      use_tree_entry != nullptr) {
    XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported,
                         type_info->GetImportedOrError(use_tree_entry));
    return imported->module;
  }

  TypeAlias* type_alias = dynamic_cast<TypeAlias*>(definer);
  if (type_alias == nullptr) {
    return make_subject_error();
  }

  if (type_alias->owner() != type_info->module()) {
    // We need to get the right type info for the enum's containing module. We
    // can get the top-level module since [currently?] enums can't be
    // parameterized (and we know this must be an enum, per the above).
    type_info = import_data->GetRootTypeInfo(type_alias->owner()).value();
  }
  XLS_ASSIGN_OR_RETURN(auto resolved, ResolveTypeAliasToDirectColonRefSubject(
                                          import_data, type_info, type_alias));
  return WidenVariantTo<ColonRefSubjectT>(resolved);
}

absl::StatusOr<ColonRefSubjectT> ResolveColonRefSubjectForTypeChecking(
    ImportData* import_data, const TypeInfo* type_info,
    const ColonRef* colon_ref) {
  XLS_RET_CHECK_EQ(colon_ref->owner(), type_info->module());

  VLOG(5) << absl::StreamFormat("ResolveColonRefSubject for `%s`",
                                colon_ref->ToString());

  if (std::holds_alternative<TypeRefTypeAnnotation*>(colon_ref->subject())) {
    return std::get<TypeRefTypeAnnotation*>(colon_ref->subject());
  }

  // If the subject is a name reference we use a helper routine.
  if (std::holds_alternative<NameRef*>(colon_ref->subject())) {
    NameRef* name_ref = std::get<NameRef*>(colon_ref->subject());
    return ResolveColonRefNameRefSubject(name_ref, import_data, type_info);
  }
  XLS_RET_CHECK(std::holds_alternative<ColonRef*>(colon_ref->subject()));
  ColonRef* subject = std::get<ColonRef*>(colon_ref->subject());
  XLS_ASSIGN_OR_RETURN(
      ColonRefSubjectT resolved_subject,
      ResolveColonRefSubjectForTypeChecking(import_data, type_info, subject));

  // Has to be a module, since it's a ColonRef inside a ColonRef.
  if (!std::holds_alternative<Module*>(resolved_subject)) {
    return TypeInferenceErrorStatus(
        subject->span(), nullptr,
        absl::StrFormat("Cannot resolve `::` -- subject is %s",
                        ToAstNode(resolved_subject)->GetNodeTypeName()),
        import_data->file_table());
  }

  Module* module = std::get<Module*>(resolved_subject);

  // And the subject has to be a type, namely an enum, since the ColonRef must
  // be of the form: <MODULE>::SOMETHING::SOMETHING_ELSE. Keep in mind, though,
  // that we might have to traverse an EnumDef.
  absl::StatusOr<TypeDefinition> td =
      module->GetTypeDefinition(subject->attr());
  if (!td.status().ok() && absl::IsNotFound(td.status())) {
    return TypeInferenceErrorStatus(
        colon_ref->span(), nullptr,
        absl::StrFormat(
            "Cannot resolve `::` to type definition -- module: `%s` attr: `%s`",
            module->name(), subject->attr()),
        import_data->file_table());
  }
  CHECK_OK(td.status())
      << "Only not-found error expected in retrieving type definition.";

  using ReturnT = absl::StatusOr<ColonRefSubjectT>;

  return absl::visit(
      Visitor{
          [&](TypeAlias* type_alias) -> ReturnT {
            XLS_ASSIGN_OR_RETURN(auto resolved,
                                 ResolveTypeAliasToDirectColonRefSubject(
                                     import_data, type_info, type_alias));
            return WidenVariantTo<ColonRefSubjectT>(resolved);
          },
          [](StructDef* struct_def) -> ReturnT { return struct_def; },
          [](ProcDef* proc_def) -> ReturnT {
            // TODO: https://github.com/google/xls/issues/836 - Support this.
            LOG(FATAL)
                << "Type deduction for impl-style procs is not yet supported.";
          },
          [](EnumDef* enum_def) -> ReturnT { return enum_def; },
          [](ColonRef* colon_ref) -> ReturnT { return colon_ref; },
          [](UseTreeEntry* use_tree_entry) -> ReturnT {
            LOG(FATAL) << "Extern UseTreeEntry not yet supported: "
                       << use_tree_entry->ToString();
          },
      },
      td.value());
}

absl::StatusOr<std::variant<Module*, EnumDef*, BuiltinNameDef*,
                            ArrayTypeAnnotation*, Impl*>>
ResolveColonRefSubjectAfterTypeChecking(ImportData* import_data,
                                        const TypeInfo* type_info,
                                        const ColonRef* colon_ref) {
  XLS_ASSIGN_OR_RETURN(
      ColonRefSubjectT result,
      ResolveColonRefSubjectForTypeChecking(import_data, type_info, colon_ref));
  using ReturnT =
      absl::StatusOr<std::variant<Module*, EnumDef*, BuiltinNameDef*,
                                  ArrayTypeAnnotation*, Impl*>>;
  return absl::visit(
      Visitor{
          [](Module* x) -> ReturnT { return x; },
          [](EnumDef* x) -> ReturnT { return x; },
          [](BuiltinNameDef* x) -> ReturnT { return x; },
          [](ArrayTypeAnnotation* x) -> ReturnT { return x; },
          [](StructDef* x) -> ReturnT {
            std::optional<Impl*> impl = x->impl();
            XLS_RET_CHECK(impl.has_value());
            return impl.value();
          },
          [&](TypeRefTypeAnnotation* x) -> ReturnT {
            const TypeInfo* ti = *(import_data->GetRootTypeInfo(x->owner()));
            XLS_ASSIGN_OR_RETURN(
                StructDef * struct_def,
                DerefToStruct(colon_ref->span(), x->ToString(), *x, ti));
            XLS_RET_CHECK(struct_def->impl().has_value());
            return struct_def->impl().value();
          },
          [](ColonRef*) -> ReturnT {
            return absl::InternalError(
                "After type checking colon-ref subject cannot be a ColonRef");
          },
      },
      result);
}

absl::StatusOr<Function*> ResolveFunction(Expr* callee,
                                          const TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    // If the definer is a UseTreeEntry, we need to resolve the function from
    // the module that the UseTreeEntry is in.
    if (std::holds_alternative<const NameDef*>(name_ref->name_def())) {
      const NameDef* name_def = std::get<const NameDef*>(name_ref->name_def());
      if (auto* use_tree_entry =
              dynamic_cast<UseTreeEntry*>(name_def->definer());
          use_tree_entry != nullptr) {
        XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported_info,
                             type_info->GetImportedOrError(use_tree_entry));
        return imported_info->module->GetMemberOrError<Function>(
            name_ref->identifier());
      }
    }

    return name_ref->owner()->GetMemberOrError<Function>(
        name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  std::optional<ImportSubject> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported_info,
                       type_info->GetImportedOrError(import.value()));
  return imported_info->module->GetMemberOrError<Function>(colon_ref->attr());
}

absl::StatusOr<Proc*> ResolveProc(Expr* callee, const TypeInfo* type_info) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(callee); name_ref != nullptr) {
    return name_ref->owner()->GetMemberOrError<Proc>(name_ref->identifier());
  }

  auto* colon_ref = dynamic_cast<ColonRef*>(callee);
  XLS_RET_CHECK_NE(colon_ref, nullptr);
  std::optional<ImportSubject> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value())
      << "ColonRef did not refer to an import: " << colon_ref->ToString();
  XLS_ASSIGN_OR_RETURN(const ImportedInfo* imported_info,
                       type_info->GetImportedOrError(import.value()));
  return imported_info->module->GetMemberOrError<Proc>(colon_ref->attr());
}

absl::StatusOr<StartAndWidth> ResolveBitSliceIndices(
    int64_t bit_count, std::optional<int64_t> start_opt,
    std::optional<int64_t> limit_opt) {
  XLS_RET_CHECK_GE(bit_count, 0);
  int64_t start = 0;
  int64_t limit = bit_count;

  if (start_opt.has_value()) {
    start = *start_opt;
  }
  if (limit_opt.has_value()) {
    limit = *limit_opt;
  }

  if (start < 0) {
    start += bit_count;
  }
  if (limit < 0) {
    limit += bit_count;
  }

  limit = std::clamp(limit, int64_t{0}, bit_count);
  start = std::clamp(start, int64_t{0}, limit);
  return StartAndWidth{.start = start, .width = limit - start};
}

absl::StatusOr<std::unique_ptr<Type>> ParametricBindingToType(
    const ParametricBinding& binding, DeduceCtx* ctx) {
  Module* binding_module = binding.owner();
  ImportData* import_data = ctx->import_data();
  XLS_ASSIGN_OR_RETURN(TypeInfo * binding_type_info,
                       import_data->GetRootTypeInfo(binding_module));

  auto binding_ctx =
      ctx->MakeCtxWithSameFnStack(binding_type_info, binding_module);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> metatype,
                       binding_ctx->Deduce(binding.type_annotation()));
  return UnwrapMetaType(std::move(metatype), binding.type_annotation()->span(),
                        "parametric binding type", ctx->file_table());
}

absl::StatusOr<std::vector<ParametricWithType>> ParametricBindingsToTyped(
    absl::Span<ParametricBinding* const> bindings, DeduceCtx* ctx) {
  std::vector<ParametricWithType> typed_parametrics;
  for (ParametricBinding* binding : bindings) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> binding_type,
                         ParametricBindingToType(*binding, ctx));
    typed_parametrics.push_back(
        ParametricWithType(*binding, std::move(binding_type)));
  }
  return typed_parametrics;
}

absl::StatusOr<StructDef*> DerefToStruct(const Span& span,
                                         std::string_view original_ref_text,
                                         TypeDefinition current,
                                         const TypeInfo* type_info) {
  const FileTable& file_table = type_info->file_table();
  while (true) {
    StructDef* retval = nullptr;

    // This visitor populates `retval` if we're done, otherwise updates
    // `current` or gives an error status.
    absl::Status status = absl::visit(
        Visitor{
            [&](StructDef* n) -> absl::Status {  // Done dereferencing.
              retval = n;
              return absl::OkStatus();
            },
            [&](ProcDef* n) -> absl::Status {
              // TODO: https://github.com/google/xls/issues/836 - Support this.
              return absl::InvalidArgumentError(
                  "Type deduction for impl-style procs is not yet supported.");
            },
            [&](TypeAlias* type_alias) -> absl::Status {
              TypeAnnotation& annotation = type_alias->type_annotation();
              TypeRefTypeAnnotation* type_ref =
                  dynamic_cast<TypeRefTypeAnnotation*>(&annotation);
              if (type_ref == nullptr) {
                return TypeInferenceErrorStatus(
                    span, nullptr,
                    absl::StrFormat(
                        "Could not resolve struct from %s; found: %s @ %s",
                        original_ref_text, annotation.ToString(),
                        annotation.span().ToString(file_table)),
                    file_table);
              }
              current = type_ref->type_ref()->type_definition();
              return absl::OkStatus();
            },
            [&](ColonRef* colon_ref) -> absl::Status {
              // Colon ref has to be dereferenced, may be a module reference.
              ColonRef::Subject subject = colon_ref->subject();
              // TODO(leary): 2020-12-12 Original logic was this way, but we
              // should be able to violate this assertion.
              XLS_RET_CHECK(std::holds_alternative<NameRef*>(subject));
              auto* name_ref = std::get<NameRef*>(subject);
              AnyNameDef any_name_def = name_ref->name_def();
              XLS_RET_CHECK(
                  std::holds_alternative<const NameDef*>(any_name_def));
              const NameDef* name_def = std::get<const NameDef*>(any_name_def);
              AstNode* definer = name_def->definer();
              auto* import = dynamic_cast<Import*>(definer);
              if (import == nullptr) {
                return TypeInferenceErrorStatus(
                    span, nullptr,
                    absl::StrFormat(
                        "Could not resolve struct from %s; found: %s @ %s",
                        original_ref_text, name_ref->ToString(),
                        name_ref->span().ToString(file_table)),
                    file_table);
              }
              std::optional<const ImportedInfo*> imported =
                  type_info->GetImported(import);
              XLS_RET_CHECK(imported.has_value());
              Module* module = imported.value()->module;
              XLS_ASSIGN_OR_RETURN(
                  current, module->GetTypeDefinition(colon_ref->attr()));
              XLS_ASSIGN_OR_RETURN(
                  retval, DerefToStruct(span, original_ref_text, current,
                                        imported.value()->type_info));
              return absl::OkStatus();
            },
            [&](EnumDef* enum_def) {
              return TypeInferenceErrorStatus(
                  span, nullptr,
                  absl::StrFormat(
                      "Expected struct reference, but found enum: %s",
                      enum_def->identifier()),
                  file_table);
            },
            [&](UseTreeEntry* use_tree_entry) -> absl::Status {
              XLS_ASSIGN_OR_RETURN(
                  const ImportedInfo* imported,
                  type_info->GetImportedOrError(use_tree_entry));
              std::string_view identifier =
                  use_tree_entry->GetLeafNameDef().value()->identifier();
              const Module& module = *imported->module;
              XLS_ASSIGN_OR_RETURN(current,
                                   module.GetTypeDefinition(identifier));
              XLS_ASSIGN_OR_RETURN(
                  retval, DerefToStruct(span, original_ref_text, current,
                                        imported->type_info));
              return absl::OkStatus();
            }},
        current);
    XLS_RETURN_IF_ERROR(status);
    if (retval != nullptr) {
      return retval;
    }
  }
}

absl::StatusOr<StructDef*> DerefToStruct(const Span& span,
                                         std::string_view original_ref_text,
                                         const TypeAnnotation& type_annotation,
                                         const TypeInfo* type_info) {
  const FileTable& file_table = type_info->file_table();
  auto* type_ref_type_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(&type_annotation);
  if (type_ref_type_annotation == nullptr) {
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat("Could not resolve struct from %s (%s) @ %s",
                        type_annotation.ToString(),
                        type_annotation.GetNodeTypeName(),
                        type_annotation.span().ToString(file_table)),
        file_table);
  }

  return DerefToStruct(span, original_ref_text,
                       type_ref_type_annotation->type_ref()->type_definition(),
                       type_info);
}

static int64_t Size(TupleTypeOrAnnotation t) {
  if (std::holds_alternative<const TupleType*>(t)) {
    return std::get<const TupleType*>(t)->size();
  }
  return std::get<const TupleTypeAnnotation*>(t)->size();
}

absl::Status TypeOrAnnotationErrorStatus(Span span, TupleTypeOrAnnotation type,
                                         std::string_view message,
                                         const FileTable& file_table) {
  if (std::holds_alternative<const TupleType*>(type)) {
    return TypeInferenceErrorStatus(span, std::get<const TupleType*>(type),
                                    message, file_table);
  }
  return TypeInferenceErrorStatusForAnnotation(
      span, std::get<const TupleTypeAnnotation*>(type), message, file_table);
}

absl::StatusOr<std::pair<int64_t, int64_t>> GetTupleSizes(
    const NameDefTree* name_def_tree, TupleTypeOrAnnotation tuple_type) {
  const FileTable& file_table = *name_def_tree->owner()->file_table();
  bool rest_of_tuple_found = false;
  for (const NameDefTree* node : name_def_tree->nodes()) {
    if (node->IsRestOfTupleLeaf()) {
      if (rest_of_tuple_found) {
        return TypeOrAnnotationErrorStatus(
            node->span(), tuple_type,
            absl::StrFormat("`..` can only be used once per tuple pattern."),
            file_table);
      }
      rest_of_tuple_found = true;
    }
  }
  int64_t number_of_tuple_elements = Size(tuple_type);
  int64_t number_of_names = name_def_tree->nodes().size();
  bool number_mismatch = number_of_names != number_of_tuple_elements;
  if (rest_of_tuple_found) {
    // There's a "rest of tuple" in the name def tree; we only need to have
    // enough tuple elements to bind to the required names.
    // Subtract 1 for the  ".."
    number_of_names--;
    number_mismatch = number_of_names > number_of_tuple_elements;
  }
  if (number_mismatch) {
    return TypeOrAnnotationErrorStatus(
        name_def_tree->span(), tuple_type,
        absl::StrFormat("Cannot match a %d-element tuple to %d values.",
                        number_of_tuple_elements, number_of_names),
        file_table);
  }
  return std::make_pair(number_of_tuple_elements, number_of_names);
}

static absl::StatusOr<TupleTypeOrAnnotation> GetTupleType(
    TypeOrAnnotation type, Span span, const FileTable& file_table) {
  std::string error_msg = "Expected a tuple type for these names, but got";
  if (std::holds_alternative<const Type*>(type)) {
    const Type* t = std::get<const Type*>(type);
    if (auto* tuple_type = dynamic_cast<const TupleType*>(t)) {
      return tuple_type;
    }
    return TypeInferenceErrorStatus(
        span, t, absl::StrFormat("%s %s", error_msg, t->ToString()),
        file_table);
  }
  const TypeAnnotation* type_annot = std::get<const TypeAnnotation*>(type);
  if (auto* tuple_type_annotation =
          dynamic_cast<const TupleTypeAnnotation*>(type_annot)) {
    return tuple_type_annotation;
  }
  return TypeInferenceErrorStatusForAnnotation(
      span, type_annot,
      absl::StrFormat("%s %s", error_msg, type_annot->ToString()), file_table);
}

static TypeOrAnnotation GetSubType(TupleTypeOrAnnotation type, int64_t index) {
  if (std::holds_alternative<const TupleType*>(type)) {
    auto* tuple_type = std::get<const TupleType*>(type);
    return &(tuple_type->GetMemberType(index));
  }
  return std::get<const TupleTypeAnnotation*>(type)->members()[index];
}

absl::Status MatchTupleNodeToType(
    std::function<absl::Status(AstNode*, TypeOrAnnotation,
                               std::optional<InterpValue>)>
        process_tuple_member,
    const NameDefTree* name_def_tree, const TypeOrAnnotation type,
    const FileTable& file_table, std::optional<InterpValue> constexpr_value) {
  if (name_def_tree->is_leaf()) {
    AstNode* name_def = ToAstNode(name_def_tree->leaf());
    XLS_RETURN_IF_ERROR(process_tuple_member(name_def, type, constexpr_value));
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(TupleTypeOrAnnotation tuple_type,
                       GetTupleType(type, name_def_tree->span(), file_table));

  XLS_ASSIGN_OR_RETURN((auto [number_of_tuple_elements, number_of_names]),
                       GetTupleSizes(name_def_tree, tuple_type));
  // Index into the current tuple type.
  int64_t tuple_index = 0;
  // Must iterate through the actual nodes size, not number_of_names, because
  // there may be a "rest of tuple" leaf which decreases the number of names.
  for (int64_t name_index = 0; name_index < name_def_tree->nodes().size();
       ++name_index) {
    NameDefTree* subtree = name_def_tree->nodes()[name_index];
    if (subtree->IsRestOfTupleLeaf()) {
      // Skip ahead.
      tuple_index += number_of_tuple_elements - number_of_names;
      continue;
    }
    TypeOrAnnotation subtype = GetSubType(tuple_type, tuple_index);
    XLS_RETURN_IF_ERROR(process_tuple_member(subtree, subtype, std::nullopt));

    std::optional<InterpValue> sub_value;
    if (constexpr_value.has_value()) {
      sub_value = constexpr_value.value().GetValuesOrDie()[tuple_index];
    }
    XLS_RETURN_IF_ERROR(MatchTupleNodeToType(process_tuple_member, subtree,
                                             subtype, file_table, sub_value));

    ++tuple_index;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Type>> ConcretizeBuiltinTypeAnnotation(
    const BuiltinTypeAnnotation& annotation, const FileTable& file_table) {
  if (annotation.builtin_type() == BuiltinType::kToken) {
    return std::make_unique<TokenType>();
  }
  absl::StatusOr<bool> signedness = annotation.GetSignedness();
  if (!signedness.ok()) {
    return TypeInferenceErrorStatus(
        annotation.span(), nullptr,
        absl::StrFormat("Could not determine signedness to turn "
                        "`%s` into a concrete bits type.",
                        annotation.ToString()),
        file_table);
  }
  int64_t bit_count = annotation.GetBitCount();
  return std::make_unique<BitsType>(signedness.value(), bit_count);
}

absl::StatusOr<std::optional<Function*>> ImplFnFromCallee(
    const Attr* attr, const TypeInfo* type_info) {
  if (!type_info->Contains(attr->lhs())) {
    return std::nullopt;
  }
  std::optional<Type*> type = type_info->GetItem(attr->lhs());
  if (!type.has_value()) {
    return std::nullopt;
  }
  if (!(*type)->IsStruct()) {
    return TypeInferenceErrorStatus(
        attr->span(), nullptr,
        absl::StrFormat("Cannot invoke method `%s` on non-struct type `%s`",
                        attr->attr(), (*type)->ToString()),
        type_info->file_table());
  }
  StructDef sd = (*type)->AsStruct().nominal_type();
  return sd.GetImplFunction(attr->attr());
}

bool IsAcceptableCast(const Type& from, const Type& to) {
  auto is_enum = [](const Type& ct) -> bool {
    return dynamic_cast<const EnumType*>(&ct) != nullptr;
  };
  auto is_bits_array = [&](const Type& ct) -> bool {
    const ArrayType* at = dynamic_cast<const ArrayType*>(&ct);
    if (at == nullptr) {
      return false;
    }
    if (IsBitsLike(at->element_type())) {
      return true;
    }
    return false;
  };
  if ((is_bits_array(from) && IsBitsLike(to)) ||
      (IsBitsLike(from) && is_bits_array(to))) {
    TypeDim from_total_bit_count = from.GetTotalBitCount().value();
    TypeDim to_total_bit_count = to.GetTotalBitCount().value();
    return from_total_bit_count == to_total_bit_count;
  }
  if ((IsBitsLike(from) || is_enum(from)) && IsBitsLike(to)) {
    return true;
  }
  if (IsBitsLike(from) && is_enum(to)) {
    return true;
  }
  return false;
}

absl::Status NoteBuiltinInvocationConstExpr(std::string_view fn_name,
                                            const Invocation* invocation,
                                            const FunctionType& fn_type,
                                            TypeInfo* ti,
                                            ImportData* import_data) {
  // array_size is always a constexpr result since it just needs the type
  // information
  if (fn_name == "array_size") {
    auto* array_type = down_cast<const ArrayType*>(fn_type.params()[0].get());
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    ti->NoteConstExpr(invocation,
                      InterpValue::MakeU32(static_cast<int32_t>(array_size)));
  }

  // bit_count and element_count are similar to array_size, but use the
  // parametric argument rather than a value.
  if (fn_name == "bit_count" || fn_name == "element_count") {
    XLS_RET_CHECK_EQ(invocation->explicit_parametrics().size(), 1);
    std::optional<const Type*> explicit_parametric_type = ti->GetItem(
        std::get<TypeAnnotation*>(invocation->explicit_parametrics()[0]));
    XLS_RET_CHECK(explicit_parametric_type.has_value());
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        fn_name == "element_count"
            ? GetElementCountAsInterpValue(*explicit_parametric_type)
            : GetBitCountAsInterpValue(*explicit_parametric_type));
    ti->NoteConstExpr(invocation, value);
  }

  if (fn_name == "configured_value_or") {
    XLS_RET_CHECK_EQ(invocation->args().size(), 2);
    std::string key = down_cast<const String*>(invocation->args()[0])->text();
    if (!invocation->owner()->configured_values().contains(key)) {
      Expr* default_value_expr = invocation->args()[1];
      XLS_ASSIGN_OR_RETURN(InterpValue default_value,
                           ti->GetConstExpr(default_value_expr));
      ti->NoteConstExpr(invocation, default_value);
      return absl::OkStatus();
    }
    std::optional<std::string> override_str =
        invocation->owner()->configured_values().at(key);
    std::optional<const Type*> explicit_parametric_type = ti->GetItem(
        std::get<TypeAnnotation*>(invocation->explicit_parametrics()[0]));
    XLS_ASSIGN_OR_RETURN(
        InterpValue result_value,
        GetConfiguredValueAsInterpValue(override_str.value(),
                                        explicit_parametric_type.value(), ti,
                                        import_data, invocation->span()));
    ti->NoteConstExpr(invocation, result_value);
  }
  return absl::OkStatus();
}

const TypeInfo& GetTypeInfoForNodeIfDifferentModule(
    AstNode* node, const TypeInfo& current_type_info,
    const ImportData& import_data) {
  if (node->owner() == current_type_info.module()) {
    return current_type_info;
  }
  absl::StatusOr<const TypeInfo*> type_info =
      import_data.GetRootTypeInfoForNode(node);
  CHECK_OK(type_info.status())
      << "Must be able to get root type info for node " << node->ToString();
  CHECK(type_info.value() != nullptr);
  return *type_info.value();
}

void WarnOnInappropriateConstantName(std::string_view identifier,
                                     const Span& span, const Module& module,
                                     WarningCollector* warning_collector) {
  if (!IsScreamingSnakeCase(identifier) &&
      !module.attributes().contains(
          ModuleAttribute::kAllowNonstandardConstantNaming)) {
    warning_collector->Add(
        span, WarningKind::kConstantNaming,
        absl::StrFormat("Standard style is SCREAMING_SNAKE_CASE for constant "
                        "identifiers; got: `%s`",
                        identifier));
  }
}

absl::StatusOr<InterpValue> GetBitCountAsInterpValue(const Type* type) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  XLS_ASSIGN_OR_RETURN(TypeDim bit_count_ctd, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t bit_count,
      std::get<InterpValue>(bit_count_ctd.value()).GetBitValueViaSign());
  return InterpValue::MakeU32(bit_count);
}

absl::StatusOr<InterpValue> GetElementCountAsInterpValue(const Type* type) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  if (const auto* array_type = dynamic_cast<const ArrayType*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64_t size, array_type->size().GetAsInt64());
    CHECK(static_cast<uint32_t>(size) == size);
    return InterpValue::MakeU32(size);
  }
  if (const auto* tuple_type = dynamic_cast<const TupleType*>(type)) {
    return InterpValue::MakeU32(tuple_type->members().size());
  }
  if (const auto* struct_type = dynamic_cast<const StructType*>(type)) {
    return InterpValue::MakeU32(struct_type->members().size());
  }
  return GetBitCountAsInterpValue(type);
}

absl::StatusOr<InterpValue> GetConfiguredValueAsInterpValue(
    std::string override_value, const Type* type, const TypeInfo* type_info,
    ImportData* import_data, const Span& span) {
  if (type->IsMeta()) {
    XLS_ASSIGN_OR_RETURN(type, UnwrapMetaType(*type));
  }
  if (IsBool(*type)) {
    return InterpValue::MakeBool(override_value == "true");
  }

  if (const auto* enum_type = dynamic_cast<const EnumType*>(type)) {
    XLS_RET_CHECK(type_info != nullptr);
    XLS_RET_CHECK(import_data != nullptr);
    const EnumDef* enum_def = &enum_type->nominal_type();
    const TypeInfo& enum_type_info = GetTypeInfoForNodeIfDifferentModule(
        const_cast<EnumDef*>(enum_def), *type_info, *import_data);
    std::vector<std::string_view> parts = absl::StrSplit(override_value, "::");
    std::string_view enum_identifier = parts.back();
    for (const EnumMember& member : enum_def->values()) {
      if (member.name_def->identifier() == enum_identifier) {
        XLS_ASSIGN_OR_RETURN(InterpValue value,
                             enum_type_info.GetConstExpr(member.value));
        return InterpValue::MakeEnum(value.GetBitsOrDie(), enum_type, enum_def);
      }
    }
    return TypeInferenceErrorStatus(
        span, nullptr,
        absl::StrFormat("Invalid value \'%s\' for enum %s", override_value,
                        enum_type->ToString()),
        import_data->file_table());
  }

  if (const auto* bits_type = dynamic_cast<const BitsType*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type->size().GetAsInt64());
    bool is_signed = bits_type->is_signed();
    std::string_view value_str = override_value;
    if (auto pos = value_str.find(':'); pos != std::string_view::npos) {
      value_str = value_str.substr(pos + 1);
    }
    XLS_ASSIGN_OR_RETURN(Bits bits, ParseNumber(value_str));

    if (bits.bit_count() > bit_count) {
      return TypeInferenceErrorStatus(
          span, nullptr,
          absl::StrFormat("Parsed value \'%s\' (which is %d bits) is too large "
                          "for type %s (which is %d bits)",
                          override_value, bits.bit_count(),
                          bits_type->ToString(), bit_count),
          import_data->file_table());
    }

    if (bits.bit_count() < bit_count) {
      if (is_signed) {
        bits = bits_ops::SignExtend(bits, bit_count);
      } else {
        bits = bits_ops::ZeroExtend(bits, bit_count);
      }
    }
    return InterpValue::MakeBits(is_signed, bits);
  }

  return TypeInferenceErrorStatus(
      span, nullptr,
      absl::StrFormat("Unsupported configured value type: %s for: \'%s\'",
                      type->ToString(), override_value),
      import_data->file_table());
}

std::string PatternsToString(const MatchArm* arm) {
  return absl::StrJoin(arm->patterns(), " | ",
                       [](std::string* out, NameDefTree* ndt) {
                         absl::StrAppend(out, ndt->ToString());
                       });
}

}  // namespace xls::dslx
