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

#include "xls/dslx/cpp_evaluate.h"

#include "xls/common/status/ret_check.h"
#include "xls/dslx/type_info.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

using Value = InterpValue;
using Tag = InterpValueTag;

}  // namespace

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateNumber(Number* expr,
                                           InterpBindings* bindings,
                                           ConcreteType* type_context,
                                           InterpCallbackData* callbacks) {
  XLS_VLOG(4) << "Evaluating number: " << expr->ToString() << " @ "
              << expr->span();
  std::unique_ptr<ConcreteType> type_context_value;
  if (type_context == nullptr && expr->kind() == NumberKind::kCharacter) {
    type_context_value = BitsType::MakeU8();
    type_context = type_context_value.get();
  }
  if (type_context == nullptr && expr->kind() == NumberKind::kBool) {
    type_context_value = BitsType::MakeU1();
    type_context = type_context_value.get();
  }
  if (type_context == nullptr && expr->type() == nullptr) {
    return absl::InternalError(
        absl::StrFormat("FailureError: %s No type context for expression, "
                        "should be caught by type inference.",
                        expr->span().ToString()));
  }
  if (type_context == nullptr) {
    XLS_ASSIGN_OR_RETURN(
        type_context_value,
        ConcretizeTypeAnnotation(expr->type(), bindings, callbacks));
    type_context = type_context_value.get();
  }

  BitsType* bits_type = dynamic_cast<BitsType*>(type_context);
  XLS_RET_CHECK(bits_type != nullptr)
      << "Type for number should be 'bits' kind.";
  InterpValueTag tag =
      bits_type->is_signed() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  int64 bit_count = absl::get<int64>(bits_type->size().value());
  XLS_ASSIGN_OR_RETURN(Bits bits, expr->GetBits(bit_count));
  return InterpValue::MakeBits(tag, std::move(bits));
}

static absl::StatusOr<EnumDef*> EvaluateToEnum(TypeDefinition type_definition,
                                               InterpBindings* bindings,
                                               InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(
      DerefVariant deref,
      EvaluateToStructOrEnumOrAnnotation(type_definition, bindings, callbacks));
  if (absl::holds_alternative<EnumDef*>(deref)) {
    return absl::get<EnumDef*>(deref);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type definition did not dereference to an enum, found: ",
                   ToAstNode(deref)->GetNodeTypeName()));
}

static absl::StatusOr<StructDef*> EvaluateToStruct(
    StructRef struct_ref, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       ToTypeDefinition(ToAstNode(struct_ref)));
  XLS_ASSIGN_OR_RETURN(
      DerefVariant deref,
      EvaluateToStructOrEnumOrAnnotation(type_definition, bindings, callbacks));
  if (absl::holds_alternative<StructDef*>(deref)) {
    return absl::get<StructDef*>(deref);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type definition did not dereference to a struct, found: ",
                   ToAstNode(deref)->GetNodeTypeName()));
}

absl::StatusOr<InterpValue> EvaluateXlsTuple(XlsTuple* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             InterpCallbackData* callbacks) {
  auto get_type_context =
      [type_context](int64 i) -> std::unique_ptr<ConcreteType> {
    if (type_context == nullptr) {
      return nullptr;
    }
    auto tuple_type_context = dynamic_cast<TupleType*>(type_context);
    return tuple_type_context->GetMemberType(i).CloneToUnique();
  };

  std::vector<InterpValue> members;
  for (int64 i = 0; i < expr->members().size(); ++i) {
    Expr* m = expr->members()[i];
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         callbacks->Eval(m, bindings, get_type_context(i)));
    members.push_back(std::move(value));
  }
  return InterpValue::MakeTuple(std::move(members));
}

// Turns an enum to corresponding (bits) concrete type (w/signedness).
//
// For example, used in conversion checks.
//
// Args:
//  enum_def: AST node (enum definition) to convert.
//  bit_count: The bit count of the underlying bits type for the enum
//    definition, as determined by type inference or interpretation.
static std::unique_ptr<ConcreteType> StrengthReduceEnum(EnumDef* enum_def,
                                                        int64 bit_count) {
  bool is_signed = enum_def->signedness().value();
  return absl::make_unique<BitsType>(is_signed, bit_count);
}

// Returns the concrete type of 'value'.
//
// Note that:
// * Non-zero-length arrays are assumed (for zero length arrays we can't
//    currently deduce the type from the value because the concrete element type
//    is not reified in the array value.
// * Enums are strength-reduced to their underlying bits (storage) type.
//
// Args:
//   value: Value to determine the concrete type for.
static std::unique_ptr<ConcreteType> ConcreteTypeFromValue(
    const InterpValue& value) {
  switch (value.tag()) {
    case InterpValueTag::kUBits:
    case InterpValueTag::kSBits: {
      bool signedness = value.tag() == InterpValueTag::kSBits;
      return absl::make_unique<BitsType>(signedness,
                                         value.GetBitCount().value());
    }
    case InterpValueTag::kArray: {
      std::unique_ptr<ConcreteType> element_type;
      if (value.GetLength().value() == 0) {
        // Can't determine the type from the value of a 0-element array, so we
        // just use nil.
        element_type = ConcreteType::MakeNil();
      } else {
        element_type =
            ConcreteTypeFromValue(value.Index(Value::MakeU32(0)).value());
      }
      return std::make_unique<ArrayType>(
          std::move(element_type), ConcreteTypeDim(value.GetLength().value()));
    }
    case InterpValueTag::kTuple: {
      std::vector<std::unique_ptr<ConcreteType>> members;
      for (const InterpValue& m : value.GetValuesOrDie()) {
        members.push_back(ConcreteTypeFromValue(m));
      }
      return absl::make_unique<TupleType>(std::move(members));
    }
    case InterpValueTag::kEnum:
      return StrengthReduceEnum(value.type(), value.GetBitCount().value());
    case InterpValueTag::kFunction:
      break;
  }
  XLS_LOG(FATAL) << "Invalid value tag for ConcreteTypeFromValue: "
                 << static_cast<int64>(value.tag());
}

absl::StatusOr<bool> ValueCompatibleWithType(const ConcreteType& type,
                                             const InterpValue& value) {
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&type)) {
    XLS_RET_CHECK_EQ(value.tag(), InterpValueTag::kTuple);
    int64 member_count = tuple_type->size();
    const auto& elements = value.GetValuesOrDie();
    if (member_count != elements.size()) {
      return false;
    }
    for (int64 i = 0; i < member_count; ++i) {
      const ConcreteType& member_type = tuple_type->GetMemberType(i);
      const InterpValue& member_value = elements[i];
      XLS_ASSIGN_OR_RETURN(bool compatible,
                           ValueCompatibleWithType(member_type, member_value));
      if (!compatible) {
        return false;
      }
    }
    return true;
  }

  if (auto* array_type = dynamic_cast<const ArrayType*>(&type)) {
    // For arrays, we check the first value in the array conforms to the element
    // type to determine compatibility.
    const ConcreteType& element_type = array_type->element_type();
    const auto& elements = value.GetValuesOrDie();
    int64 array_size = absl::get<int64>(array_type->size().value());
    if (elements.size() != array_size) {
      return false;
    }
    if (elements.empty()) {
      return true;
    }
    const InterpValue& member_value = elements[0];
    XLS_ASSIGN_OR_RETURN(bool compatible,
                         ValueCompatibleWithType(element_type, member_value));
    return compatible;
  }

  // For enum type and enum value we just compare the nominal type to see if
  // they're identical.
  if (auto* enum_type = dynamic_cast<const EnumType*>(&type);
      enum_type != nullptr && value.tag() == InterpValueTag::kEnum) {
    return enum_type->nominal_type() == value.type();
  }

  if (auto* bits_type = dynamic_cast<const BitsType*>(&type)) {
    if (!bits_type->is_signed() && value.tag() == InterpValueTag::kUBits) {
      return value.GetBitCount().value() ==
             absl::get<int64>(bits_type->size().value());
    }

    if (bits_type->is_signed() && value.tag() == InterpValueTag::kSBits) {
      return value.GetBitCount().value() ==
             absl::get<int64>(bits_type->size().value());
    }

    // Enum values can be converted to bits type if the signedness/bit counts
    // line up.
    if (value.tag() == InterpValueTag::kEnum) {
      return value.type()->signedness().value() == bits_type->is_signed() &&
             value.GetBitCount().value() ==
                 absl::get<int64>(bits_type->size().value());
    }

    // Arrays can be converted to unsigned bits types by flattening, but we must
    // check the flattened bit count is the same as the target bit type.
    if (!bits_type->is_signed() && value.tag() == InterpValueTag::kArray) {
      int64 flat_bit_count = value.Flatten().value().GetBitCount().value();
      return flat_bit_count == absl::get<int64>(bits_type->size().value());
    }
  }

  if (auto* enum_type = dynamic_cast<const EnumType*>(&type);
      enum_type != nullptr && value.IsBits()) {
    return enum_type->signedness().value() ==
               (value.tag() == InterpValueTag::kSBits) &&
           enum_type->GetTotalBitCount().value() == value.GetBitCount().value();
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Cannot determine type/value compatibility; type: %s value: %s",
      type.ToString(), value.ToString()));
}

absl::StatusOr<bool> ConcreteTypeAcceptsValue(const ConcreteType& type,
                                              const InterpValue& value) {
  switch (value.tag()) {
    case InterpValueTag::kUBits: {
      auto* bits_type = dynamic_cast<const BitsType*>(&type);
      if (bits_type == nullptr) {
        return false;
      }
      return !bits_type->is_signed() &&
             value.GetBitCount().value() ==
                 absl::get<int64>(bits_type->size().value());
    }
    case InterpValueTag::kSBits: {
      auto* bits_type = dynamic_cast<const BitsType*>(&type);
      if (bits_type == nullptr) {
        return false;
      }
      return bits_type->is_signed() &&
             value.GetBitCount().value() ==
                 absl::get<int64>(bits_type->size().value());
    }
    case InterpValueTag::kArray:
    case InterpValueTag::kTuple:
    case InterpValueTag::kEnum:
      // TODO(leary): 2020-11-16 We should be able to use stricter checks here
      // than "can I cast value to type".
      return ValueCompatibleWithType(type, value);
    case InterpValueTag::kFunction:
      break;
  }
  return absl::UnimplementedError(
      absl::StrCat("ConcreteTypeAcceptsValue not implemented for tag: ",
                   static_cast<int64>(value.tag())));
}

// Converts 'value' into a value of "type".
absl::StatusOr<InterpValue> ConcreteTypeConvertValue(
    const ConcreteType& type, const InterpValue& value, const Span& span,
    absl::optional<std::vector<InterpValue>> enum_values,
    absl::optional<bool> enum_signed) {
  // Converting unsigned bits to an array.
  if (auto* array_type = dynamic_cast<const ArrayType*>(&type);
      array_type != nullptr && value.IsUBits()) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim element_bit_count,
                         array_type->element_type().GetTotalBitCount());
    int64 bits_per_element = absl::get<int64>(element_bit_count.value());
    const Bits& bits = value.GetBitsOrDie();

    auto bit_slice_value_at_index = [&](int64 i) -> InterpValue {
      int64 lo = i * bits_per_element;
      Bits rev = bits_ops::Reverse(bits);
      Bits slice = rev.Slice(lo, bits_per_element);
      Bits result = bits_ops::Reverse(slice);
      return InterpValue::MakeBits(InterpValueTag::kUBits, result).value();
    };

    std::vector<InterpValue> values;
    for (int64 i = 0; i < absl::get<int64>(array_type->size().value()); ++i) {
      values.push_back(bit_slice_value_at_index(i));
    }

    return Value::MakeArray(values);
  }

  // Converting bits-having-thing into an enum.
  if (auto* enum_type = dynamic_cast<const EnumType*>(&type);
      enum_type != nullptr && value.HasBits() &&
      value.GetBitCount().value() ==
          absl::get<int64>(type.GetTotalBitCount().value().value())) {
    EnumDef* enum_def = enum_type->nominal_type();
    bool found = false;
    for (const InterpValue& enum_value : *enum_values) {
      if (value.GetBitsOrDie() == enum_value.GetBitsOrDie()) {
        found = true;
        break;
      }
    }
    if (!found) {
      return absl::InternalError(absl::StrFormat(
          "FailureError: %s Value is not valid for enum %s: %s",
          span.ToString(), enum_def->identifier(), value.ToString()));
    }
    return Value::MakeEnum(value.GetBitsOrDie(), enum_def);
  }

  // Converting enum value to bits.
  if (auto* bits_type = dynamic_cast<const BitsType*>(&type);
      value.IsEnum() && bits_type != nullptr &&
      type.GetTotalBitCount().value() == value.GetBitCount().value()) {
    auto tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                      : InterpValueTag::kUBits;
    return InterpValue::MakeBits(tag, value.GetBitsOrDie());
  }

  auto zero_ext = [&]() -> InterpValue {
    auto* bits_type = dynamic_cast<const BitsType*>(&type);
    XLS_CHECK(bits_type != nullptr);
    InterpValueTag tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                                : InterpValueTag::kUBits;
    int64 bit_count = absl::get<int64>(bits_type->size().value());
    return InterpValue::MakeBits(tag, value.ZeroExt(bit_count)->GetBitsOrDie())
        .value();
  };

  auto sign_ext = [&]() -> InterpValue {
    auto* bits_type = dynamic_cast<const BitsType*>(&type);
    XLS_CHECK(bits_type != nullptr);
    InterpValueTag tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                                : InterpValueTag::kUBits;
    int64 bit_count = absl::get<int64>(bits_type->size().value());
    return InterpValue::MakeBits(
               tag, value.SignExt(bit_count).value().GetBitsOrDie())
        .value();
  };

  if (value.IsUBits()) {
    return zero_ext();
  }

  if (value.IsSBits()) {
    return sign_ext();
  }

  if (value.IsEnum()) {
    return enum_signed.value() ? sign_ext() : zero_ext();
  }

  if (value.IsArray() && dynamic_cast<const BitsType*>(&type) != nullptr) {
    return value.Flatten();
  }

  XLS_ASSIGN_OR_RETURN(bool type_accepts_value,
                       ConcreteTypeAcceptsValue(type, value));
  if (type_accepts_value) {
    // Vacuous conversion.
    return value;
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("FailureError: %s Cannot convert value %s to type %s",
                      span.ToString(), value.ToString(), type.ToString()));
}

// Retrieves the flat/evaluated members of enum if type_ is an EnumType.
static absl::StatusOr<absl::optional<std::vector<InterpValue>>> GetEnumValues(
    const ConcreteType& type, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  auto* enum_type = dynamic_cast<const EnumType*>(&type);
  if (enum_type == nullptr) {
    return absl::nullopt;
  }

  EnumDef* enum_def = enum_type->nominal_type();
  std::vector<InterpValue> result;
  for (const EnumMember& member : enum_def->values()) {
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         callbacks->Eval(ToExprNode(member.value), bindings));
    result.push_back(std::move(value));
  }
  return absl::make_optional(std::move(result));
}

absl::StatusOr<InterpValue> EvaluateArray(Array* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks) {
  std::unique_ptr<ConcreteType> type;
  if (type_context == nullptr && expr->type() != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        type, ConcretizeTypeAnnotation(expr->type(), bindings, callbacks));
    type_context = type.get();
  }

  auto* array_type = dynamic_cast<ArrayType*>(type_context);

  const ConcreteType* element_type = nullptr;
  if (type_context != nullptr) {
    // If we have a type context it must be an array.
    XLS_RET_CHECK(array_type != nullptr);
    element_type = &array_type->element_type();
  }

  std::vector<InterpValue> elements;
  elements.reserve(expr->members().size());
  for (Expr* m : expr->members()) {
    std::unique_ptr<ConcreteType> type_context;
    if (element_type != nullptr) {
      type_context = element_type->CloneToUnique();
    }
    XLS_ASSIGN_OR_RETURN(InterpValue e,
                         callbacks->Eval(m, bindings, std::move(type_context)));
    elements.push_back(std::move(e));
  }
  if (expr->has_ellipsis()) {
    XLS_RET_CHECK(array_type != nullptr);
    int64 target_size = absl::get<int64>(array_type->size().value());
    while (elements.size() < target_size) {
      elements.push_back(elements.back());
    }
  }
  return InterpValue::MakeArray(std::move(elements));
}

absl::StatusOr<InterpValue> EvaluateCast(Cast* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> type,
      ConcretizeTypeAnnotation(expr->type(), bindings, callbacks));
  XLS_ASSIGN_OR_RETURN(
      InterpValue value,
      callbacks->Eval(expr->expr(), bindings, type->CloneToUnique()));
  XLS_ASSIGN_OR_RETURN(absl::optional<std::vector<InterpValue>> enum_values,
                       GetEnumValues(*type, bindings, callbacks));
  return ConcreteTypeConvertValue(
      *type, value, expr->span(), std::move(enum_values),
      value.type() == nullptr
          ? absl::nullopt
          : absl::make_optional(value.type()->signedness().value()));
}

absl::StatusOr<InterpValue> EvaluateLet(Let* expr, InterpBindings* bindings,
                                        ConcreteType* type_context,
                                        InterpCallbackData* callbacks) {
  std::unique_ptr<ConcreteType> want_type;
  if (expr->type() != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        want_type, ConcretizeTypeAnnotation(expr->type(), bindings, callbacks));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue to_bind,
                       callbacks->Eval(expr->rhs(), bindings));
  if (want_type != nullptr) {
    XLS_ASSIGN_OR_RETURN(bool accepted,
                         ConcreteTypeAcceptsValue(*want_type, to_bind));
    if (!accepted) {
      return absl::InternalError(absl::StrFormat(
          "EvaluateError: %s Type error found! Let-expression right hand side "
          "did not "
          "conform to annotated type\n\twant: %s\n\tgot:  %s\n\tvalue: %s",
          expr->span().ToString(), want_type->ToString(),
          ConcreteTypeFromValue(to_bind)->ToString(), to_bind.ToString()));
    }
  }

  std::shared_ptr<InterpBindings> new_bindings = InterpBindings::CloneWith(
      bindings->shared_from_this(), expr->name_def_tree(), to_bind);
  return callbacks->Eval(expr->body(), new_bindings.get());
}

absl::StatusOr<InterpValue> EvaluateStructInstance(
    StructInstance* expr, InterpBindings* bindings, ConcreteType* type_context,
    InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(
      StructDef * struct_def,
      EvaluateToStruct(expr->struct_def(), bindings, callbacks));
  std::vector<InterpValue> members;
  for (auto [name, field_expr] : expr->GetOrderedMembers(struct_def)) {
    XLS_ASSIGN_OR_RETURN(InterpValue member,
                         callbacks->Eval(field_expr, bindings));
    members.push_back(member);
  }
  return InterpValue::MakeTuple(std::move(members));
}

absl::StatusOr<InterpValue> EvaluateSplatStructInstance(
    SplatStructInstance* expr, InterpBindings* bindings,
    ConcreteType* type_context, InterpCallbackData* callbacks) {
  // First we grab the "basis" struct value (the subject of the 'splat') that
  // we're going to update with the modified fields.
  XLS_ASSIGN_OR_RETURN(InterpValue named_tuple,
                       callbacks->Eval(expr->splatted(), bindings));
  XLS_ASSIGN_OR_RETURN(
      StructDef * struct_def,
      EvaluateToStruct(expr->struct_ref(), bindings, callbacks));
  for (auto [name, field_expr] : expr->members()) {
    XLS_ASSIGN_OR_RETURN(InterpValue new_value,
                         callbacks->Eval(field_expr, bindings));
    int64 i = struct_def->GetMemberIndex(name).value();
    XLS_ASSIGN_OR_RETURN(
        named_tuple, named_tuple.Update(InterpValue::MakeU64(i), new_value));
  }

  return named_tuple;
}

absl::StatusOr<InterpValue> EvaluateEnumRef(EnumRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(
      EnumDef * enum_def,
      EvaluateToEnum(ToTypeDefinition(ToAstNode(expr->enum_def())).value(),
                     bindings, callbacks));
  XLS_ASSIGN_OR_RETURN(auto value_node, enum_def->GetValue(expr->attr()));
  XLS_ASSIGN_OR_RETURN(
      InterpBindings fresh_bindings,
      MakeTopLevelBindings(expr->owner()->shared_from_this(), callbacks));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> concrete_type,
      ConcretizeTypeAnnotation(enum_def->type(), &fresh_bindings, callbacks));
  Expr* value_expr = ToExprNode(value_node);
  XLS_ASSIGN_OR_RETURN(InterpValue raw_value,
                       callbacks->Eval(value_expr, &fresh_bindings));
  return InterpValue::MakeEnum(raw_value.GetBitsOrDie(), enum_def);
}

absl::StatusOr<InterpValue> EvaluateUnop(Unop* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue arg,
                       callbacks->Eval(expr->operand(), bindings));
  switch (expr->kind()) {
    case UnopKind::kInvert:
      return arg.BitwiseNegate();
    case UnopKind::kNegate:
      return arg.ArithmeticNegate();
  }
  return absl::InternalError(absl::StrCat("Invalid unary operation kind: ",
                                          static_cast<int64>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateBinop(Binop* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, callbacks->Eval(expr->lhs(), bindings));
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, callbacks->Eval(expr->rhs(), bindings));

  // Check some preconditions; e.g. all logical operands are guaranteed to have
  // single-bit inputs by type checking so we can share the implementation with
  // bitwise or/and.
  switch (expr->kind()) {
    case BinopKind::kLogicalOr:
    case BinopKind::kLogicalAnd:
      XLS_RET_CHECK_EQ(lhs.GetBitCount().value(), 1);
      XLS_RET_CHECK_EQ(rhs.GetBitCount().value(), 1);
      break;
    default:
      break;
  }

  switch (expr->kind()) {
    case BinopKind::kAdd:
      return lhs.Add(rhs);
    case BinopKind::kSub:
      return lhs.Sub(rhs);
    case BinopKind::kConcat:
      return lhs.Concat(rhs);
    case BinopKind::kMul:
      return lhs.Mul(rhs);
    case BinopKind::kDiv:
      return lhs.FloorDiv(rhs);
    case BinopKind::kOr:
    case BinopKind::kLogicalOr:
      return lhs.BitwiseOr(rhs);
    case BinopKind::kAnd:
    case BinopKind::kLogicalAnd:
      return lhs.BitwiseAnd(rhs);
    case BinopKind::kXor:
      return lhs.BitwiseXor(rhs);
    case BinopKind::kShll:
      return lhs.Shll(rhs);
    case BinopKind::kShrl:
      return lhs.Shrl(rhs);
    case BinopKind::kShra:
      return lhs.Shra(rhs);
    case BinopKind::kEq:
      return Value::MakeBool(lhs.Eq(rhs));
    case BinopKind::kNe:
      return Value::MakeBool(lhs.Ne(rhs));
    case BinopKind::kGt:
      return lhs.Gt(rhs);
    case BinopKind::kLt:
      return lhs.Lt(rhs);
    case BinopKind::kLe:
      return lhs.Le(rhs);
    case BinopKind::kGe:
      return lhs.Ge(rhs);
  }
  return absl::InternalError(absl::StrCat("Invalid binary operation kind: ",
                                          static_cast<int64>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateTernary(Ternary* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue test,
                       callbacks->Eval(expr->test(), bindings));
  if (test.IsTrue()) {
    return callbacks->Eval(expr->consequent(), bindings);
  }
  return callbacks->Eval(expr->alternate(), bindings);
}

absl::StatusOr<InterpValue> EvaluateAttr(Attr* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, callbacks->Eval(expr->lhs(), bindings));
  std::shared_ptr<TypeInfo> type_info = callbacks->get_type_info();
  absl::optional<ConcreteType*> maybe_type = type_info->GetItem(expr->lhs());
  XLS_RET_CHECK(maybe_type.has_value()) << "LHS of attr should have type info";
  TupleType* tuple_type = dynamic_cast<TupleType*>(maybe_type.value());
  XLS_RET_CHECK(tuple_type != nullptr) << (*maybe_type)->ToString();
  absl::optional<int64> index;
  for (int64 i = 0; i < tuple_type->size(); ++i) {
    absl::string_view name = tuple_type->GetMemberName(i);
    if (name == expr->attr()->identifier()) {
      index = i;
      break;
    }
  }
  XLS_RET_CHECK(index.has_value())
      << "Unable to find attribute " << expr->attr()
      << ": should be caught by type inference";
  return lhs.GetValuesOrDie().at(*index);
}

static absl::StatusOr<InterpValue> EvaluateIndexBitSlice(
    Index* expr, InterpBindings* bindings, InterpCallbackData* callbacks,
    const Bits& bits) {
  IndexRhs index = expr->rhs();
  XLS_RET_CHECK(absl::holds_alternative<Slice*>(index));
  auto index_slice = absl::get<Slice*>(index);

  const SymbolicBindings& sym_bindings = bindings->fn_ctx()->sym_bindings;

  std::shared_ptr<TypeInfo> type_info = callbacks->get_type_info();
  absl::optional<SliceData::StartWidth> maybe_saw =
      type_info->GetSliceStartWidth(index_slice, sym_bindings);
  XLS_RET_CHECK(maybe_saw.has_value());
  const auto& saw = maybe_saw.value();
  return Value::MakeBits(Tag::kUBits, bits.Slice(saw.start, saw.width));
}

static absl::StatusOr<InterpValue> EvaluateIndexWidthSlice(
    Index* expr, InterpBindings* bindings, InterpCallbackData* callbacks,
    Bits bits) {
  auto width_slice = absl::get<WidthSlice*>(expr->rhs());
  XLS_ASSIGN_OR_RETURN(
      InterpValue start,
      callbacks->Eval(
          width_slice->start(), bindings,
          std::make_unique<BitsType>(/*is_signed=*/false, bits.bit_count())));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> width_type,
      ConcretizeTypeAnnotation(width_slice->width(), bindings, callbacks));
  XLS_ASSIGN_OR_RETURN(uint64 start_index, start.GetBitValueUint64());
  auto* width_bits_type = dynamic_cast<BitsType*>(width_type.get());
  XLS_RET_CHECK(width_bits_type != nullptr);
  int64 width_value = absl::get<int64>(width_bits_type->size().value());

  if (start_index >= bits.bit_count()) {
    // Return early  to potentially avoid an unreasonably long zero extend (e.g.
    // if the start index was a large negative number).
    return Value::MakeUBits(/*bit_count=*/width_value, /*value=*/0);
  }

  if (start_index + width_value > bits.bit_count()) {
    // Slicing off the end zero-fills, so we zext.
    bits = bits_ops::ZeroExtend(bits, start_index + width_value);
  }

  Bits result = bits.Slice(start_index, width_value);
  InterpValueTag tag = width_bits_type->is_signed() ? InterpValueTag::kSBits
                                                    : InterpValueTag::kUBits;
  return InterpValue::MakeBits(tag, result);
}

absl::StatusOr<InterpValue> EvaluateIndex(Index* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, callbacks->Eval(expr->lhs(), bindings));
  if (lhs.IsBits()) {
    if (absl::holds_alternative<Slice*>(expr->rhs())) {
      return EvaluateIndexBitSlice(expr, bindings, callbacks,
                                   lhs.GetBitsOrDie());
    }
    XLS_RET_CHECK(absl::holds_alternative<WidthSlice*>(expr->rhs()));
    return EvaluateIndexWidthSlice(expr, bindings, callbacks,
                                   lhs.GetBitsOrDie());
  }

  Expr* index = absl::get<Expr*>(expr->rhs());
  // Note: since we permit a type-unannotated literal number we provide a type
  // context here.
  XLS_ASSIGN_OR_RETURN(InterpValue index_value,
                       callbacks->Eval(index, bindings, BitsType::MakeU32()));
  XLS_ASSIGN_OR_RETURN(uint64 index_int, index_value.GetBitValueUint64());
  int64 length = lhs.GetLength().value();
  if (index_int >= length) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FailureError: %s Indexing out of bounds: %d vs size %d",
        expr->span().ToString(), index_int, length));
  }
  return lhs.GetValuesOrDie().at(index_int);
}

// Returns whether this matcher pattern is accepted.
//
// Note that some patterns don't always match -- we call those "refutable"; e.g.
// in:
//
//       match (u32:3, u32:4) {
//         (u32:2, y) => y;
//         (x, _) => x;
//       }
//
// The first pattern will not match, and so this method would return false for
// that match arm. If you had a pattern that was just `_` it would match
// everything, and thus it is "irrefutable".
//
// Args:
//  pattern: Decribes the pattern attempting to match against the value.
//  to_match: The value being matched against.
//  bindings: The bindings to populate if the pattern has bindings associated
//    with it.
static absl::StatusOr<bool> EvaluateMatcher(NameDefTree* pattern,
                                            const InterpValue& to_match,
                                            InterpBindings* bindings,
                                            InterpCallbackData* callbacks) {
  if (pattern->is_leaf()) {
    NameDefTree::Leaf leaf = pattern->leaf();
    if (absl::holds_alternative<WildcardPattern*>(leaf)) {
      return true;
    }
    if (absl::holds_alternative<NameDef*>(leaf)) {
      bindings->AddValue(absl::get<NameDef*>(leaf)->identifier(), to_match);
      return true;
    }
    if (absl::holds_alternative<Number*>(leaf) ||
        absl::holds_alternative<EnumRef*>(leaf)) {
      XLS_ASSIGN_OR_RETURN(InterpValue target,
                           callbacks->Eval(ToExprNode(leaf), bindings));
      return target.Eq(to_match);
    }
    XLS_RET_CHECK(absl::holds_alternative<NameRef*>(leaf));
    XLS_ASSIGN_OR_RETURN(InterpValue target,
                         callbacks->Eval(absl::get<NameRef*>(leaf), bindings));
    return target.Eq(to_match);
  }

  XLS_RET_CHECK_EQ(to_match.GetLength().value(), pattern->nodes().size());
  for (int64 i = 0; i < pattern->nodes().size(); ++i) {
    NameDefTree* subtree = pattern->nodes()[i];
    const InterpValue& member = to_match.GetValuesOrDie().at(i);
    XLS_ASSIGN_OR_RETURN(bool matched,
                         EvaluateMatcher(subtree, member, bindings, callbacks));
    if (!matched) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<InterpValue> EvaluateMatch(Match* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          InterpCallbackData* callbacks) {
  XLS_ASSIGN_OR_RETURN(InterpValue to_match,
                       callbacks->Eval(expr->matched(), bindings));
  for (MatchArm* arm : expr->arms()) {
    for (NameDefTree* pattern : arm->patterns()) {
      auto arm_bindings = std::make_shared<InterpBindings>(
          /*parent=*/bindings->shared_from_this());
      XLS_ASSIGN_OR_RETURN(
          bool did_match,
          EvaluateMatcher(pattern, to_match, arm_bindings.get(), callbacks));
      if (did_match) {
        return callbacks->Eval(arm->expr(), arm_bindings.get());
      }
    }
  }
  return absl::InternalError(
      absl::StrFormat("FailureError: %s The program being interpreted failed "
                      "with an incomplete match; value: %s",
                      expr->span().ToString(), to_match.ToString()));
}

absl::StatusOr<InterpBindings> MakeTopLevelBindings(
    const std::shared_ptr<Module>& module, InterpCallbackData* callbacks) {
  XLS_VLOG(3) << "Making top level bindings for module: " << module->name();
  InterpBindings b(/*parent=*/nullptr);

  // Add all the builtin functions.
  for (Builtin builtin : kAllBuiltins) {
    b.AddFn(BuiltinToString(builtin), InterpValue::MakeFunction(builtin));
  }

  // Add all the functions in the top level scope for the module.
  for (Function* f : module->GetFunctions()) {
    b.AddFn(f->identifier(),
            InterpValue::MakeFunction(InterpValue::UserFnData{module, f}));
  }

  // Add all the type definitions in the top level scope for the module to the
  // bindings.
  for (TypeDefinition td : module->GetTypeDefinitions()) {
    if (absl::holds_alternative<TypeDef*>(td)) {
      auto* type_def = absl::get<TypeDef*>(td);
      b.AddTypeDef(type_def->identifier(), type_def);
    } else if (absl::holds_alternative<StructDef*>(td)) {
      auto* struct_def = absl::get<StructDef*>(td);
      b.AddStructDef(struct_def->identifier(), struct_def);
    } else {
      auto* enum_def = absl::get<EnumDef*>(td);
      b.AddEnumDef(enum_def->identifier(), enum_def);
    }
  }

  // Add constants/imports present at the top level to the bindings.
  for (ModuleMember member : module->top()) {
    XLS_VLOG(3) << "Evaluating module member: "
                << ToAstNode(member)->ToString();
    if (absl::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = absl::get<ConstantDef*>(member);
      if (callbacks->is_wip(constant_def)) {
        XLS_VLOG(3) << "Saw WIP constant definition; breaking early! "
                    << constant_def->ToString();
        break;
      }
      XLS_VLOG(3) << "MakeTopLevelBindings evaluating: "
                  << constant_def->ToString();
      absl::optional<InterpValue> precomputed =
          callbacks->note_wip(constant_def, absl::nullopt);
      absl::optional<InterpValue> result;
      if (precomputed.has_value()) {  // If we already computed it, use that.
        result = precomputed.value();
      } else {  // Otherwise, evaluate it and make a note.
        XLS_ASSIGN_OR_RETURN(result,
                             callbacks->Eval(constant_def->value(), &b));
        callbacks->note_wip(constant_def, *result);
      }
      XLS_CHECK(result.has_value());
      b.AddValue(constant_def->identifier(), *result);
      XLS_VLOG(3) << "MakeTopLevelBindings evaluated: "
                  << constant_def->ToString() << " to " << result->ToString();
      continue;
    }
    if (absl::holds_alternative<Import*>(member)) {
      auto* import = absl::get<Import*>(member);
      XLS_VLOG(3) << "MakeTopLevelBindings importing: " << import->ToString();
      XLS_ASSIGN_OR_RETURN(
          const ModuleInfo* imported,
          DoImport(callbacks->typecheck, ImportTokens(import->subject()),
                   callbacks->cache));
      XLS_VLOG(3) << "MakeTopLevelBindings adding import " << import->ToString()
                  << " as \"" << import->identifier() << "\"";
      b.AddModule(import->identifier(), imported->module.get());
      continue;
    }
  }

  // Add a helpful value to the binding keys just to indicate what module these
  // top level bindings were created for, helpful for debugging.
  b.AddValue(absl::StrCat("__top_level_bindings_", module->name()),
             InterpValue::MakeNil());

  return b;
}

absl::StatusOr<int64> ResolveDim(
    absl::variant<Expr*, int64, ConcreteTypeDim> dim,
    InterpBindings* bindings) {
  if (absl::holds_alternative<int64>(dim)) {
    return absl::get<int64>(dim);
  }
  if (absl::holds_alternative<Expr*>(dim)) {
    Expr* expr = absl::get<Expr*>(dim);
    if (Number* number = dynamic_cast<Number*>(expr)) {
      return number->GetAsInt64();
    }
    if (NameRef* name_ref = dynamic_cast<NameRef*>(expr)) {
      const std::string& identifier = name_ref->identifier();
      XLS_ASSIGN_OR_RETURN(InterpValue value,
                           bindings->ResolveValueFromIdentifier(identifier));
      return value.GetBitValueInt64();
    }
    return absl::UnimplementedError(
        "Resolve dim expression: " + expr->ToString() + " @ " +
        expr->span().ToString());
  }

  XLS_RET_CHECK(absl::holds_alternative<ConcreteTypeDim>(dim));
  ConcreteTypeDim ctdim = absl::get<ConcreteTypeDim>(dim);
  if (const int64* value = absl::get_if<int64>(&ctdim.value())) {
    return *value;
  }

  const auto& parametric_expr =
      absl::get<ConcreteTypeDim::OwnedParametric>(ctdim.value());
  if (auto* parametric_symbol =
          dynamic_cast<ParametricSymbol*>(parametric_expr.get())) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue value,
        bindings->ResolveValueFromIdentifier(parametric_symbol->identifier()));
    return value.GetBitValueInt64();
  }

  return absl::UnimplementedError("Resolve dim");
}

absl::StatusOr<DerefVariant> EvaluateToStructOrEnumOrAnnotation(
    TypeDefinition type_definition, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  while (absl::holds_alternative<TypeDef*>(type_definition)) {
    TypeDef* type_def = absl::get<TypeDef*>(type_definition);
    TypeAnnotation* annotation = type_def->type();
    if (auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(annotation)) {
      type_definition = type_ref->type_ref()->type_definition();
    } else {
      return annotation;
    }
  }

  if (absl::holds_alternative<StructDef*>(type_definition)) {
    return absl::get<StructDef*>(type_definition);
  }
  if (absl::holds_alternative<EnumDef*>(type_definition)) {
    return absl::get<EnumDef*>(type_definition);
  }

  ModRef* modref = absl::get<ModRef*>(type_definition);
  XLS_ASSIGN_OR_RETURN(Module * imported_module,
                       bindings->ResolveModule(modref->import()->identifier()));
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       imported_module->GetTypeDefinition(modref->attr()));
  XLS_ASSIGN_OR_RETURN(
      InterpBindings imported_bindings,
      MakeTopLevelBindings(imported_module->shared_from_this(), callbacks));
  return EvaluateToStructOrEnumOrAnnotation(td, &imported_bindings, callbacks);
}

static absl::StatusOr<DerefVariant> DerefTypeRef(
    TypeRef* type_ref, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  if (absl::holds_alternative<ModRef*>(type_ref->type_definition())) {
    auto* mod_ref = absl::get<ModRef*>(type_ref->type_definition());
    return EvaluateToStructOrEnumOrAnnotation(mod_ref, bindings, callbacks);
  }

  XLS_ASSIGN_OR_RETURN(auto result,
                       bindings->ResolveTypeDefinition(type_ref->text()));
  return result;
}

// Returns new (derived) Bindings populated with `parametrics`.
//
// For example, if we have a struct defined as `struct [N: u32, M: u32] Foo`,
// and provided parametrics with values [A, 16], we'll create a new set of
// Bindings out of `bindings` and add (N, A) and (M, 16) to that.
//
// Args:
//   struct: The struct that may have parametric bindings.
//   parametrics: The parametric bindings that correspond to those on the
//     struct.
//   bindings: Bindings to use as the parent.
static absl::StatusOr<InterpBindings> BindingsWithStructParametrics(
    StructDef* struct_def, const std::vector<Expr*>& parametrics,
    InterpBindings* bindings) {
  InterpBindings nested_bindings(bindings->shared_from_this());
  XLS_CHECK_EQ(struct_def->parametric_bindings().size(), parametrics.size());
  for (int64 i = 0; i < parametrics.size(); ++i) {
    ParametricBinding* p = struct_def->parametric_bindings()[i];
    Expr* d = parametrics[i];
    if (Number* n = dynamic_cast<Number*>(d)) {
      int64 value = n->GetAsInt64().value();
      TypeAnnotation* type = n->type();
      if (type == nullptr) {
        // If the number didn't have a type annotation, use the one from the
        // parametric we're binding to.
        type = p->type();
      }
      XLS_RET_CHECK(type != nullptr)
          << "`" << n->ToString() << "` @ " << n->span();
      auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type);
      XLS_CHECK(builtin_type != nullptr);
      int64 bit_count = builtin_type->GetBitCount();
      nested_bindings.AddValue(p->name_def()->identifier(),
                               InterpValue::MakeUBits(bit_count, value));
    } else {
      auto* name_ref = dynamic_cast<NameRef*>(d);
      XLS_CHECK(name_ref != nullptr)
          << d->GetNodeTypeName() << " " << d->ToString();
      InterpValue value =
          nested_bindings.ResolveValueFromIdentifier(name_ref->identifier())
              .value();
      nested_bindings.AddValue(p->name_def()->identifier(), value);
    }
  }
  return nested_bindings;
}

// Turns the various possible subtypes for a TypeAnnotation AST node into a
// concrete type.
absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeTypeAnnotation(
    TypeAnnotation* type, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  XLS_VLOG(3) << "Concretizing type annotation: " << type->ToString();

  // class TypeRefTypeAnnotation
  if (auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(DerefVariant deref, DerefTypeRef(type_ref->type_ref(),
                                                          bindings, callbacks));
    absl::optional<InterpBindings> struct_parametric_bindings;
    if (type_ref->HasParametrics()) {
      XLS_RET_CHECK(absl::holds_alternative<StructDef*>(deref));
      auto* struct_def = absl::get<StructDef*>(deref);
      XLS_ASSIGN_OR_RETURN(struct_parametric_bindings,
                           BindingsWithStructParametrics(
                               struct_def, type_ref->parametrics(), bindings));
      bindings = &struct_parametric_bindings.value();
    }
    TypeDefinition type_defn = type_ref->type_ref()->type_definition();
    if (absl::holds_alternative<EnumDef*>(type_defn)) {
      auto* enum_def = absl::get<EnumDef*>(type_defn);
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ConcreteType> underlying_type,
          ConcretizeType(enum_def->type(), bindings, callbacks));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count,
                           underlying_type->GetTotalBitCount());
      return absl::make_unique<EnumType>(enum_def, bit_count);
    }
    return ConcretizeType(deref, bindings, callbacks);
  }

  // class TupleTypeAnnotation
  if (auto* tuple = dynamic_cast<TupleTypeAnnotation*>(type)) {
    std::vector<std::unique_ptr<ConcreteType>> members;
    for (TypeAnnotation* member : tuple->members()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_member,
                           ConcretizeType(member, bindings, callbacks));
      members.push_back(std::move(concrete_member));
    }
    return absl::make_unique<TupleType>(std::move(members));
  }

  // class ArrayTypeAnnotation
  if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64 dim, ResolveDim(array->dim(), bindings));
    TypeAnnotation* elem_type = array->element_type();
    XLS_VLOG(3) << "Resolved array dim to: " << dim
                << " elem_type: " << elem_type->ToString();
    if (auto* builtin_elem = dynamic_cast<BuiltinTypeAnnotation*>(elem_type);
        builtin_elem != nullptr && builtin_elem->GetBitCount() == 0) {
      return std::make_unique<BitsType>(builtin_elem->GetSignedness(), dim);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_elem_type,
                         ConcretizeType(elem_type, bindings, callbacks));
    return std::make_unique<ArrayType>(std::move(concrete_elem_type),
                                       ConcreteTypeDim(dim));
  }

  // class BuiltinTypeAnnotation
  if (auto* builtin = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    bool signedness = builtin->GetSignedness();
    int64 bit_count = builtin->GetBitCount();
    return absl::make_unique<BitsType>(signedness, bit_count);
  }

  return absl::UnimplementedError("Cannot concretize type annotation: " +
                                  type->ToString());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeType(
    ConcretizeVariant type, InterpBindings* bindings,
    InterpCallbackData* callbacks) {
  // class EnumDef
  if (EnumDef** penum_def = absl::get_if<EnumDef*>(&type)) {
    return ConcretizeType((*penum_def)->type(), bindings, callbacks);
  }
  // class StructDef
  if (StructDef** pstruct_def = absl::get_if<StructDef*>(&type)) {
    std::vector<std::unique_ptr<ConcreteType>> members;
    for (auto& [name_def, type_annotation] : (*pstruct_def)->members()) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ConcreteType> concretized,
          ConcretizeTypeAnnotation(type_annotation, bindings, callbacks));
      members.push_back(std::move(concretized));
    }
    return absl::make_unique<TupleType>(std::move(members));
  }
  // class TypeAnnotation
  return ConcretizeTypeAnnotation(absl::get<TypeAnnotation*>(type), bindings,
                                  callbacks);
}

}  // namespace xls::dslx
