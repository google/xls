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

#include "xls/dslx/evaluate.h"

#include "absl/strings/match.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

using Value = InterpValue;
using Tag = InterpValueTag;

}  // namespace

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            AbstractInterpreter* interp) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateCarry(Carry* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp) {
  return bindings->ResolveValueFromIdentifier("carry");
}

absl::StatusOr<InterpValue> EvaluateWhile(While* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue carry, interp->Eval(expr->init(), bindings));
  InterpBindings new_bindings(bindings);
  new_bindings.AddValue("carry", carry);
  while (true) {
    XLS_ASSIGN_OR_RETURN(InterpValue test,
                         interp->Eval(expr->test(), &new_bindings));
    if (!test.IsTrue()) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(carry, interp->Eval(expr->body(), &new_bindings));
    new_bindings.AddValue("carry", carry);
  }
  return carry;
}

absl::StatusOr<InterpValue> EvaluateNumber(Number* expr,
                                           InterpBindings* bindings,
                                           ConcreteType* type_context,
                                           AbstractInterpreter* interp) {
  XLS_VLOG(4) << "Evaluating number: " << expr->ToString() << " @ "
              << expr->span();
  std::unique_ptr<ConcreteType> type_context_value;
  if (type_context == nullptr && expr->kind() == NumberKind::kCharacter) {
    type_context_value = BitsType::MakeU8();
    type_context = type_context_value.get();
  }
  // Note that if there an explicit type annotation on a boolean value; e.g.
  // `s1:true` we skip providing the type context as u1 and pick up the
  // evaluation of the s1 below.
  if (type_context == nullptr && expr->kind() == NumberKind::kBool &&
      expr->type_annotation() == nullptr) {
    type_context_value = BitsType::MakeU1();
    type_context = type_context_value.get();
  }
  if (type_context == nullptr && expr->type_annotation() == nullptr) {
    return absl::InternalError(
        absl::StrFormat("FailureError: %s No type context for expression, "
                        "should be caught by type inference.",
                        expr->span().ToString()));
  }
  if (type_context == nullptr) {
    XLS_ASSIGN_OR_RETURN(
        type_context_value,
        ConcretizeTypeAnnotation(expr->type_annotation(), bindings, interp));
    type_context = type_context_value.get();
  }

  BitsType* bits_type = dynamic_cast<BitsType*>(type_context);
  XLS_RET_CHECK(bits_type != nullptr)
      << "Type for number should be 'bits' kind.";
  InterpValueTag tag =
      bits_type->is_signed() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  XLS_ASSIGN_OR_RETURN(
      int64_t bit_count,
      absl::get<InterpValue>(bits_type->size().value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, expr->GetBits(bit_count));
  return InterpValue::MakeBits(tag, std::move(bits));
}

absl::StatusOr<InterpValue> EvaluateString(String* expr,
                                           InterpBindings* bindings,
                                           ConcreteType* type_context,
                                           AbstractInterpreter* interp) {
  std::vector<InterpValue> elements;
  for (const unsigned char letter : expr->text()) {
    elements.push_back(InterpValue::MakeUBits(8, letter));
  }
  return InterpValue::MakeArray(elements);
}

static absl::StatusOr<EnumDef*> EvaluateToEnum(TypeDefinition type_definition,
                                               InterpBindings* bindings,
                                               AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(DerefVariant deref,
                       EvaluateToStructOrEnumOrAnnotation(
                           type_definition, bindings, interp, nullptr));
  if (absl::holds_alternative<EnumDef*>(deref)) {
    return absl::get<EnumDef*>(deref);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Type definition did not dereference to an enum, found: ",
                   ToAstNode(deref)->GetNodeTypeName()));
}

static absl::StatusOr<StructDef*> EvaluateToStruct(
    StructRef struct_ref, InterpBindings* bindings,
    AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       ToTypeDefinition(ToAstNode(struct_ref)));
  XLS_ASSIGN_OR_RETURN(DerefVariant deref,
                       EvaluateToStructOrEnumOrAnnotation(
                           type_definition, bindings, interp, nullptr));
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
                                             AbstractInterpreter* interp) {
  XLS_VLOG(5) << "Evaluating tuple @ " << expr->span()
              << " :: " << expr->ToString();
  auto get_type_context =
      [type_context](int64_t i) -> std::unique_ptr<ConcreteType> {
    if (type_context == nullptr) {
      return nullptr;
    }
    auto tuple_type_context = dynamic_cast<TupleType*>(type_context);
    return tuple_type_context->GetMemberType(i).CloneToUnique();
  };

  std::vector<InterpValue> members;
  for (int64_t i = 0; i < expr->members().size(); ++i) {
    Expr* m = expr->members()[i];
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         interp->Eval(m, bindings, get_type_context(i)));
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
                                                        int64_t bit_count) {
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
static absl::StatusOr<std::unique_ptr<ConcreteType>> ConcreteTypeFromValue(
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
        element_type = ConcreteType::MakeUnit();
      } else {
        XLS_ASSIGN_OR_RETURN(
            element_type,
            ConcreteTypeFromValue(value.Index(Value::MakeU32(0)).value()));
      }
      auto dim = ConcreteTypeDim::CreateU32(value.GetLength().value());
      return std::make_unique<ArrayType>(std::move(element_type), dim);
    }
    case InterpValueTag::kTuple: {
      std::vector<std::unique_ptr<ConcreteType>> members;
      for (const InterpValue& m : value.GetValuesOrDie()) {
        XLS_ASSIGN_OR_RETURN(auto dim, ConcreteTypeFromValue(m));
        members.push_back(std::move(dim));
      }
      return absl::make_unique<TupleType>(std::move(members));
    }
    case InterpValueTag::kEnum:
      return StrengthReduceEnum(value.type(), value.GetBitCount().value());
    case InterpValueTag::kFunction:
      break;
    case InterpValueTag::kToken:
      break;
  }
  XLS_LOG(FATAL) << "Invalid value tag for ConcreteTypeFromValue: "
                 << static_cast<int64_t>(value.tag());
}

// Helper that returns whether the bit width of `value` matches `target`.
//
// Precondition: value is bits typed, target contains a bits-typed InterpValue.
static bool BitCountMatches(const InterpValue& value,
                            const ConcreteTypeDim& target) {
  int64_t bit_count = value.GetBitCount().value();
  return target.GetAsInt64().value() == bit_count;
}

absl::StatusOr<bool> ValueCompatibleWithType(const ConcreteType& type,
                                             const InterpValue& value) {
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&type)) {
    XLS_RET_CHECK_EQ(value.tag(), InterpValueTag::kTuple);
    int64_t member_count = tuple_type->size();
    const auto& elements = value.GetValuesOrDie();
    if (member_count != elements.size()) {
      return false;
    }
    for (int64_t i = 0; i < member_count; ++i) {
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
    XLS_ASSIGN_OR_RETURN(
        int64_t array_size,
        absl::get<InterpValue>(array_type->size().value()).GetBitValueInt64());
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
      return BitCountMatches(value, bits_type->size());
    }

    if (bits_type->is_signed() && value.tag() == InterpValueTag::kSBits) {
      return BitCountMatches(value, bits_type->size());
    }

    // Enum values can be converted to bits type if the signedness/bit counts
    // line up.
    if (value.tag() == InterpValueTag::kEnum) {
      return value.type()->signedness().value() == bits_type->is_signed() &&
             BitCountMatches(value, bits_type->size());
    }

    // Arrays can be converted to unsigned bits types by flattening, but we must
    // check the flattened bit count is the same as the target bit type.
    if (!bits_type->is_signed() && value.tag() == InterpValueTag::kArray) {
      int64_t flat_bit_count = value.Flatten().value().GetBitCount().value();
      return flat_bit_count == absl::get<InterpValue>(bits_type->size().value())
                                   .GetBitValueInt64()
                                   .value();
    }
  }

  if (auto* enum_type = dynamic_cast<const EnumType*>(&type);
      enum_type != nullptr && value.IsBits()) {
    return enum_type->signedness().value() ==
               (value.tag() == InterpValueTag::kSBits) &&
           BitCountMatches(value, enum_type->GetTotalBitCount().value());
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
             BitCountMatches(value, bits_type->size());
    }
    case InterpValueTag::kSBits: {
      auto* bits_type = dynamic_cast<const BitsType*>(&type);
      if (bits_type == nullptr) {
        return false;
      }
      return bits_type->is_signed() &&
             BitCountMatches(value, bits_type->size());
    }
    case InterpValueTag::kArray:
    case InterpValueTag::kTuple:
    case InterpValueTag::kEnum:
      // TODO(leary): 2020-11-16 We should be able to use stricter checks here
      // than "can I cast value to type".
      return ValueCompatibleWithType(type, value);
    case InterpValueTag::kFunction:
      break;
    case InterpValueTag::kToken:
      break;
  }
  return absl::UnimplementedError(
      absl::StrCat("ConcreteTypeAcceptsValue not implemented for tag: ",
                   static_cast<int64_t>(value.tag())));
}

// Evaluates the parametric values derived from other parametric values.
//
// Populates the "bindings" mapping with results computed by the typechecker.
//
// For example, in:
//
//  fn [X: u32, Y: u32 = X+X] f(x: bits[X]) { ... }
//
// Args:
//  fn: Function to evaluate parametric bindings for.
//  bindings: Bindings mapping to populate with newly evaluated parametric
//    binding names.
//  bound_dims: Parametric bindings computed by the typechecker.
static absl::Status EvaluateDerivedParametrics(
    Function* fn, InterpBindings* bindings, AbstractInterpreter* interp,
    const absl::flat_hash_map<std::string, InterpValue>& bound_dims) {
  // Formatter for elements in "bound_dims".
  auto dims_formatter = [](std::string* out,
                           const std::pair<std::string, InterpValue>& p) {
    out->append(absl::StrCat(p.first, ":", p.second.ToString()));
  };
  XLS_VLOG(5) << "EvaluateDerivedParametrics; fn: " << fn->identifier()
              << " bound_dims: ["
              << absl::StrJoin(bound_dims, ", ", dims_formatter) << "]";
  XLS_RET_CHECK_EQ(bound_dims.size(), fn->parametric_bindings().size())
      << "bound dims: [" << absl::StrJoin(bound_dims, ", ", dims_formatter)
      << "] fn: " << fn->ToString();
  for (ParametricBinding* parametric : fn->parametric_bindings()) {
    std::string id = parametric->identifier();
    if (bindings->Contains(id)) {
      continue;  // Already bound.
    }

    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type,
                         ConcretizeTypeAnnotation(parametric->type_annotation(),
                                                  bindings, interp));
    // We already computed derived parametrics in the parametric instantiator.
    // All that's left is to add it to the current bindings.
    bindings->AddValue(id, bound_dims.at(id));
  }
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> EvaluateFunction(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    const SymbolicBindings& symbolic_bindings, AbstractInterpreter* interp) {
  XLS_RET_CHECK_EQ(f->owner(), interp->GetCurrentTypeInfo()->module());
  XLS_VLOG(5) << "Evaluating function: " << f->identifier()
              << " symbolic_bindings: " << symbolic_bindings;
  if (args.size() != f->params().size()) {
    return absl::InternalError(
        absl::StrFormat("EvaluateError: %s Argument arity mismatch for "
                        "invocation; want %d got %d",
                        span.ToString(), f->params().size(), args.size()));
  }

  Module* m = f->owner();
  XLS_ASSIGN_OR_RETURN(const InterpBindings* top_level_bindings,
                       GetOrCreateTopLevelBindings(m, interp));
  XLS_VLOG(5) << "Evaluated top level bindings for module: " << m->name()
              << "; keys: {"
              << absl::StrJoin(top_level_bindings->GetKeys(), ", ") << "}";
  InterpBindings fn_bindings(/*parent=*/top_level_bindings);
  XLS_RETURN_IF_ERROR(EvaluateDerivedParametrics(f, &fn_bindings, interp,
                                                 symbolic_bindings.ToMap()));

  fn_bindings.set_fn_ctx(FnCtx{m->name(), f->identifier(), symbolic_bindings});
  for (int64_t i = 0; i < f->params().size(); ++i) {
    fn_bindings.AddValue(f->params()[i]->identifier(), args[i]);
  }

  return interp->Eval(f->body(), &fn_bindings);
}

absl::StatusOr<InterpValue> ConcreteTypeConvertValue(
    const ConcreteType& type, const InterpValue& value, const Span& span,
    absl::optional<std::vector<InterpValue>> enum_values,
    absl::optional<bool> enum_signed) {
  // Converting unsigned bits to an array.
  if (auto* array_type = dynamic_cast<const ArrayType*>(&type);
      array_type != nullptr && value.IsUBits()) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim element_bit_count,
                         array_type->element_type().GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t bits_per_element,
                         element_bit_count.GetAsInt64());
    const Bits& bits = value.GetBitsOrDie();

    auto bit_slice_value_at_index = [&](int64_t i) -> InterpValue {
      int64_t lo = i * bits_per_element;
      Bits rev = bits_ops::Reverse(bits);
      Bits slice = rev.Slice(lo, bits_per_element);
      Bits result = bits_ops::Reverse(slice);
      return InterpValue::MakeBits(InterpValueTag::kUBits, result).value();
    };

    std::vector<InterpValue> values;
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    for (int64_t i = 0; i < array_size; ++i) {
      values.push_back(bit_slice_value_at_index(i));
    }

    return Value::MakeArray(values);
  }

  // Converting bits-having-thing into an enum.
  if (auto* enum_type = dynamic_cast<const EnumType*>(&type);
      enum_type != nullptr && value.HasBits() &&
      BitCountMatches(value, type.GetTotalBitCount().value())) {
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

  auto zero_ext = [&]() -> absl::StatusOr<InterpValue> {
    auto* bits_type = dynamic_cast<const BitsType*>(&type);
    XLS_CHECK(bits_type != nullptr);
    InterpValueTag tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                                : InterpValueTag::kUBits;
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type->size().GetAsInt64());
    XLS_ASSIGN_OR_RETURN(Bits zext_bits, value.ZeroExt(bit_count)->GetBits());
    return InterpValue::MakeBits(tag, zext_bits);
  };

  auto sign_ext = [&]() -> absl::StatusOr<InterpValue> {
    auto* bits_type = dynamic_cast<const BitsType*>(&type);
    XLS_CHECK(bits_type != nullptr);
    InterpValueTag tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                                : InterpValueTag::kUBits;
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type->size().GetAsInt64());
    XLS_ASSIGN_OR_RETURN(InterpValue sign_ext_value, value.SignExt(bit_count));
    XLS_ASSIGN_OR_RETURN(Bits sign_ext_bits, sign_ext_value.GetBits());
    return InterpValue::MakeBits(tag, std::move(sign_ext_bits));
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
    AbstractInterpreter* interp) {
  auto* enum_type = dynamic_cast<const EnumType*>(&type);
  if (enum_type == nullptr) {
    return absl::nullopt;
  }

  EnumDef* enum_def = enum_type->nominal_type();
  std::vector<InterpValue> result;
  for (const EnumMember& member : enum_def->values()) {
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         interp->Eval(member.value, bindings));
    result.push_back(std::move(value));
  }
  return absl::make_optional(std::move(result));
}

absl::StatusOr<InterpValue> EvaluateArray(Array* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp) {
  XLS_VLOG(5) << "Evaluating array @ " << expr->span()
              << " :: " << expr->ToString();
  std::unique_ptr<ConcreteType> type;
  if (type_context == nullptr && expr->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(type, ConcretizeTypeAnnotation(expr->type_annotation(),
                                                        bindings, interp));
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
                         interp->Eval(m, bindings, std::move(type_context)));
    elements.push_back(std::move(e));
  }
  if (expr->has_ellipsis()) {
    XLS_RET_CHECK(array_type != nullptr);
    XLS_ASSIGN_OR_RETURN(
        int64_t target_size,
        absl::get<InterpValue>(array_type->size().value()).GetBitValueInt64());
    XLS_RET_CHECK_GE(target_size, 0);
    XLS_VLOG(5) << "Array has ellipsis @ " << expr->span()
                << ": repeating to target size: " << target_size;
    while (elements.size() < target_size) {
      elements.push_back(elements.back());
    }
  }
  return InterpValue::MakeArray(std::move(elements));
}

absl::StatusOr<InterpValue> EvaluateCast(Cast* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         AbstractInterpreter* interp) {
  XLS_RET_CHECK_EQ(expr->owner(), interp->GetCurrentTypeInfo()->module());
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> type,
      ConcretizeTypeAnnotation(expr->type_annotation(), bindings, interp));
  XLS_ASSIGN_OR_RETURN(InterpValue value, interp->Eval(expr->expr(), bindings,
                                                       type->CloneToUnique()));
  XLS_ASSIGN_OR_RETURN(absl::optional<std::vector<InterpValue>> enum_values,
                       GetEnumValues(*type, bindings, interp));
  return ConcreteTypeConvertValue(
      *type, value, expr->span(), std::move(enum_values),
      value.type() == nullptr
          ? absl::nullopt
          : absl::make_optional(value.type()->signedness().value()));
}

absl::StatusOr<InterpValue> EvaluateLet(Let* expr, InterpBindings* bindings,
                                        ConcreteType* type_context,
                                        AbstractInterpreter* interp) {
  std::unique_ptr<ConcreteType> want_type;
  if (expr->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        want_type,
        ConcretizeTypeAnnotation(expr->type_annotation(), bindings, interp));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue to_bind,
                       interp->Eval(expr->rhs(), bindings));
  if (want_type != nullptr) {
    XLS_ASSIGN_OR_RETURN(bool accepted,
                         ConcreteTypeAcceptsValue(*want_type, to_bind));
    if (!accepted) {
      XLS_ASSIGN_OR_RETURN(auto concrete_type, ConcreteTypeFromValue(to_bind));
      return absl::InternalError(absl::StrFormat(
          "EvaluateError: %s Type error found! Let-expression right hand side "
          "did not "
          "conform to annotated type\n\twant: %s\n\tgot:  %s\n\tvalue: %s",
          expr->span().ToString(), want_type->ToString(),
          concrete_type->ToString(), to_bind.ToString()));
    }
  }

  InterpBindings new_bindings =
      InterpBindings::CloneWith(bindings, expr->name_def_tree(), to_bind);
  return interp->Eval(expr->body(), &new_bindings);
}

absl::StatusOr<InterpValue> EvaluateFor(For* expr, InterpBindings* bindings,
                                        ConcreteType* type_context,
                                        AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue iterable,
                       interp->Eval(expr->iterable(), bindings));

  std::unique_ptr<ConcreteType> concrete_iteration_type;
  if (expr->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        concrete_iteration_type,
        ConcretizeTypeAnnotation(expr->type_annotation(), bindings, interp));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue carry, interp->Eval(expr->init(), bindings));
  XLS_ASSIGN_OR_RETURN(int64_t length, iterable.GetLength());
  for (int64_t i = 0; i < length; ++i) {
    const InterpValue& x = iterable.GetValuesOrDie().at(i);
    InterpValue iteration = InterpValue::MakeTuple({x, carry});

    // If there's a type annotation, validate that the value we evaluated
    // conforms to it as a spot check.
    if (concrete_iteration_type != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          bool type_checks,
          ConcreteTypeAcceptsValue(*concrete_iteration_type, iteration));
      if (!type_checks) {
        XLS_ASSIGN_OR_RETURN(auto concrete_type,
                             ConcreteTypeFromValue(iteration));
        return absl::InternalError(absl::StrFormat(
            "EvaluateError: %s Type error found! Iteration value does not "
            "conform to type annotation at top of iteration %d:\n  got value: "
            "%s\n  type: %s\n  want: %s",
            expr->span().ToString(), i, iteration.ToString(),
            concrete_type->ToString(), concrete_iteration_type->ToString()));
      }
    }

    InterpBindings new_bindings =
        InterpBindings::CloneWith(bindings, expr->names(), iteration);
    XLS_ASSIGN_OR_RETURN(carry, interp->Eval(expr->body(), &new_bindings));
  }
  return carry;
}

absl::StatusOr<InterpValue> EvaluateStructInstance(
    StructInstance* expr, InterpBindings* bindings, ConcreteType* type_context,
    AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       EvaluateToStruct(expr->struct_def(), bindings, interp));
  std::vector<InterpValue> members;
  for (auto [name, field_expr] : expr->GetOrderedMembers(struct_def)) {
    XLS_ASSIGN_OR_RETURN(InterpValue member,
                         interp->Eval(field_expr, bindings));
    members.push_back(member);
  }
  return InterpValue::MakeTuple(std::move(members));
}

absl::StatusOr<InterpValue> EvaluateSplatStructInstance(
    SplatStructInstance* expr, InterpBindings* bindings,
    ConcreteType* type_context, AbstractInterpreter* interp) {
  // First we grab the "basis" struct value (the subject of the 'splat') that
  // we're going to update with the modified fields.
  XLS_ASSIGN_OR_RETURN(InterpValue named_tuple,
                       interp->Eval(expr->splatted(), bindings));
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       EvaluateToStruct(expr->struct_ref(), bindings, interp));
  for (auto [name, field_expr] : expr->members()) {
    XLS_ASSIGN_OR_RETURN(InterpValue new_value,
                         interp->Eval(field_expr, bindings));
    int64_t i = struct_def->GetMemberIndex(name).value();
    XLS_ASSIGN_OR_RETURN(
        named_tuple, named_tuple.Update(InterpValue::MakeU32(i), new_value));
  }

  return named_tuple;
}

static absl::StatusOr<InterpValue> EvaluateEnumRefHelper(
    Expr* expr, EnumDef* enum_def, absl::string_view attr,
    AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(auto value_node, enum_def->GetValue(attr));
  // Note: we have grab bindings from the underlying type (not from the expr,
  // which may have been in a different module from the enum_def).
  TypeAnnotation* underlying_type = enum_def->type_annotation();
  XLS_ASSIGN_OR_RETURN(
      const InterpBindings* top_bindings,
      GetOrCreateTopLevelBindings(underlying_type->owner(), interp));

  AbstractInterpreter::ScopedTypeInfoSwap stis_type(interp, underlying_type);
  InterpBindings fresh_bindings(/*parent=*/top_bindings);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> concrete_type,
      ConcretizeTypeAnnotation(underlying_type, &fresh_bindings, interp));

  AbstractInterpreter::ScopedTypeInfoSwap stis_value(interp, value_node);
  XLS_ASSIGN_OR_RETURN(InterpValue raw_value,
                       interp->Eval(value_node, &fresh_bindings));
  return InterpValue::MakeEnum(raw_value.GetBitsOrDie(), enum_def);
}

// This resolves the "LHS" entity for this colon ref -- following resolving the
// left hand side, we do an attribute access, either evaluating to an enum
// value, or to a function/constant in the case of a module.
//
// Note that the LHS of a colon ref may be another colon ref, so in:
//
//    some_module::SomeEnum::VALUE
//
// We'll have the grouping:
//
//    ColonRef(subject: ColonRef(subject: NameRef(some_module), attr: SomeEnum),
//             attr:VALUE)
//
// In that case the inner ColonRef will resolve the module, then the subsequent
// step will resolve the EnumDef inside of that module.
static absl::StatusOr<absl::variant<EnumDef*, Module*>> ResolveColonRefSubject(
    ColonRef* expr, InterpBindings* bindings, AbstractInterpreter* interp) {
  if (absl::holds_alternative<NameRef*>(expr->subject())) {
    auto* name_ref = absl::get<NameRef*>(expr->subject());
    absl::optional<InterpBindings::Entry> entry =
        bindings->ResolveEntry(name_ref->identifier());
    XLS_RET_CHECK(entry.has_value());
    // Subject resolves directly to an enum definition.
    if (absl::holds_alternative<EnumDef*>(*entry)) {
      return absl::get<EnumDef*>(*entry);
    }
    // Subject resolves to an (imported) module.
    if (absl::holds_alternative<Module*>(*entry)) {
      return absl::get<Module*>(*entry);
    }
    // Subject resolves to a typedef.
    if (absl::holds_alternative<TypeDef*>(*entry)) {
      auto* type_def = absl::get<TypeDef*>(*entry);
      XLS_ASSIGN_OR_RETURN(EnumDef * enum_def,
                           EvaluateToEnum(type_def, bindings, interp));
      return enum_def;
    }
    return absl::InternalError(absl::StrFormat(
        "EvaluateError: %s Unsupported colon-reference subject.",
        expr->span().ToString()));
  }

  XLS_RET_CHECK(absl::holds_alternative<ColonRef*>(expr->subject()));
  auto* subject = absl::get<ColonRef*>(expr->subject());
  XLS_ASSIGN_OR_RETURN(auto subject_resolved,
                       ResolveColonRefSubject(subject, bindings, interp));
  // Has to be a module as the subject, since it's a nested colon-reference.
  XLS_RET_CHECK(absl::holds_alternative<Module*>(subject_resolved));
  auto* subject_module = absl::get<Module*>(subject_resolved);

  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       subject_module->GetTypeDefinition(subject->attr()));
  XLS_ASSIGN_OR_RETURN(const InterpBindings* top_level_bindings,
                       GetOrCreateTopLevelBindings(subject_module, interp));
  InterpBindings fresh_bindings(/*parent=*/top_level_bindings);
  XLS_ASSIGN_OR_RETURN(
      EnumDef * enum_def,
      EvaluateToEnum(type_definition, &fresh_bindings, interp));
  return enum_def;
}

absl::StatusOr<InterpValue> EvaluateColonRef(ColonRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context,
                                             AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(auto subject,
                       ResolveColonRefSubject(expr, bindings, interp));
  XLS_VLOG(5) << "ColonRef resolved subject: " << ToAstNode(subject)->ToString()
              << " attr: " << expr->attr();
  if (absl::holds_alternative<EnumDef*>(subject)) {
    auto* enum_def = absl::get<EnumDef*>(subject);
    return EvaluateEnumRefHelper(expr, enum_def, expr->attr(), interp);
  }

  auto* module = absl::get<Module*>(subject);
  absl::optional<ModuleMember*> member =
      module->FindMemberWithName(expr->attr());
  XLS_RET_CHECK(member.has_value());
  if (absl::holds_alternative<Function*>(*member.value())) {
    auto* f = absl::get<Function*>(*member.value());
    return InterpValue::MakeFunction(InterpValue::UserFnData{f->owner(), f});
  }
  if (absl::holds_alternative<ConstantDef*>(*member.value())) {
    auto* cd = absl::get<ConstantDef*>(*member.value());
    XLS_VLOG(5) << "ColonRef resolved to ConstantDef: " << cd->ToString();
    XLS_ASSIGN_OR_RETURN(const InterpBindings* module_top,
                         GetOrCreateTopLevelBindings(module, interp));
    InterpBindings bindings(module_top);
    AbstractInterpreter::ScopedTypeInfoSwap stis(interp, module);
    return interp->Eval(cd->value(), &bindings);
  }
  return absl::InternalError(
      absl::StrFormat("EvaluateError: %s Unsupported module reference.",
                      expr->span().ToString()));
}

absl::StatusOr<InterpValue> EvaluateUnop(Unop* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue arg,
                       interp->Eval(expr->operand(), bindings));
  switch (expr->kind()) {
    case UnopKind::kInvert:
      return arg.BitwiseNegate();
    case UnopKind::kNegate:
      return arg.ArithmeticNegate();
  }
  return absl::InternalError(absl::StrCat("Invalid unary operation kind: ",
                                          static_cast<int64_t>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateShift(Binop* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp) {
  XLS_VLOG(6) << "EvaluateShift: " << expr->ToString() << " @ " << expr->span();
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  std::unique_ptr<ConcreteType> rhs_type = nullptr;
  // Retrieve a type context for the right hand side as an un-type-annotated
  // literal number is permitted.
  absl::optional<ConcreteType*> rhs_item =
      interp->GetCurrentTypeInfo()->GetItem(expr->rhs());
  if (rhs_item.has_value()) {
    rhs_type = rhs_item.value()->CloneToUnique();
  }
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, interp->Eval(expr->rhs(), bindings,
                                                     std::move(rhs_type)));

  switch (expr->kind()) {
    case BinopKind::kShl:
      return lhs.Shl(rhs);
    case BinopKind::kShr:
      if (lhs.IsSigned()) {
        return lhs.Shra(rhs);
      }
      return lhs.Shrl(rhs);
    default:
      // Not an exhaustive list: this function only handles the shift operators.
      break;
  }
  return absl::InternalError(absl::StrCat("Invalid shift operation kind: ",
                                          static_cast<int64_t>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateBinop(Binop* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp) {
  if (GetBinopShifts().contains(expr->kind())) {
    return EvaluateShift(expr, bindings, type_context, interp);
  }

  XLS_VLOG(6) << "EvaluateBinop: " << expr->ToString() << " @ " << expr->span();
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, interp->Eval(expr->rhs(), bindings));

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
    default:
      // Not an exhaustive list as the shift cases are handled in
      // EvaluateShift().
      break;
  }
  return absl::InternalError(absl::StrCat("Invalid binary operation kind: ",
                                          static_cast<int64_t>(expr->kind())));
}

absl::StatusOr<InterpValue> EvaluateTernary(Ternary* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context,
                                            AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue test, interp->Eval(expr->test(), bindings));
  if (test.IsTrue()) {
    return interp->Eval(expr->consequent(), bindings);
  }
  return interp->Eval(expr->alternate(), bindings);
}

absl::StatusOr<InterpValue> EvaluateAttr(Attr* expr, InterpBindings* bindings,
                                         ConcreteType* type_context,
                                         AbstractInterpreter* interp) {
  TypeInfo* type_info = interp->GetCurrentTypeInfo();
  XLS_RET_CHECK_EQ(expr->owner(), type_info->module());

  // Resolve the tuple type to figure out what index of the tuple we're
  // grabbing.
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  absl::optional<const ConcreteType*> maybe_type =
      type_info->GetItem(expr->lhs());
  XLS_RET_CHECK(maybe_type.has_value())
      << "LHS of attr: " << expr << " should have type info in: " << type_info
      << " @ " << expr->lhs()->span();
  auto* struct_type = dynamic_cast<const StructType*>(maybe_type.value());
  XLS_RET_CHECK(struct_type != nullptr) << (*maybe_type)->ToString();

  absl::optional<int64_t> index;
  for (int64_t i = 0; i < struct_type->size(); ++i) {
    absl::string_view name = struct_type->GetMemberName(i);
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
    Index* expr, InterpBindings* bindings, AbstractInterpreter* interp,
    const Bits& bits) {
  IndexRhs index = expr->rhs();
  XLS_RET_CHECK(absl::holds_alternative<Slice*>(index));
  auto index_slice = absl::get<Slice*>(index);

  const SymbolicBindings& sym_bindings = bindings->fn_ctx()->sym_bindings;

  TypeInfo* type_info = interp->GetCurrentTypeInfo();
  absl::optional<StartAndWidth> maybe_saw =
      type_info->GetSliceStartAndWidth(index_slice, sym_bindings);
  XLS_RET_CHECK(maybe_saw.has_value())
      << "Slice start/width missing for slice @ " << expr->span();
  const auto& saw = maybe_saw.value();
  return Value::MakeBits(Tag::kUBits, bits.Slice(saw.start, saw.width));
}

static absl::StatusOr<InterpValue> EvaluateIndexWidthSlice(
    Index* expr, InterpBindings* bindings, AbstractInterpreter* interp,
    Bits bits) {
  auto width_slice = absl::get<WidthSlice*>(expr->rhs());
  XLS_ASSIGN_OR_RETURN(
      InterpValue start,
      interp->Eval(
          width_slice->start(), bindings,
          std::make_unique<BitsType>(/*is_signed=*/false, bits.bit_count())));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ConcreteType> width_type,
      ConcretizeTypeAnnotation(width_slice->width(), bindings, interp));

  auto* width_bits_type = dynamic_cast<BitsType*>(width_type.get());
  XLS_RET_CHECK(width_bits_type != nullptr);
  XLS_ASSIGN_OR_RETURN(int64_t width_value,
                       absl::get<InterpValue>(width_bits_type->size().value())
                           .GetBitValueInt64());

  // Make a value which corresponds to the slice being completely out of bounds.
  auto make_oob_value = [&]() {
    return Value::MakeUBits(/*bit_count=*/width_value, /*value=*/0);
  };

  if (!start.FitsInUint64()) {
    return make_oob_value();
  }

  XLS_ASSIGN_OR_RETURN(uint64_t start_index, start.GetBitValueUint64());
  if (start_index >= bits.bit_count()) {
    // Return early  to potentially avoid an unreasonably long zero extend (e.g.
    // if the start index was a large negative number).
    return make_oob_value();
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
                                          AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, interp->Eval(expr->lhs(), bindings));
  if (lhs.IsBits()) {
    if (absl::holds_alternative<Slice*>(expr->rhs())) {
      return EvaluateIndexBitSlice(expr, bindings, interp, lhs.GetBitsOrDie());
    }
    XLS_RET_CHECK(absl::holds_alternative<WidthSlice*>(expr->rhs()));
    return EvaluateIndexWidthSlice(expr, bindings, interp, lhs.GetBitsOrDie());
  }

  Expr* index = absl::get<Expr*>(expr->rhs());
  // Note: since we permit a type-unannotated literal number we provide a type
  // context here.
  XLS_ASSIGN_OR_RETURN(InterpValue index_value,
                       interp->Eval(index, bindings, BitsType::MakeU32()));
  XLS_ASSIGN_OR_RETURN(uint64_t index_int, index_value.GetBitValueUint64());
  int64_t length = lhs.GetLength().value();
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
                                            AbstractInterpreter* interp) {
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
        absl::holds_alternative<ColonRef*>(leaf)) {
      XLS_ASSIGN_OR_RETURN(InterpValue target,
                           interp->Eval(ToExprNode(leaf), bindings));
      return target.Eq(to_match);
    }
    XLS_RET_CHECK(absl::holds_alternative<NameRef*>(leaf));
    XLS_ASSIGN_OR_RETURN(InterpValue target,
                         interp->Eval(absl::get<NameRef*>(leaf), bindings));
    return target.Eq(to_match);
  }

  XLS_RET_CHECK_EQ(to_match.GetLength().value(), pattern->nodes().size());
  for (int64_t i = 0; i < pattern->nodes().size(); ++i) {
    NameDefTree* subtree = pattern->nodes()[i];
    const InterpValue& member = to_match.GetValuesOrDie().at(i);
    XLS_ASSIGN_OR_RETURN(bool matched,
                         EvaluateMatcher(subtree, member, bindings, interp));
    if (!matched) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<InterpValue> EvaluateMatch(Match* expr, InterpBindings* bindings,
                                          ConcreteType* type_context,
                                          AbstractInterpreter* interp) {
  XLS_ASSIGN_OR_RETURN(InterpValue to_match,
                       interp->Eval(expr->matched(), bindings));
  for (MatchArm* arm : expr->arms()) {
    for (NameDefTree* pattern : arm->patterns()) {
      InterpBindings arm_bindings(
          /*parent=*/bindings);
      XLS_ASSIGN_OR_RETURN(
          bool did_match,
          EvaluateMatcher(pattern, to_match, &arm_bindings, interp));
      if (did_match) {
        return interp->Eval(arm->expr(), &arm_bindings);
      }
    }
  }
  return absl::InternalError(
      absl::StrFormat("FailureError: %s The program being interpreted failed "
                      "with an incomplete match; value: %s",
                      expr->span().ToString(), to_match.ToString()));
}

absl::StatusOr<const InterpBindings*> GetOrCreateTopLevelBindings(
    Module* module, AbstractInterpreter* interp) {
  ImportData* import_data = interp->GetImportData();
  InterpBindings& b = import_data->GetOrCreateTopLevelBindings(module);

  // If they're marked as done in the import cache, we can return them directly.
  // Otherwise, we'll populate them, and if we populate everything, mark it as
  // done.
  if (import_data->IsTopLevelBindingsDone(module)) {
    return &b;
  }

  AbstractInterpreter::ScopedTypeInfoSwap stis(interp, module);

  XLS_VLOG(4) << "Making top level bindings for module: " << module->name();

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

  bool saw_wip = false;

  // Add constants/imports present at the top level to the bindings.
  for (ModuleMember member : module->top()) {
    XLS_VLOG(3) << "Evaluating module member: " << ToAstNode(member)->ToString()
                << " (" << ToAstNode(member)->GetNodeTypeName() << ")";
    if (interp->IsWip(ToAstNode(member))) {
      XLS_VLOG(3) << "Saw WIP module member; breaking early!";
      saw_wip = true;
      break;
    }
    if (absl::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = absl::get<ConstantDef*>(member);
      XLS_VLOG(3) << "GetOrCreateTopLevelBindings evaluating: "
                  << constant_def->ToString();
      absl::optional<InterpValue> precomputed =
          interp->NoteWip(constant_def, absl::nullopt);
      absl::optional<InterpValue> result;
      if (precomputed.has_value()) {  // If we already computed it, use that.
        result = precomputed.value();
      } else {  // Otherwise, evaluate it and make a note.
        XLS_ASSIGN_OR_RETURN(result, interp->Eval(constant_def->value(), &b));
        interp->NoteWip(constant_def, *result);
      }
      XLS_CHECK(result.has_value());
      b.AddValue(constant_def->identifier(), *result);
      XLS_VLOG(3) << "GetOrCreateTopLevelBindings evaluated: "
                  << constant_def->ToString() << " to " << result->ToString();
      continue;
    }
    if (absl::holds_alternative<Import*>(member)) {
      auto* import = absl::get<Import*>(member);
      XLS_VLOG(3) << "GetOrCreateTopLevelBindings importing: "
                  << import->ToString();
      XLS_ASSIGN_OR_RETURN(
          const ModuleInfo* imported,
          DoImport(interp->GetTypecheckFn(), ImportTokens(import->subject()),
                   interp->GetAdditionalSearchPaths(), interp->GetImportData(),
                   import->span()));
      XLS_VLOG(3) << "GetOrCreateTopLevelBindings adding import "
                  << import->ToString() << " as \"" << import->identifier()
                  << "\"";
      b.AddModule(import->identifier(), imported->module.get());
      continue;
    }
  }

  // Add a "helpful" value to the binding keys -- just to indicate what module
  // these top level bindings were created for, can be helpful for debugging
  // when you have an arbitrary InterpBindings and want to understand its
  // provenance.
  b.AddValue(absl::StrCat("__top_level_bindings_", module->name()),
             InterpValue::MakeUnit());

  if (!saw_wip) {
    // Marking the top level bindings as done avoids needless re-evaluation in
    // the future.
    import_data->MarkTopLevelBindingsDone(module);
  }
  return &b;
}

// Resolves a dimension (e.g. as present in a type annotation) to an int64_t.
absl::StatusOr<int64_t> ResolveDim(
    absl::variant<Expr*, int64_t, ConcreteTypeDim> dim,
    InterpBindings* bindings, AbstractInterpreter* interp) {
  if (absl::holds_alternative<int64_t>(dim)) {
    int64_t result = absl::get<int64_t>(dim);
    XLS_RET_CHECK_GE(result, 0);
    return result;
  }
  if (absl::holds_alternative<Expr*>(dim)) {
    Expr* expr = absl::get<Expr*>(dim);
    XLS_VLOG(5) << "Resolving dim @ " << expr->span()
                << " :: " << expr->ToString();
    XLS_ASSIGN_OR_RETURN(InterpValue result,
                         interp->Eval(expr, bindings, BitsType::MakeU32()));
    return result.GetBitValueUint64();
  }

  XLS_RET_CHECK(absl::holds_alternative<ConcreteTypeDim>(dim));
  ConcreteTypeDim ctdim = absl::get<ConcreteTypeDim>(dim);
  if (absl::holds_alternative<InterpValue>(ctdim.value())) {
    return absl::get<InterpValue>(ctdim.value()).GetBitValueInt64();
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

  XLS_LOG(FATAL) << "Unhandled variant for ConcreteTypeDim: "
                 << ctdim.ToString();
}

absl::StatusOr<DerefVariant> EvaluateToStructOrEnumOrAnnotation(
    TypeDefinition type_definition, InterpBindings* bindings,
    AbstractInterpreter* interp, std::vector<Expr*>* parametrics) {
  while (absl::holds_alternative<TypeDef*>(type_definition)) {
    TypeDef* type_def = absl::get<TypeDef*>(type_definition);
    TypeAnnotation* annotation = type_def->type_annotation();
    if (auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(annotation)) {
      type_definition = type_ref->type_ref()->type_definition();
      if (parametrics != nullptr) {
        parametrics->insert(parametrics->end(), type_ref->parametrics().begin(),
                            type_ref->parametrics().end());
      }
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

  auto* colon_ref = absl::get<ColonRef*>(type_definition);
  std::string identifier =
      absl::get<NameRef*>(colon_ref->subject())->identifier();
  std::string attr = colon_ref->attr();
  XLS_ASSIGN_OR_RETURN(Module * imported_module,
                       bindings->ResolveModule(identifier));
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       imported_module->GetTypeDefinition(attr));
  XLS_ASSIGN_OR_RETURN(const InterpBindings* imported_bindings,
                       GetOrCreateTopLevelBindings(imported_module, interp));
  InterpBindings new_bindings(/*parent=*/imported_bindings);
  return EvaluateToStructOrEnumOrAnnotation(td, &new_bindings, interp,
                                            parametrics);
}

// Helper that grabs the type_definition field out of a TypeRef and resolves it.
static absl::StatusOr<DerefVariant> DerefTypeRef(
    TypeRef* type_ref, InterpBindings* bindings, AbstractInterpreter* interp,
    std::vector<Expr*>* parametrics) {
  return EvaluateToStructOrEnumOrAnnotation(type_ref->type_definition(),
                                            bindings, interp, parametrics);
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
  InterpBindings nested_bindings(bindings);
  XLS_CHECK_EQ(struct_def->parametric_bindings().size(), parametrics.size());
  for (int64_t i = 0; i < parametrics.size(); ++i) {
    ParametricBinding* p = struct_def->parametric_bindings()[i];
    Expr* d = parametrics[i];
    if (Number* n = dynamic_cast<Number*>(d)) {
      int64_t value = n->GetAsUint64().value();
      TypeAnnotation* type_annotation = n->type_annotation();
      if (type_annotation == nullptr) {
        // If the number didn't have a type annotation, use the one from the
        // parametric we're binding to.
        type_annotation = p->type_annotation();
      }
      XLS_RET_CHECK(type_annotation != nullptr)
          << "`" << n->ToString() << "` @ " << n->span();
      auto* builtin_type =
          dynamic_cast<BuiltinTypeAnnotation*>(type_annotation);
      XLS_CHECK(builtin_type != nullptr);
      int64_t bit_count = builtin_type->GetBitCount();
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

// ConcretizeTypeAnnotation() broken out case for handling
// TypeRefTypeAnnotation, as it's the most involved (propagates parametrics
// through type aliases).
static absl::StatusOr<std::unique_ptr<ConcreteType>>
ConcretizeTypeRefTypeAnnotation(TypeRefTypeAnnotation* type_ref,
                                InterpBindings* bindings,
                                AbstractInterpreter* interp) {
  TypeDefinition type_defn = type_ref->type_ref()->type_definition();
  if (absl::holds_alternative<EnumDef*>(type_defn)) {
    auto* enum_def = absl::get<EnumDef*>(type_defn);
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<ConcreteType> underlying_type,
        ConcretizeType(enum_def->type_annotation(), bindings, interp));
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count,
                         underlying_type->GetTotalBitCount());
    return absl::make_unique<EnumType>(enum_def, bit_count);
  }

  // Start with the parametrics that are given in the type reference, and then
  // as we dereference to the final type we accumulate more parametrics.
  std::vector<Expr*> parametrics = type_ref->parametrics();
  XLS_ASSIGN_OR_RETURN(
      DerefVariant deref,
      DerefTypeRef(type_ref->type_ref(), bindings, interp, &parametrics));

  // We may have resolved to a deref'd definition outside of the current
  // module, in which case we need to get the top level bindings for that
  // (dereferenced) module.
  absl::optional<InterpBindings> derefd_bindings;
  Module* derefd_module = ToAstNode(deref)->owner();
  if (derefd_module != type_ref->owner()) {
    XLS_ASSIGN_OR_RETURN(const InterpBindings* top_level_bindings,
                         GetOrCreateTopLevelBindings(derefd_module, interp));
    derefd_bindings.emplace(top_level_bindings);
    bindings = &*derefd_bindings;
  }

  absl::optional<InterpBindings> struct_parametric_bindings;
  if (!parametrics.empty()) {
    XLS_RET_CHECK(absl::holds_alternative<StructDef*>(deref));
    auto* struct_def = absl::get<StructDef*>(deref);
    XLS_ASSIGN_OR_RETURN(
        struct_parametric_bindings,
        BindingsWithStructParametrics(struct_def, parametrics, bindings));
    bindings = &struct_parametric_bindings.value();
  }

  AbstractInterpreter::ScopedTypeInfoSwap stis(interp, derefd_module);
  return ConcretizeType(deref, bindings, interp);
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeTypeAnnotation(
    TypeAnnotation* type, InterpBindings* bindings,
    AbstractInterpreter* interp) {
  XLS_RET_CHECK_EQ(type->owner(), interp->GetCurrentTypeInfo()->module());
  XLS_VLOG(5) << "Concretizing type annotation: " << type->ToString()
              << " node type: " << type->GetNodeTypeName() << " in module "
              << type->owner()->name();

  // class TypeRefTypeAnnotation
  if (auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(type)) {
    return ConcretizeTypeRefTypeAnnotation(type_ref, bindings, interp);
  }

  // class TupleTypeAnnotation
  if (auto* tuple = dynamic_cast<TupleTypeAnnotation*>(type)) {
    std::vector<std::unique_ptr<ConcreteType>> members;
    for (TypeAnnotation* member : tuple->members()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_member,
                           ConcretizeType(member, bindings, interp));
      members.push_back(std::move(concrete_member));
    }
    return absl::make_unique<TupleType>(std::move(members));
  }

  // class ArrayTypeAnnotation
  if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    XLS_ASSIGN_OR_RETURN(int64_t dim,
                         ResolveDim(array->dim(), bindings, interp));
    TypeAnnotation* elem_type = array->element_type();
    XLS_VLOG(3) << "Resolved array dim to: " << dim
                << " elem_type: " << elem_type->ToString();
    if (auto* builtin_elem = dynamic_cast<BuiltinTypeAnnotation*>(elem_type);
        builtin_elem != nullptr && builtin_elem->GetBitCount() == 0) {
      return std::make_unique<BitsType>(builtin_elem->GetSignedness(), dim);
    }
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_elem_type,
                         ConcretizeType(elem_type, bindings, interp));
    auto concrete_dim = ConcreteTypeDim::CreateU32(dim);
    return std::make_unique<ArrayType>(std::move(concrete_elem_type),
                                       concrete_dim);
  }

  // class BuiltinTypeAnnotation
  if (auto* builtin = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    bool signedness = builtin->GetSignedness();
    int64_t bit_count = builtin->GetBitCount();
    return absl::make_unique<BitsType>(signedness, bit_count);
  }

  return absl::UnimplementedError("Cannot concretize type annotation: " +
                                  type->ToString());
}

absl::StatusOr<std::unique_ptr<ConcreteType>> ConcretizeType(
    ConcretizeVariant type, InterpBindings* bindings,
    AbstractInterpreter* interp) {
  XLS_RET_CHECK_EQ(ToAstNode(type)->owner(),
                   interp->GetCurrentTypeInfo()->module());
  // class EnumDef
  if (EnumDef** penum_def = absl::get_if<EnumDef*>(&type)) {
    return ConcretizeType((*penum_def)->type_annotation(), bindings, interp);
  }
  // class StructDef
  if (StructDef** pstruct_def = absl::get_if<StructDef*>(&type)) {
    XLS_VLOG(5) << "ConcretizeType StructDef: " << (*pstruct_def)->ToString();
    std::vector<std::unique_ptr<ConcreteType>> members;
    for (auto& [name_def, type_annotation] : (*pstruct_def)->members()) {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ConcreteType> concretized,
          ConcretizeTypeAnnotation(type_annotation, bindings, interp));
      members.push_back(std::move(concretized));
    }
    return absl::make_unique<TupleType>(std::move(members));
  }
  // class TypeAnnotation
  return ConcretizeTypeAnnotation(absl::get<TypeAnnotation*>(type), bindings,
                                  interp);
}

}  // namespace xls::dslx
