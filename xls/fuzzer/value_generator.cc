// Copyright 2022 The XLS Authors
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
#include "xls/fuzzer/value_generator.h"

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"

namespace xls {

using dslx::ArrayType;
using dslx::BitsType;
using dslx::ChannelType;
using dslx::ConcreteType;
using dslx::ConstantDef;
using dslx::Expr;
using dslx::InterpValue;
using dslx::InterpValueTag;
using dslx::Module;
using dslx::Number;
using dslx::TupleType;
using dslx::TypeAnnotation;

bool ValueGenerator::RandomBool() {
  std::bernoulli_distribution d(0.5);
  return d(rng_);
}

int64_t ValueGenerator::RandomIntWithExpectedValue(float expected_value,
                                                   int64_t lower_limit) {
  XLS_CHECK_GE(expected_value, lower_limit);
  std::poisson_distribution<int64_t> distribution(expected_value - lower_limit);
  return distribution(rng_) + lower_limit;
}

float ValueGenerator::RandomFloat() {
  std::uniform_real_distribution<float> g(0.0f, 1.0f);
  return g(rng_);
}

double ValueGenerator::RandomDouble() {
  std::uniform_real_distribution<double> g(0.0f, 1.0f);
  return g(rng_);
}

int64_t ValueGenerator::RandRange(int64_t limit) { return RandRange(0, limit); }

int64_t ValueGenerator::RandRange(int64_t start, int64_t limit) {
  XLS_CHECK_GT(limit, start);
  std::uniform_int_distribution<int64_t> g(start, limit - 1);
  int64_t value = g(rng_);
  XLS_CHECK_LT(value, limit);
  XLS_CHECK_GE(value, start);
  return value;
}

int64_t ValueGenerator::RandRangeBiasedTowardsZero(int64_t limit) {
  XLS_CHECK_GT(limit, 0);
  if (limit == 1) {  // Only one possible value.
    return 0;
  }
  std::array<double, 3> i = {{0, 0, static_cast<double>(limit)}};
  std::array<double, 3> w = {{0, 1, 0}};
  std::piecewise_linear_distribution<double> d(i.begin(), i.end(), w.begin());
  double triangular = d(rng_);
  int64_t result = static_cast<int64_t>(std::ceil(triangular)) - 1;
  XLS_CHECK_GE(result, 0);
  XLS_CHECK_LT(result, limit);
  return result;
}

Bits ValueGenerator::GenerateBits(int64_t bit_count) {
  if (bit_count == 0) {
    return Bits(0);
  }
  enum PatternKind {
    kZero,
    kAllOnes,
    // Just the high bit is unset, otherwise all ones.
    kAllButHighOnes,
    // Alternating 01 bit pattern.
    kOffOn,
    // Alternating 10 bit pattern.
    kOnOff,
    kOneHot,
    kRandom,
  };
  PatternKind choice = static_cast<PatternKind>(RandRange(kRandom + 1));
  switch (choice) {
    case kZero:
      return Bits(bit_count);
    case kAllOnes:
      return Bits::AllOnes(bit_count);
    case kAllButHighOnes:
      return bits_ops::ShiftRightLogical(Bits::AllOnes(bit_count), 1);
    case kOffOn: {
      InlineBitmap bitmap(bit_count);
      for (int64_t i = 1; i < bit_count; i += 2) {
        bitmap.Set(i, true);
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kOnOff: {
      InlineBitmap bitmap(bit_count);
      for (int64_t i = 0; i < bit_count; i += 2) {
        bitmap.Set(i, true);
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kOneHot: {
      InlineBitmap bitmap(bit_count);
      int64_t index = RandRange(bit_count);
      bitmap.Set(index, true);
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kRandom: {
      InlineBitmap bitmap(bit_count);
      for (int64_t i = 0; i < bit_count; ++i) {
        bitmap.Set(i, RandomBool());
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
    default:
      XLS_LOG(FATAL) << "Impossible choice: " << choice;
  }
}

absl::StatusOr<InterpValue> ValueGenerator::GenerateBitValue(int64_t bit_count,
                                                             bool is_signed) {
  InterpValueTag tag =
      is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  if (bit_count == 0) {
    return InterpValue::MakeBits(tag, Bits(0));
  }

  return InterpValue::MakeBits(tag, GenerateBits(bit_count));
}

// Note: "unbiased" here refers to the fact we don't use the history of
// previously generated values, but just sample arbitrarily something for the
// given bit count of the bits type. You'll see other routines taking "prior" as
// a history to help prevent repetition that could hide bugs.
absl::StatusOr<InterpValue> ValueGenerator::GenerateUnbiasedValue(
    const BitsType& bits_type) {
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type.size().GetAsInt64());
  return GenerateBitValue(bit_count, bits_type.is_signed());
}

absl::StatusOr<InterpValue> ValueGenerator::GenerateInterpValue(
    const ConcreteType& arg_type, absl::Span<const InterpValue> prior) {
  if (auto* channel_type = dynamic_cast<const ChannelType*>(&arg_type)) {
    // For channels, the argument must be of its payload type.
    return GenerateInterpValue(channel_type->payload_type(), prior);
  }
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&arg_type)) {
    std::vector<InterpValue> members;
    for (const std::unique_ptr<ConcreteType>& t : tuple_type->members()) {
      XLS_ASSIGN_OR_RETURN(InterpValue member, GenerateInterpValue(*t, prior));
      members.push_back(member);
    }
    return InterpValue::MakeTuple(members);
  }
  if (auto* array_type = dynamic_cast<const ArrayType*>(&arg_type)) {
    std::vector<InterpValue> elements;
    const ConcreteType& element_type = array_type->element_type();
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    for (int64_t i = 0; i < array_size; ++i) {
      XLS_ASSIGN_OR_RETURN(InterpValue element,
                           GenerateInterpValue(element_type, prior));
      elements.push_back(element);
    }
    return InterpValue::MakeArray(std::move(elements));
  }
  auto* bits_type = dynamic_cast<const BitsType*>(&arg_type);
  XLS_RET_CHECK(bits_type != nullptr);
  if (prior.empty() || RandomDouble() < 0.5) {
    return GenerateUnbiasedValue(*bits_type);
  }

  // Try to mutate a prior argument. If it happens to not be a bits type that we
  // look at, then just generate an unbiased argument.
  int64_t index = RandRange(prior.size());
  if (!prior[index].IsBits()) {
    return GenerateUnbiasedValue(*bits_type);
  }

  Bits to_mutate = prior[index].GetBitsOrDie();

  XLS_ASSIGN_OR_RETURN(const int64_t target_bit_count,
                       bits_type->size().GetAsInt64());
  if (target_bit_count > to_mutate.bit_count()) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue addendum,
        GenerateBitValue(target_bit_count - to_mutate.bit_count(),
                         /*is_signed=*/false));
    to_mutate = bits_ops::Concat({to_mutate, addendum.GetBitsOrDie()});
  } else {
    to_mutate = to_mutate.Slice(0, target_bit_count);
  }

  InlineBitmap bitmap = to_mutate.bitmap();
  XLS_RET_CHECK_EQ(bitmap.bit_count(), target_bit_count);
  int64_t mutation_count = RandRangeBiasedTowardsZero(target_bit_count);

  for (int64_t i = 0; i < mutation_count; ++i) {
    // Pick a random bit and flip it.
    int64_t bitno = RandRange(target_bit_count);
    bitmap.Set(bitno, !bitmap.Get(bitno));
  }

  bool is_signed = bits_type->is_signed();
  auto tag = is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  return InterpValue::MakeBits(tag, Bits::FromBitmap(std::move(bitmap)));
}

absl::StatusOr<std::vector<InterpValue>> ValueGenerator::GenerateInterpValues(
    absl::Span<const ConcreteType* const> arg_types) {
  std::vector<InterpValue> args;
  for (const ConcreteType* arg_type : arg_types) {
    XLS_RET_CHECK(arg_type != nullptr);
    XLS_ASSIGN_OR_RETURN(InterpValue arg, GenerateInterpValue(*arg_type, args));
    args.push_back(std::move(arg));
  }
  return args;
}

absl::StatusOr<int64_t> ValueGenerator::GetArraySize(const Expr* dim) {
  if (const auto* number = dynamic_cast<const dslx::Number*>(dim);
      number != nullptr) {
    return ParseNumberAsInt64(number->text());
  }

  if (auto* name_ref = dynamic_cast<const dslx::NameRef*>(dim);
      name_ref != nullptr) {
    const dslx::NameDef* name_def =
        std::get<const dslx::NameDef*>(name_ref->name_def());
    const dslx::AstNode* definer = name_def->definer();
    if (const auto* const_def = dynamic_cast<const dslx::ConstantDef*>(definer);
        const_def != nullptr) {
      return GetArraySize(const_def->value());
    }

    const Expr* expr = dynamic_cast<const Expr*>(definer);
    XLS_RET_CHECK_NE(expr, nullptr);
    return GetArraySize(expr);
  }

  auto* constant_def = dynamic_cast<const ConstantDef*>(dim);
  XLS_RET_CHECK_NE(constant_def, nullptr);

  // Currently, the fuzzer only generates constants with Number-typed
  // values. Should that change (e.g., Binop-defining RHS), this'll need to
  // be updated.
  Number* number = dynamic_cast<Number*>(constant_def->value());
  XLS_RET_CHECK_NE(number, nullptr);
  return ParseNumberAsInt64(number->text());
}

absl::StatusOr<Expr*> ValueGenerator::GenerateDslxConstant(
    Module* module, TypeAnnotation* type) {
  dslx::Span fake_span = dslx::FakeSpan();
  if (auto* builtin_type = dynamic_cast<dslx::BuiltinTypeAnnotation*>(type);
      builtin_type != nullptr) {
    XLS_ASSIGN_OR_RETURN(dslx::InterpValue num_value,
                         GenerateBitValue(builtin_type->GetBitCount(),
                                          builtin_type->GetSignedness()));
    return module->Make<Number>(fake_span, num_value.ToHumanString(),
                                dslx::NumberKind::kOther, type);
  }

  if (auto* array_type = dynamic_cast<dslx::ArrayTypeAnnotation*>(type);
      array_type != nullptr) {
    dslx::TypeAnnotation* element_type = array_type->element_type();
    XLS_ASSIGN_OR_RETURN(int64_t array_size, GetArraySize(array_type->dim()));
    // Handle the array-type-is-actually-a-bits-type case.
    if (auto* builtin_type_annot =
            dynamic_cast<dslx::BuiltinTypeAnnotation*>(element_type);
        builtin_type_annot != nullptr) {
      dslx::BuiltinType builtin_type = builtin_type_annot->builtin_type();
      if (builtin_type == dslx::BuiltinType::kBits ||
          builtin_type == dslx::BuiltinType::kUN ||
          builtin_type == dslx::BuiltinType::kSN) {
        Bits num_value = GenerateBits(array_size);
        return module->Make<Number>(fake_span, num_value.ToString(),
                                    dslx::NumberKind::kOther, type);
      }
    }

    std::vector<Expr*> members;
    members.reserve(array_size);
    for (int i = 0; i < array_size; i++) {
      XLS_ASSIGN_OR_RETURN(Expr * member,
                           GenerateDslxConstant(module, element_type));
      members.push_back(member);
    }

    return module->Make<dslx::Array>(fake_span, members,
                                     /*has_ellipsis=*/false);
  }

  if (auto* tuple_type = dynamic_cast<dslx::TupleTypeAnnotation*>(type);
      tuple_type != nullptr) {
    std::vector<Expr*> members;
    for (auto* member_type : tuple_type->members()) {
      XLS_ASSIGN_OR_RETURN(Expr * member,
                           GenerateDslxConstant(module, member_type));
      members.push_back(member);
    }
    return module->Make<dslx::XlsTuple>(fake_span, members);
  }

  auto* typeref_type = dynamic_cast<dslx::TypeRefTypeAnnotation*>(type);
  XLS_RET_CHECK_NE(typeref_type, nullptr);
  auto* typeref = typeref_type->type_ref();
  return std::visit(
      Visitor{
          [&](dslx::TypeDef* type_def) -> absl::StatusOr<Expr*> {
            return GenerateDslxConstant(module, type_def->type_annotation());
          },
          [&](dslx::StructDef* struct_def) -> absl::StatusOr<Expr*> {
            std::vector<std::pair<std::string, Expr*>> members;
            for (const auto& [member_name, member_type] :
                 struct_def->members()) {
              XLS_ASSIGN_OR_RETURN(Expr * member_value,
                                   GenerateDslxConstant(module, member_type));
              members.push_back(
                  std::make_pair(member_name->identifier(), member_value));
            }
            return module->Make<dslx::StructInstance>(fake_span, struct_def,
                                                      members);
          },
          [&](dslx::EnumDef* enum_def) -> absl::StatusOr<Expr*> {
            const std::vector<dslx::EnumMember>& values = enum_def->values();
            int64_t value_idx = RandRange(values.size());
            const dslx::EnumMember& value = values[value_idx];
            auto* name_ref = module->Make<dslx::NameRef>(
                fake_span, value.name_def->identifier(), value.name_def);
            return module->Make<dslx::ColonRef>(fake_span, name_ref,
                                                value.name_def->identifier());
          },
          [&](dslx::ColonRef* colon_ref) -> absl::StatusOr<Expr*> {
            return absl::UnimplementedError(
                "Generating constants of ColonRef types isn't yet supported.");
          },
      },
      typeref->type_definition());
}

}  // namespace xls
