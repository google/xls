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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"

namespace xls {

// keep-sorted start
using ::xls::dslx::ConstantDef;
using ::xls::dslx::Expr;
using ::xls::dslx::InterpValue;
using ::xls::dslx::InterpValueTag;
using ::xls::dslx::Module;
using ::xls::dslx::Number;
using ::xls::dslx::TypeAnnotation;
// keep-sorted end

namespace {

absl::StatusOr<InterpValue> GenerateBitValue(absl::BitGenRef bit_gen,
                                             int64_t bit_count,
                                             bool is_signed) {
  InterpValueTag tag =
      is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  if (bit_count == 0) {
    return InterpValue::MakeBits(tag, Bits(0));
  }

  return InterpValue::MakeBits(tag, GenerateBits(bit_gen, bit_count));
}

absl::StatusOr<InterpValue> GenerateBitValue(
    absl::BitGenRef bit_gen, const dslx::BitsLikeProperties& bits_like) {
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_like.size.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like.is_signed.GetAsBool());
  return GenerateBitValue(bit_gen, bit_count, is_signed);
}

// Evaluates the given `Expr*` (holding the declaration of e.g. an
// `ArrayTypeAnnotation`'s size) and returns its resolved integer value.
//
// This relies on current behavior of `AstGenerator`, namely that array dims are
// pure `Number` nodes or are references to `ConstantDefs` (potentially via a
// series of `NameRefs`) whose values are `Number` nodes.
absl::StatusOr<int64_t> EvaluateDimExpr(const dslx::Expr* dim) {
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
      return EvaluateDimExpr(const_def->value());
    }

    const Expr* expr = dynamic_cast<const Expr*>(definer);
    XLS_RET_CHECK_NE(expr, nullptr);
    return EvaluateDimExpr(expr);
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

// Returns a number n in the range [0, limit), where Pr(n == k) is proportional
// to (limit - k); the distribution density is "uniformly decreasing", so the
// result is biased toward zero.
int64_t UniformlyDecreasing(absl::BitGenRef bit_gen, int64_t limit) {
  CHECK_GT(limit, 0);
  if (limit == 1) {  // Only one possible value.
    return 0;
  }
  int64_t x = absl::Uniform(bit_gen, 0, limit);
  int64_t y = absl::Uniform(bit_gen, 0, limit + 1);
  return std::min(x, y);
}

}  // namespace

Bits GenerateBits(absl::BitGenRef bit_gen, int64_t bit_count) {
  if (bit_count == 0) {
    return Bits(0);
  }
  enum PatternKind : std::uint8_t {
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

    // Sentinel marking the end of the enum
    kEndSentinel,
  };
  PatternKind choice = static_cast<PatternKind>(
      absl::Uniform<std::underlying_type_t<PatternKind>>(bit_gen, kZero,
                                                         kEndSentinel));
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
      int64_t index = absl::Uniform(bit_gen, 0, bit_count);
      bitmap.Set(index, true);
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kRandom: {
      InlineBitmap bitmap(bit_count);
      for (int64_t i = 0; i < bit_count; ++i) {
        bitmap.Set(i, absl::Bernoulli(bit_gen, 0.5));
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
    default:
      LOG(FATAL) << "Impossible choice: " << choice;
  }
}

absl::StatusOr<Expr*> GenerateDslxConstant(absl::BitGenRef bit_gen,
                                           Module* module,
                                           TypeAnnotation* type) {
  dslx::Span fake_span = dslx::FakeSpan();

  if (std::optional<dslx::BitVectorMetadata> metadata =
          dslx::ExtractBitVectorMetadata(type);
      metadata.has_value()) {
    absl::StatusOr<int64_t> bit_count = absl::visit(
        xls::Visitor{
            [&](int64_t bit_count) -> absl::StatusOr<int64_t> {
              return bit_count;
            },
            [&](Expr* expr) -> absl::StatusOr<int64_t> {
              absl::StatusOr<int64_t> bit_count = EvaluateDimExpr(expr);
              // If we were able to opportunistically evaluate the dim
              // expression to an `int64_t`, then we're good and we just return
              // that.
              if (bit_count.ok()) {
                return bit_count.value();
              }
              return absl::InvalidArgumentError(
                  absl::StrFormat("Cannot generate constants via parameterized "
                                  "bit counts; got: `%s` in `%s`",
                                  expr->ToString(), type->ToString()));
            },
        },
        metadata->bit_count);
    XLS_RETURN_IF_ERROR(bit_count.status());
    XLS_ASSIGN_OR_RETURN(
        dslx::InterpValue num_value,
        GenerateBitValue(bit_gen, bit_count.value(), metadata->is_signed));
    return module->Make<Number>(fake_span, num_value.ToHumanString(),
                                dslx::NumberKind::kOther, type);
  }

  if (auto* array_type = dynamic_cast<dslx::ArrayTypeAnnotation*>(type);
      array_type != nullptr) {
    dslx::TypeAnnotation* element_type = array_type->element_type();
    XLS_ASSIGN_OR_RETURN(int64_t array_size,
                         EvaluateDimExpr(array_type->dim()));
    // Handle the array-type-is-actually-a-bits-type case.
    if (auto* builtin_type_annot =
            dynamic_cast<dslx::BuiltinTypeAnnotation*>(element_type);
        builtin_type_annot != nullptr) {
      dslx::BuiltinType builtin_type = builtin_type_annot->builtin_type();
      if (builtin_type == dslx::BuiltinType::kBits ||
          builtin_type == dslx::BuiltinType::kUN ||
          builtin_type == dslx::BuiltinType::kSN) {
        Bits num_value = GenerateBits(bit_gen, array_size);
        return module->Make<Number>(fake_span, BitsToString(num_value),
                                    dslx::NumberKind::kOther, type);
      }
    }

    std::vector<Expr*> members;
    members.reserve(array_size);
    for (int i = 0; i < array_size; i++) {
      XLS_ASSIGN_OR_RETURN(Expr * member,
                           GenerateDslxConstant(bit_gen, module, element_type));
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
                           GenerateDslxConstant(bit_gen, module, member_type));
      members.push_back(member);
    }
    return module->Make<dslx::XlsTuple>(fake_span, members,
                                        /*has_trailing_comma=*/false);
  }

  auto* typeref_type = dynamic_cast<dslx::TypeRefTypeAnnotation*>(type);
  XLS_RET_CHECK_NE(typeref_type, nullptr);
  auto* typeref = typeref_type->type_ref();
  return absl::visit(
      Visitor{
          [&](dslx::TypeAlias* type_alias) -> absl::StatusOr<Expr*> {
            return GenerateDslxConstant(bit_gen, module,
                                        &type_alias->type_annotation());
          },
          [&](dslx::StructDef* struct_def) -> absl::StatusOr<Expr*> {
            std::vector<std::pair<std::string, Expr*>> members;
            for (const auto& [_, member_name, member_type] :
                 struct_def->members()) {
              XLS_ASSIGN_OR_RETURN(
                  Expr * member_value,
                  GenerateDslxConstant(bit_gen, module, member_type));
              members.push_back(std::make_pair(member_name, member_value));
            }
            auto* type_ref = module->Make<dslx::TypeRef>(fake_span, struct_def);
            auto* type_ref_type_annotation =
                module->Make<dslx::TypeRefTypeAnnotation>(
                    fake_span, type_ref, std::vector<dslx::ExprOrType>{});
            return module->Make<dslx::StructInstance>(
                fake_span, type_ref_type_annotation, members);
          },
          [&](dslx::ProcDef* proc_def) -> absl::StatusOr<Expr*> {
            // TODO: https://github.com/google/xls/issues/836 - Support
            // impl-style procs.
            return absl::InvalidArgumentError(
                "Impl-style procs are not yet supported.");
          },
          [&](dslx::EnumDef* enum_def) -> absl::StatusOr<Expr*> {
            const std::vector<dslx::EnumMember>& values = enum_def->values();
            int64_t value_idx =
                absl::Uniform(bit_gen, size_t{0}, values.size());
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

static absl::StatusOr<InterpValue> GenerateBitsLikeInterpValue(
    absl::BitGenRef bit_gen, const dslx::BitsLikeProperties& bits_like,
    absl::Span<const InterpValue> prior) {
  if (prior.empty() || absl::Bernoulli(bit_gen, 0.5)) {
    return GenerateBitValue(bit_gen, bits_like);
  }

  // Try to mutate a prior argument. If it happens to not be a bits type that we
  // look at, then just generate an unbiased argument.
  int64_t index = absl::Uniform(bit_gen, size_t{0}, prior.size());
  if (!prior[index].IsBits()) {
    return GenerateBitValue(bit_gen, bits_like);
  }

  Bits to_mutate = prior[index].GetBitsOrDie();

  XLS_ASSIGN_OR_RETURN(int64_t target_bit_count, bits_like.size.GetAsInt64());
  if (target_bit_count > to_mutate.bit_count()) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue addendum,
        GenerateBitValue(bit_gen, target_bit_count - to_mutate.bit_count(),
                         /*is_signed=*/false));
    to_mutate = bits_ops::Concat({to_mutate, addendum.GetBitsOrDie()});
  } else {
    to_mutate = to_mutate.Slice(0, target_bit_count);
  }

  InlineBitmap bitmap = to_mutate.bitmap();
  XLS_RET_CHECK_EQ(bitmap.bit_count(), target_bit_count);
  if (target_bit_count > 0) {
    int64_t mutation_count = UniformlyDecreasing(bit_gen, target_bit_count);
    for (int64_t i = 0; i < mutation_count; ++i) {
      // Pick a random bit and flip it.
      int64_t bitno = absl::Uniform<int64_t>(bit_gen, 0, target_bit_count);
      bitmap.Set(bitno, !bitmap.Get(bitno));
    }
  }
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like.is_signed.GetAsBool());
  auto tag = is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  return InterpValue::MakeBits(tag, Bits::FromBitmap(std::move(bitmap)));
}

absl::StatusOr<InterpValue> GenerateInterpValue(
    absl::BitGenRef bit_gen, const dslx::Type& arg_type,
    absl::Span<const InterpValue> prior) {
  XLS_RET_CHECK(!arg_type.IsMeta()) << arg_type.ToString();
  XLS_RET_CHECK(dynamic_cast<const dslx::BitsConstructorType*>(&arg_type) ==
                nullptr)
      << "`BitsConstructorType`s are not valid InterpValue types.";

  if (auto* channel_type = dynamic_cast<const dslx::ChannelType*>(&arg_type)) {
    // For channels, the argument must be of its payload type.
    return GenerateInterpValue(bit_gen, channel_type->payload_type(), prior);
  }
  if (auto* tuple_type = dynamic_cast<const dslx::TupleType*>(&arg_type)) {
    std::vector<InterpValue> members;
    for (const std::unique_ptr<dslx::Type>& t : tuple_type->members()) {
      XLS_ASSIGN_OR_RETURN(InterpValue member,
                           GenerateInterpValue(bit_gen, *t, prior));
      members.push_back(member);
    }
    return InterpValue::MakeTuple(members);
  }

  // Note: we have to test for BitsLike before ArrayType because
  // array-of-bits-constructor looks like an array but is actually bits-like.
  std::optional<dslx::BitsLikeProperties> bits_like = GetBitsLike(arg_type);
  if (bits_like.has_value()) {
    return GenerateBitsLikeInterpValue(bit_gen, bits_like.value(), prior);
  }

  if (auto* array_type = dynamic_cast<const dslx::ArrayType*>(&arg_type)) {
    std::vector<InterpValue> elements;
    const dslx::Type& element_type = array_type->element_type();
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    for (int64_t i = 0; i < array_size; ++i) {
      XLS_ASSIGN_OR_RETURN(InterpValue element,
                           GenerateInterpValue(bit_gen, element_type, prior));
      elements.push_back(element);
    }
    return InterpValue::MakeArray(std::move(elements));
  }

  return absl::UnimplementedError("Unsupported type for GenerateInterpValue");
}

absl::StatusOr<std::vector<InterpValue>> GenerateInterpValues(
    absl::BitGenRef bit_gen, absl::Span<const dslx::Type* const> arg_types) {
  std::vector<InterpValue> args;
  for (const dslx::Type* arg_type : arg_types) {
    XLS_RET_CHECK(arg_type != nullptr);
    XLS_RET_CHECK(!arg_type->IsMeta());
    XLS_RET_CHECK(dynamic_cast<const dslx::BitsConstructorType*>(arg_type) ==
                  nullptr)
        << "`BitsConstructorType`s are not valid parameter types.";
    XLS_ASSIGN_OR_RETURN(InterpValue arg,
                         GenerateInterpValue(bit_gen, *arg_type, args));
    args.push_back(std::move(arg));
  }
  return args;
}

}  // namespace xls
