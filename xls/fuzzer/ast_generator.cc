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

#include "xls/fuzzer/ast_generator.h"

#include <algorithm>
#include <optional>
#include <set>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {

/* static */ std::pair<std::vector<Expr*>, std::vector<TypeAnnotation*>>
AstGenerator::Unzip(absl::Span<const TypedExpr> typed_exprs) {
  std::vector<Expr*> exprs;
  std::vector<TypeAnnotation*> types;
  for (auto& typed_expr : typed_exprs) {
    exprs.push_back(typed_expr.expr);
    types.push_back(typed_expr.type);
  }
  return std::make_pair(std::move(exprs), std::move(types));
}

/* static */ bool AstGenerator::IsUBits(TypeAnnotation* t) {
  if (auto* builtin = dynamic_cast<BuiltinTypeAnnotation*>(t)) {
    return builtin->GetBitCount() != 0 && !builtin->GetSignedness();
  }
  if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(t)) {
    if (auto* builtin =
            dynamic_cast<BuiltinTypeAnnotation*>(array->element_type())) {
      switch (builtin->builtin_type()) {
        case BuiltinType::kBits:
          return true;
        case BuiltinType::kUN:
          return true;
        default:
          return false;
      }
    }
  }
  if (auto* def = dynamic_cast<TypeDef*>(t)) {
    return IsUBits(def->type_annotation());
  }
  return false;
}

/* static */ bool AstGenerator::IsBits(TypeAnnotation* t) {
  if (auto* builtin = dynamic_cast<BuiltinTypeAnnotation*>(t)) {
    return builtin->GetBitCount() != 0;
  }
  if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(t)) {
    if (auto* builtin =
            dynamic_cast<BuiltinTypeAnnotation*>(array->element_type())) {
      switch (builtin->builtin_type()) {
        case BuiltinType::kBits:
          return true;
        case BuiltinType::kUN:
          return true;
        case BuiltinType::kSN:
          return true;
        default:
          return false;
      }
    }
  }
  if (auto* def = dynamic_cast<TypeDef*>(t)) {
    return IsBits(def->type_annotation());
  }
  return false;
}

/* static */ bool AstGenerator::IsArray(TypeAnnotation* t) {
  if (dynamic_cast<ArrayTypeAnnotation*>(t) != nullptr) {
    return !IsBits(t);
  }
  return false;
}

/* static */ bool AstGenerator::IsTuple(TypeAnnotation* t) {
  return dynamic_cast<TupleTypeAnnotation*>(t) != nullptr;
}

/* static */ bool AstGenerator::IsToken(TypeAnnotation* t) {
  auto* token = dynamic_cast<BuiltinTypeAnnotation*>(t);
  return token != nullptr && token->builtin_type() == BuiltinType::kToken;
}

/* static */ bool AstGenerator::IsChannel(TypeAnnotation* t) {
  return dynamic_cast<ChannelTypeAnnotation*>(t) != nullptr;
}

/* static */ bool AstGenerator::IsNil(TypeAnnotation* t) {
  if (auto* tuple = dynamic_cast<TupleTypeAnnotation*>(t);
      tuple != nullptr && tuple->empty()) {
    return true;
  }
  return false;
}

/* static */ bool AstGenerator::EnvContainsArray(const Env& e) {
  return std::any_of(e.begin(), e.end(), [](const auto& item) -> bool {
    return IsArray(item.second.type);
  });
}

/* static */ bool AstGenerator::EnvContainsTuple(const Env& e) {
  return std::any_of(e.begin(), e.end(), [](const auto& item) -> bool {
    return IsTuple(item.second.type);
  });
}

/* static */ bool AstGenerator::EnvContainsToken(const Env& e) {
  return std::any_of(e.begin(), e.end(), [](const auto& item) -> bool {
    return IsToken(item.second.type);
  });
}

/* static */ bool AstGenerator::EnvContainsChannel(const Env& e) {
  return std::any_of(e.begin(), e.end(), [](const auto& item) -> bool {
    return IsChannel(item.second.type);
  });
}

Param* AstGenerator::GenerateParam(TypeAnnotation* type) {
  std::string identifier = GenSym();
  if (type == nullptr) {
    type = GenerateType();
  }
  NameDef* name_def = module_->Make<NameDef>(fake_span_, std::move(identifier),
                                             /*definer=*/nullptr);
  Param* param = module_->Make<Param>(name_def, type);
  name_def->set_definer(param);
  return param;
}

std::vector<Param*> AstGenerator::GenerateParams(int64_t count) {
  std::vector<Param*> params;
  for (int64_t i = 0; i < count; ++i) {
    params.push_back(GenerateParam());
  }
  return params;
}

std::vector<ParametricBinding*> AstGenerator::GenerateParametricBindings(
    int64_t count) {
  std::vector<ParametricBinding*> pbs;
  for (int64_t i = 0; i < count; ++i) {
    std::string identifier = GenSym();
    NameDef* name_def =
        module_->Make<NameDef>(fake_span_, std::move(identifier),
                               /*definer=*/nullptr);
    // TODO(google/xls#460): Currently we only support non-negative values as
    // parametrics -- when that restriction is lifted we should be able to do
    // arbitrary GenerateNumber() calls.
    //
    // TODO(google/xls#461): We only support 64-bit values being mangled into
    // identifier since Bits conversion to decimal only supports that.
    //
    // Starting from 1 is to ensure that we don't get a 0-bit value.
    int64_t bit_count = RandRange(1, 65);
    TypedExpr number = GenerateNumber(BitsAndSignedness{bit_count, false});
    ParametricBinding* pb =
        module_->Make<ParametricBinding>(name_def, number.type, number.expr);
    name_def->set_definer(pb);
    pbs.push_back(pb);
  }
  return pbs;
}

BuiltinTypeAnnotation* AstGenerator::MakeTokenType() {
  return module_->Make<BuiltinTypeAnnotation>(fake_span_, BuiltinType::kToken);
}

TypeAnnotation* AstGenerator::MakeTypeAnnotation(bool is_signed,
                                                 int64_t width) {
  XLS_CHECK_GT(width, 0);
  if (width <= 64) {
    return module_->Make<BuiltinTypeAnnotation>(
        fake_span_, GetBuiltinType(is_signed, width).value());
  }
  auto* element_type = module_->Make<BuiltinTypeAnnotation>(
      fake_span_, is_signed ? BuiltinType::kSN : BuiltinType::kUN);
  Number* dim = MakeNumber(width);
  return module_->Make<ArrayTypeAnnotation>(fake_span_, element_type, dim);
}

absl::StatusOr<Expr*> AstGenerator::GenerateUmin(TypedExpr arg, int64_t other) {
  Number* rhs = MakeNumber(other, arg.type);
  Expr* test = MakeGe(arg.expr, rhs);
  return MakeSel(test, rhs, arg.expr);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCompare(Env* env) {
  BinopKind op = RandomSetChoice<BinopKind>(GetBinopComparisonKinds());
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(env));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;
  Binop* binop = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr);
  return TypedExpr{binop, MakeTypeAnnotation(false, 1)};
}

namespace {

enum class ChannelOpType {
  kRecv,
  kRecvNonBlocking,
  kRecvIf,
  kSend,
  kSendIf,
};

struct ChannelOpInfo {
  ChannelTypeAnnotation::Direction channel_direction;
  bool requires_payload;
  bool requires_predicate;
};

ChannelOpInfo GetChannelOpInfo(ChannelOpType chan_op) {
  switch (chan_op) {
    case ChannelOpType::kRecv:
      return ChannelOpInfo{
          .channel_direction = ChannelTypeAnnotation::Direction::kIn,
          .requires_payload = false,
          .requires_predicate = false};
    case ChannelOpType::kRecvNonBlocking:
      return ChannelOpInfo{
          .channel_direction = ChannelTypeAnnotation::Direction::kIn,
          .requires_payload = false,
          .requires_predicate = false};
    case ChannelOpType::kRecvIf:
      return ChannelOpInfo{
          .channel_direction = ChannelTypeAnnotation::Direction::kIn,
          .requires_payload = false,
          .requires_predicate = true};
    case ChannelOpType::kSend:
      return ChannelOpInfo{
          .channel_direction = ChannelTypeAnnotation::Direction::kOut,
          .requires_payload = true,
          .requires_predicate = false};
    case ChannelOpType::kSendIf:
      return ChannelOpInfo{
          .channel_direction = ChannelTypeAnnotation::Direction::kOut,
          .requires_payload = true,
          .requires_predicate = true};
  }
}

}  // namespace

absl::StatusOr<TypedExpr> AstGenerator::GenerateChannelOp(Env* env) {
  // Lambda that chooses a boolean value for a predicate.
  auto choose_predicate = [this](const TypedExpr& e) -> bool {
    TypeAnnotation* t = e.type;
    return IsUBits(t) && GetTypeBitCount(t) == 1;
  };
  // Equal distribution for channel ops.
  ChannelOpType chan_op_type = RandomChoice<ChannelOpType>(
      {ChannelOpType::kRecv, ChannelOpType::kRecvNonBlocking,
       ChannelOpType::kRecvIf, ChannelOpType::kSend, ChannelOpType::kSendIf});
  ChannelOpInfo chan_op_info = GetChannelOpInfo(chan_op_type);

  // If needed, generate a predicate.
  std::optional<TypedExpr> predicate;
  if (chan_op_info.requires_predicate) {
    predicate = ChooseEnvValueOptional(env, /*take=*/choose_predicate);
    if (!predicate.has_value()) {
      // If there's no natural environment value to use as the predicate,
      // generate a boolean.
      Number* boolean = MakeBool(RandomBool());
      predicate = TypedExpr{boolean, boolean->type_annotation()};
    }
  }

  // Generate an arbitrary type for the channel.
  TypeAnnotation* channel_type = GenerateType();

  // If needed, choose a payload from the environment.
  std::optional<TypedExpr> payload;
  if (chan_op_info.requires_payload) {
    // TODO(vmirian): 8-22-2002 Payloads of the type may not be present in the
    // env. Create payload of the type enabling more ops requiring a payload
    // (e.g. send and send_if).
    XLS_ASSIGN_OR_RETURN(payload, ChooseEnvValue(env, channel_type));
  }

  // Create the channel.
  // TODO(vmirian): 8-22-2022 If payload type exists, create an array of
  // channels.
  ChannelTypeAnnotation* channel_type_annotation =
      module_->Make<ChannelTypeAnnotation>(fake_span_,
                                           chan_op_info.channel_direction,
                                           channel_type, std::nullopt);
  Param* param = GenerateParam(channel_type_annotation);
  proc_properties_.params.push_back(param);
  NameRef* chan_expr = module_->Make<NameRef>(fake_span_, param->identifier(),
                                              param->name_def());

  // Choose a random token for the channel op.
  XLS_ASSIGN_OR_RETURN(TypedExpr token, ChooseEnvValue(env, MakeTokenType()));
  auto* token_name_ref = dynamic_cast<NameRef*>(token.expr);
  XLS_CHECK(token_name_ref != nullptr);

  switch (chan_op_type) {
    case ChannelOpType::kRecv:
      return TypedExpr{
          module_->Make<Recv>(fake_span_, token_name_ref, chan_expr),
          MakeTupleType({token.type, channel_type})};
    case ChannelOpType::kRecvNonBlocking:
      return TypedExpr{
          module_->Make<RecvNonBlocking>(fake_span_, token_name_ref, chan_expr),
          MakeTupleType(
              {token.type, channel_type, MakeTypeAnnotation(false, 1)})};
    case ChannelOpType::kRecvIf:
      return TypedExpr{module_->Make<RecvIf>(fake_span_, token_name_ref,
                                             chan_expr, predicate.value().expr),
                       MakeTupleType({token.type, channel_type})};
    case ChannelOpType::kSend:
      return TypedExpr{module_->Make<Send>(fake_span_, token_name_ref,
                                           chan_expr, payload.value().expr),
                       MakeTupleType({token.type})};
    case ChannelOpType::kSendIf:
      return TypedExpr{
          module_->Make<SendIf>(fake_span_, token_name_ref, chan_expr,
                                predicate.value().expr, payload.value().expr),
          token.type};
  }
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateJoinOp(Env* env) {
  // Lambda that chooses a token TypedExpr.
  auto token_predicate = [](const TypedExpr& e) -> bool {
    return IsToken(e.type);
  };
  std::vector<TypedExpr> tokens = GatherAllValues(env, token_predicate);
  int64_t token_count = RandRange(1, tokens.size() + 1);
  std::vector<Expr*> tokens_to_join(token_count);
  for (int64_t i = 0; i < token_count; ++i) {
    int64_t token_index = RandRange(0, tokens.size());
    tokens_to_join[i] = tokens[token_index].expr;
  }
  return TypedExpr{module_->Make<Join>(fake_span_, tokens_to_join),
                   MakeTokenType()};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCompareArray(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueArray(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValue(env, lhs.type));
  BinopKind op = RandomBool() ? BinopKind::kEq : BinopKind::kNe;
  return TypedExpr{module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
                   MakeTypeAnnotation(false, 1)};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCompareTuple(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueTuple(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValue(env, lhs.type));
  BinopKind op = RandomBool() ? BinopKind::kEq : BinopKind::kNe;
  return TypedExpr{module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
                   MakeTypeAnnotation(false, 1)};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateShift(Env* env) {
  BinopKind op = RandomSetChoice<BinopKind>(GetBinopShifts());
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueBits(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueUBits(env));
  if (RandomFloat() < 0.8) {
    // Clamp the shift rhs to be in range most of the time.
    int64_t bit_count = GetTypeBitCount(rhs.type);
    int64_t new_upper = RandRange(bit_count);
    XLS_ASSIGN_OR_RETURN(rhs.expr, GenerateUmin(rhs, new_upper));
  } else if (RandomBool()) {
    // Generate a numerical value (Number) as an untyped literal instead of the
    // value we chose above.
    int64_t shift_amount = RandRange(0, GetTypeBitCount(lhs.type));
    rhs = TypedExpr();
    rhs.expr = MakeNumber(shift_amount);
  }
  return TypedExpr{module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
                   lhs.type};
}

absl::StatusOr<TypedExpr>
AstGenerator::GeneratePartialProductDeterministicGroup(Env* env) {
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(env));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;
  bool is_signed = RandomBool();

  std::string op = (is_signed) ? "smulp" : "umulp";

  XLS_CHECK(IsBits(lhs.type));
  XLS_CHECK(IsBits(rhs.type));

  TypedExpr lhs_cast, rhs_cast;
  // Don't need a cast if lhs.type matches the sign of the op
  if (is_signed != IsUBits(lhs.type)) {
    lhs_cast = lhs;
  } else {
    lhs_cast.type = MakeTypeAnnotation(is_signed, GetTypeBitCount(lhs.type));
    lhs_cast.expr = module_->Make<Cast>(fake_span_, lhs.expr, lhs_cast.type);
  }
  // Don't need a cast if rhs.type matches the sign of the op
  if (is_signed != IsUBits(rhs.type)) {
    rhs_cast = rhs;
  } else {
    rhs_cast.type = MakeTypeAnnotation(is_signed, GetTypeBitCount(rhs.type));
    rhs_cast.expr = module_->Make<Cast>(fake_span_, rhs.expr, rhs_cast.type);
  }

  auto mulp = TypedExpr{module_->Make<Invocation>(
                            fake_span_, MakeBuiltinNameRef(op),
                            std::vector<Expr*>{lhs_cast.expr, rhs_cast.expr}),
                        MakeTupleType({lhs_cast.type, rhs_cast.type})};
  std::string mulp_identifier = GenSym();
  auto* mulp_name_def =
      module_->Make<NameDef>(fake_span_, mulp_identifier, /*definer=*/nullptr);
  auto* mulp_name_ref = MakeNameRef(mulp_name_def);
  auto* ndt = module_->Make<NameDefTree>(fake_span_, mulp_name_def);
  auto mulp_lhs = module_->Make<TupleIndex>(fake_span_, mulp_name_ref,
                                            /*index=*/MakeNumber(0));
  auto mulp_rhs = module_->Make<TupleIndex>(fake_span_, mulp_name_ref,
                                            /*index=*/MakeNumber(1));
  auto* sum =
      module_->Make<Binop>(fake_span_, BinopKind::kAdd, mulp_lhs, mulp_rhs);
  auto* let = module_->Make<Let>(fake_span_, /*name_def_tree=*/ndt,
                                 /*type=*/mulp.type, /*rhs=*/mulp.expr,
                                 /*body=*/sum, /*is_const=*/false);
  auto* block = module_->Make<Block>(fake_span_, let);
  return TypedExpr{block, lhs_cast.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBinop(Env* env) {
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(env));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;
  BinopKind op = RandomSetChoice(binops_);
  if (GetBinopShifts().contains(op)) {
    return GenerateShift(env);
  }
  return TypedExpr{module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
                   lhs.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateLogicalOp(Env* env) {
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(env, /*bit_count=*/1));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;

  // Pick some operation to do.
  BinopKind op = RandomChoice<BinopKind>(
      {BinopKind::kAnd, BinopKind::kOr, BinopKind::kXor});
  return TypedExpr{module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
                   lhs.type};
}

Number* AstGenerator::MakeNumber(int64_t value, TypeAnnotation* type) {
  if (IsBuiltinBool(type)) {
    XLS_CHECK(value == 0 || value == 1) << value;
    return module_->Make<Number>(fake_span_, value == 1 ? "true" : "false",
                                 NumberKind::kBool, type);
  }
  return module_->Make<Number>(fake_span_, absl::StrFormat("%d", value),
                               NumberKind::kOther, type);
}

Number* AstGenerator::MakeNumberFromBits(const Bits& value,
                                         TypeAnnotation* type) {
  return module_->Make<Number>(fake_span_,
                               value.ToString(FormatPreference::kHex),
                               NumberKind::kOther, type);
}

int64_t AstGenerator::GetTypeBitCount(TypeAnnotation* type) {
  std::string type_str = type->ToString();
  if (type_str == "uN" || type_str == "sN" || type_str == "bits") {
    // These types are not valid alone, but as the element type of an array
    // (e.g. uN[42]) where they effectively have a width of one bit.
    return 1;
  }

  if (auto* builtin = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
    return builtin->GetBitCount();
  }
  if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(type)) {
    return GetArraySize(array) * GetTypeBitCount(array->element_type());
  }
  if (auto* tuple = dynamic_cast<TupleTypeAnnotation*>(type)) {
    int64_t total = 0;
    for (TypeAnnotation* type : tuple->members()) {
      total += GetTypeBitCount(type);
    }
    return total;
  }
  if (auto* def = dynamic_cast<TypeDef*>(type)) {
    return GetTypeBitCount(def->type_annotation());
  }

  return type_bit_counts_.at(type_str);
}

int64_t AstGenerator::GetArraySize(const ArrayTypeAnnotation* type) {
  Expr* dim = type->dim();
  if (auto* number = dynamic_cast<Number*>(dim)) {
    return number->GetAsUint64().value();
  }
  auto* const_ref = dynamic_cast<ConstRef*>(dim);
  ConstantDef* const_def = constants_[const_ref->identifier()];
  Number* number = dynamic_cast<Number*>(const_def->value());
  XLS_CHECK(number != nullptr) << const_def->ToString();
  return number->GetAsUint64().value();
}

ConstRef* AstGenerator::GetOrCreateConstRef(int64_t value,
                                            std::optional<int64_t> want_width) {
  // We use a canonical naming scheme so we can detect duplicate requests for
  // the same value.
  int64_t width;
  if (want_width.has_value()) {
    width = want_width.value();
  } else {
    width = std::max(int64_t{1},
                     static_cast<int64_t>(std::ceil(std::log2(value + 1))));
  }
  std::string identifier = absl::StrFormat("W%d_V%d", width, value);
  ConstantDef* constant_def;
  if (auto it = constants_.find(identifier); it != constants_.end()) {
    constant_def = it->second;
  } else {
    TypeAnnotation* size_type = MakeTypeAnnotation(false, width);

    NameDef* name_def =
        module_->Make<NameDef>(fake_span_, identifier, /*definer=*/nullptr);
    constant_def = module_->Make<ConstantDef>(fake_span_, name_def,
                                              MakeNumber(value, size_type),
                                              /*is_public=*/false);
    name_def->set_definer(constant_def);
    constants_[identifier] = constant_def;
  }
  return module_->Make<ConstRef>(fake_span_, identifier,
                                 constant_def->name_def());
}

ArrayTypeAnnotation* AstGenerator::MakeArrayType(TypeAnnotation* element_type,
                                                 int64_t array_size) {
  Expr* dim;
  if (RandomBool()) {
    // Get-or-create a module level constant for the array size.
    dim = GetOrCreateConstRef(array_size, /*want_width=*/32);
  } else {
    dim = MakeNumber(array_size);
  }

  return module_->Make<ArrayTypeAnnotation>(
      fake_span_, MakeTypeRefTypeAnnotation(element_type), dim);
}

Range* AstGenerator::MakeRange(Expr* zero, Expr* arg) {
  return module_->Make<Range>(fake_span_, zero, arg);
}

Bits AstGenerator::ChooseBitPattern(int64_t bit_count) {
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
        bitmap.Set(i, 1);
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kOnOff: {
      InlineBitmap bitmap(bit_count);
      for (int64_t i = 0; i < bit_count; i += 2) {
        bitmap.Set(i, 1);
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kOneHot: {
      InlineBitmap bitmap(bit_count);
      int64_t index = RandRange(bit_count);
      bitmap.Set(index, 1);
      return Bits::FromBitmap(std::move(bitmap));
    }
    case kRandom: {
      InlineBitmap bitmap(bit_count);
      for (int64_t i = 0; i < bit_count; ++i) {
        bitmap.Set(i, RandomBool());
      }
      return Bits::FromBitmap(std::move(bitmap));
    }
  }
  XLS_LOG(FATAL) << "Impossible choice: " << choice;
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateOneHotSelectBuiltin(Env* env) {
  // We need to choose a selector with a certain number of bits, then form an
  // array from that many values in the environment.
  constexpr int64_t kMaxBitCount = 8;
  auto choose_value = [this](const TypedExpr& e) -> bool {
    TypeAnnotation* t = e.type;
    return IsUBits(t) && 0 <= GetTypeBitCount(t) &&
           GetTypeBitCount(t) <= kMaxBitCount;
  };

  std::optional<TypedExpr> lhs =
      ChooseEnvValueOptional(env, /*take=*/choose_value);
  if (!lhs.has_value()) {
    // If there's no natural environment value to use as the LHS, make up a
    // number and number of bits.
    int64_t bits = RandRange(1, kMaxBitCount);
    lhs = GenerateNumber(BitsAndSignedness{bits, false});
  }

  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueBits(env));
  std::vector<Expr*> cases = {rhs.expr};
  int64_t total_operands = GetTypeBitCount(lhs->type);
  for (int64_t i = 0; i < total_operands - 1; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValue(env, rhs.type));
    cases.push_back(e.expr);
  }

  XLS_RETURN_IF_ERROR(
      VerifyAggregateWidth(cases.size() * GetTypeBitCount(rhs.type)));

  auto* cases_array =
      module_->Make<Array>(fake_span_, cases, /*has_ellipsis=*/false);
  auto* invocation =
      module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("one_hot_sel"),
                                std::vector<Expr*>{lhs->expr, cases_array});
  return TypedExpr{invocation, rhs.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GeneratePrioritySelectBuiltin(
    Env* env) {
  XLS_ASSIGN_OR_RETURN(
      TypedExpr lhs,
      ChooseEnvValueBits(
          env, /*bit_count=*/RandomIntWithExpectedValue(5, /*lower_limit=*/1)));

  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueBits(env));
  std::vector<Expr*> cases = {rhs.expr};
  int64_t total_operands = GetTypeBitCount(lhs.type);
  for (int64_t i = 0; i < total_operands - 1; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValue(env, rhs.type));
    cases.push_back(e.expr);
  }

  XLS_RETURN_IF_ERROR(
      VerifyAggregateWidth(cases.size() * GetTypeBitCount(rhs.type)));

  auto* cases_array =
      module_->Make<Array>(fake_span_, cases, /*has_ellipsis=*/false);
  auto* invocation =
      module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("priority_sel"),
                                std::vector<Expr*>{lhs.expr, cases_array});
  return TypedExpr{invocation, rhs.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArrayConcat(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueArray(env));
  auto* lhs_array_type = dynamic_cast<ArrayTypeAnnotation*>(lhs.type);
  XLS_RET_CHECK(lhs_array_type != nullptr);

  auto array_compatible = [&](const TypedExpr& e) -> bool {
    TypeAnnotation* t = e.type;
    if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(t)) {
      return array->element_type() == lhs_array_type->element_type();
    }
    return false;
  };

  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValue(env, array_compatible));

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(GetTypeBitCount(lhs.type) +
                                           GetTypeBitCount(rhs.type)));

  auto* rhs_array_type = dynamic_cast<ArrayTypeAnnotation*>(rhs.type);
  XLS_RET_CHECK(rhs_array_type != nullptr);
  Binop* result =
      module_->Make<Binop>(fake_span_, BinopKind::kConcat, lhs.expr, rhs.expr);
  int64_t result_size =
      GetArraySize(lhs_array_type) + GetArraySize(rhs_array_type);
  Number* dim = MakeNumber(result_size);
  auto* result_type = module_->Make<ArrayTypeAnnotation>(
      fake_span_, lhs_array_type->element_type(), dim);
  return TypedExpr{result, result_type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArray(Env* env) {
  // Choose an arbitrary value from the environment, then gather all elements
  // from the environment of that type.
  XLS_ASSIGN_OR_RETURN(TypedExpr value, ChooseEnvValue(env));
  std::vector<TypedExpr> values = GatherAllValues(
      env, [&](const TypedExpr& t) { return t.type == value.type; });
  XLS_RET_CHECK(!values.empty());
  if (RandomBool()) {
    // Half the time extend the set of values by duplicating members. Walk
    // through the vector randomly duplicating members along the way. On average
    // this process will double the size of the array with the distribution
    // falling off exponentially.
    for (int64_t i = 0; i < values.size(); ++i) {
      if (RandomBool()) {
        int64_t idx = RandRange(values.size());
        values.push_back(values[idx]);
      }
    }
  }
  std::vector<Expr*> value_exprs;
  int64_t total_width = 0;
  value_exprs.reserve(values.size());
  for (TypedExpr t : values) {
    value_exprs.push_back(t.expr);
    total_width += GetTypeBitCount(t.type);
  }

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(total_width));

  // Create a type alias for the return type because arrays of tuples do not
  // parse. For example, the following is a parse error:
  //
  //  let x1: (u32, u16)[42] = ...
  //
  // Instead do:
  //
  //  type x2 = (u32, u16);
  //  ...
  //  let x1: (x2)[42] = ...
  //
  // TODO(https://github.com/google/xls/issues/326) 2021-03-05 Remove this alias
  // when parsing is fixed.
  auto* element_type_alias = MakeTypeRefTypeAnnotation(value.type);
  auto* result_type = module_->Make<ArrayTypeAnnotation>(
      fake_span_, element_type_alias, MakeNumber(values.size()));

  return TypedExpr{
      module_->Make<Array>(fake_span_, value_exprs, /*has_ellipsis=*/false),
      result_type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArrayIndex(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr array, ChooseEnvValueArray(env));
  ArrayTypeAnnotation* array_type = down_cast<ArrayTypeAnnotation*>(array.type);
  XLS_ASSIGN_OR_RETURN(TypedExpr index, ChooseEnvValueUBits(env));
  int64_t array_size = GetArraySize(array_type);
  // An out-of-bounds array index raises an error in the DSLX interpreter so
  // clamp the index so it is always in-bounds.
  // TODO(https://github.com/google/xls/issues/327) 2021-03-05 Unify OOB
  // behavior across different levels in XLS.
  if (GetTypeBitCount(index.type) >= Bits::MinBitCountUnsigned(array_size)) {
    int64_t index_bound = RandRange(array_size);
    XLS_ASSIGN_OR_RETURN(index.expr, GenerateUmin(index, index_bound));
  }
  return TypedExpr{module_->Make<Index>(fake_span_, array.expr, index.expr),
                   array_type->element_type()};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArrayUpdate(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr array, ChooseEnvValueArray(env));
  ArrayTypeAnnotation* array_type = down_cast<ArrayTypeAnnotation*>(array.type);
  XLS_ASSIGN_OR_RETURN(TypedExpr index, ChooseEnvValueUBits(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr element,
                       ChooseEnvValue(env, array_type->element_type()));
  int64_t array_size = GetArraySize(array_type);
  // An out-of-bounds array update raises an error in the DSLX interpreter so
  // clamp the index so it is always in-bounds.
  // TODO(https://github.com/google/xls/issues/327) 2021-03-05 Unify OOB
  // behavior across different levels in XLS.
  if (GetTypeBitCount(index.type) >= Bits::MinBitCountUnsigned(array_size)) {
    int64_t index_bound = RandRange(array_size);
    XLS_ASSIGN_OR_RETURN(index.expr, GenerateUmin(index, index_bound));
  }
  return TypedExpr{
      module_->Make<Invocation>(
          fake_span_, MakeBuiltinNameRef("update"),
          std::vector<Expr*>{array.expr, index.expr, element.expr}),
      array.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateGate(Env* env) {
  XLS_RET_CHECK(env != nullptr);
  XLS_ASSIGN_OR_RETURN(TypedExpr p, GenerateCompare(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr value, ChooseEnvValueBits(env));
  return TypedExpr{
      module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("gate!"),
                                std::vector<Expr*>{p.expr, value.expr}),
      value.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateConcat(Env* env) {
  XLS_RET_CHECK(env != nullptr);
  if (EnvContainsArray(*env) && RandomBool()) {
    return GenerateArrayConcat(env);
  }

  // Pick the number of operands of the concat. We need at least one value.
  int64_t count = GenerateNaryOperandCount(env, /*lower_limit=*/1);
  std::vector<TypedExpr> operands;
  int64_t total_width = 0;
  for (int64_t i = 0; i < count; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValueUBits(env));
    operands.push_back(e);
    total_width += GetTypeBitCount(e.type);
  }

  XLS_RETURN_IF_ERROR(VerifyBitsWidth(total_width));

  TypedExpr e = operands[0];
  Expr* result = e.expr;
  for (int64_t i = 1; i < count; ++i) {
    result = module_->Make<Binop>(fake_span_, BinopKind::kConcat, result,
                                  operands[i].expr);
  }

  TypeAnnotation* return_type = MakeTypeAnnotation(false, total_width);
  return TypedExpr{result, return_type};
}

BuiltinTypeAnnotation* AstGenerator::GeneratePrimitiveType() {
  int64_t integral = RandRange(
      std::min(kConcreteBuiltinTypeLimit, options_.max_width_bits_types + 1));
  auto type = static_cast<BuiltinType>(integral);
  return module_->Make<BuiltinTypeAnnotation>(fake_span_, type);
}

TypedExpr AstGenerator::GenerateNumber(std::optional<BitsAndSignedness> bas) {
  TypeAnnotation* type;
  if (bas.has_value()) {
    type = MakeTypeAnnotation(bas->signedness, bas->bits);
  } else {
    type = GeneratePrimitiveType();
  }
  int64_t bit_count = GetTypeBitCount(type);
  Bits value = ChooseBitPattern(bit_count);
  return TypedExpr{MakeNumberFromBits(value, type), type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateRetval(Env* env) {
  int64_t retval_count = GenerateNaryOperandCount(
      env, /*lower_limit=*/options_.generate_empty_tuples ? 0 : 1);

  std::vector<TypedExpr> env_params;
  std::vector<TypedExpr> env_non_params;
  for (auto& item : *env) {
    if (auto* name_ref = dynamic_cast<NameRef*>(item.second.expr);
        name_ref != nullptr && name_ref->DefinerIs<Param>()) {
      env_params.push_back(item.second);
    } else {
      env_non_params.push_back(item.second);
    }
  }

  XLS_RET_CHECK(!env_params.empty() || !env_non_params.empty());

  std::vector<TypedExpr> typed_exprs;
  int64_t total_bit_count = 0;
  for (int64_t i = 0; i < retval_count; ++i) {
    TypedExpr expr;
    float p = RandomFloat();
    if (env_non_params.empty() || (p < 0.1 && !env_params.empty())) {
      expr = RandomChoice<TypedExpr>(env_params);
    } else {
      expr = RandomChoice<TypedExpr>(env_non_params);
    }

    // See if the value we selected is going to push us over the "aggregate type
    // width" limit.
    if ((total_bit_count + GetTypeBitCount(expr.type) >
         options_.max_width_aggregate_types)) {
      if (options_.generate_empty_tuples) {
        // If it's ok to generate empty tuples, we just try again. Note we'll
        // end up with < retval_count elements by doing this, potentially 0.
        continue;
      }
      // Make sure we have at least one value, since
      // `!options._generate_empty_tuples`.
      if (typed_exprs.empty()) {
        typed_exprs.push_back(expr);
      }
      break;
    }

    typed_exprs.push_back(expr);
    total_bit_count += GetTypeBitCount(expr.type);
  }

  // If only a single return value is selected, most of the time just return it
  // as a non-tuple value.
  if (RandomFloat() < 0.8 && typed_exprs.size() == 1) {
    return typed_exprs[0];
  }

  auto [exprs, types] = Unzip(typed_exprs);
  if (!options_.generate_empty_tuples) {
    XLS_RET_CHECK(!exprs.empty())
        << "retval_count was: " << retval_count
        << " typed_exprs size: " << typed_exprs.size();
  }
  auto* tuple = module_->Make<XlsTuple>(fake_span_, exprs);
  return TypedExpr{tuple, MakeTupleType(types)};
}

// The return value for a proc's next function must be of the state type if a
// state is present.
absl::StatusOr<TypedExpr> AstGenerator::GenerateProcNextFunctionRetval(
    Env* env) {
  if (proc_properties_.state_types.empty()) {
    // Return an empty tuple.
    return TypedExpr{module_->Make<XlsTuple>(fake_span_, std::vector<Expr*>()),
                     MakeTupleType(std::vector<TypeAnnotation*>())};
  }
  // A state is present, therefore the return value for a proc's next function
  // must be of the state type.
  return ChooseEnvValue(env, proc_properties_.state_types[0]);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCountedFor(Env* env) {
  // Right now just generates the 'identity' for loop.
  // TODO(meheff): Generate more interesting loop bodies.
  TypeAnnotation* ivar_type = MakeTypeAnnotation(false, 4);
  Number* zero = MakeNumber(0, ivar_type);
  Number* trips = MakeNumber(RandRange(8) + 1, ivar_type);
  Expr* iterable = MakeRange(zero, trips);
  NameDef* x_def = MakeNameDef("x");
  NameDefTree* i_ndt = module_->Make<NameDefTree>(fake_span_, MakeNameDef("i"));
  NameDefTree* x_ndt = module_->Make<NameDefTree>(fake_span_, x_def);
  auto* name_def_tree = module_->Make<NameDefTree>(
      fake_span_, std::vector<NameDefTree*>{i_ndt, x_ndt});
  XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValueNotArray(env));
  NameRef* body = MakeNameRef(x_def);

  // Randomly decide to use or not-use the type annotation on the loop.
  TupleTypeAnnotation* tree_type = nullptr;
  if (RandomBool()) {
    tree_type = MakeTupleType({ivar_type, e.type});
  }
  Block* block = module_->Make<Block>(fake_span_, body);
  For* for_ = module_->Make<For>(fake_span_, name_def_tree, tree_type, iterable,
                                 block, /*init=*/e.expr);
  return TypedExpr{for_, e.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateTupleOrIndex(Env* env) {
  XLS_CHECK(env != nullptr);
  bool do_index = RandomBool() && EnvContainsTuple(*env);
  if (do_index) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValueTuple(env, /*min_size=*/1));
    auto* tuple_type = dynamic_cast<TupleTypeAnnotation*>(e.type);
    int64_t i = RandRange(tuple_type->size());
    Number* index_expr = MakeNumber(i);
    return TypedExpr{module_->Make<TupleIndex>(fake_span_, e.expr, index_expr),
                     tuple_type->members()[i]};
  }

  std::vector<TypedExpr> members;
  int64_t total_bit_count = 0;
  int64_t element_count = GenerateNaryOperandCount(
      env, /*lower_limit=*/options_.generate_empty_tuples ? 0 : 1);
  for (int64_t i = 0; i < element_count; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValue(env));
    members.push_back(e);
    total_bit_count += GetTypeBitCount(e.type);
  }

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(total_bit_count));

  auto [exprs, types] = Unzip(members);
  if (!options_.generate_empty_tuples) {
    XLS_RET_CHECK(!exprs.empty());
  }
  return TypedExpr{module_->Make<XlsTuple>(fake_span_, exprs),
                   MakeTupleType(types)};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateMap(int64_t call_depth,
                                                    Env* env) {
  std::string map_fn_name = GenSym();

  // GenerateFunction(), in turn, can call GenerateMap(), so we need some way of
  // bounding the recursion. To limit explosion, return an recoverable error
  // with exponentially increasing probability depending on the call depth.
  if (RandomFloat() > pow(10.0, -call_depth)) {
    return RecoverableError("Call depth too deep.");
  }

  // Choose a random array from the environment and create a single-argument
  // function which takes an element of that array.
  XLS_ASSIGN_OR_RETURN(TypedExpr array, ChooseEnvValueArray(env));
  ArrayTypeAnnotation* array_type = down_cast<ArrayTypeAnnotation*>(array.type);
  XLS_ASSIGN_OR_RETURN(Function * map_fn,
                       GenerateFunction(map_fn_name, call_depth + 1,
                                        /*param_types=*/
                                        std::vector<TypeAnnotation*>(
                                            {array_type->element_type()})));
  functions_.push_back(map_fn);

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(
      GetTypeBitCount(map_fn->return_type()) * GetArraySize(array_type)));

  TypeAnnotation* return_type =
      MakeArrayType(map_fn->return_type(), GetArraySize(array_type));

  NameRef* fn_ref = MakeNameRef(MakeNameDef(map_fn_name));
  auto* invocation =
      module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("map"),
                                std::vector<Expr*>{array.expr, fn_ref});
  return TypedExpr{invocation, return_type};
}

TypeAnnotation* AstGenerator::GenerateBitsType() {
  if (options_.max_width_bits_types <= 64 || RandRange(1, 10) != 1) {
    return GeneratePrimitiveType();
  }
  // Generate a type wider than 64-bits. With smallish probability choose a
  // *really* wide type if the max_width_bits_types supports it, otherwise
  // choose a width up to 128 bits.
  int64_t max_width = options_.max_width_bits_types;
  if (max_width > 128 && RandRange(1, 10) > 1) {
    max_width = 128;
  }
  bool sign = RandomBool();
  return MakeTypeAnnotation(sign, 64 + RandRange(1, max_width - 64));
}

TypeAnnotation* AstGenerator::GenerateType(int64_t nesting) {
  float r = RandomFloat();
  if (r < 0.1 * std::pow(2.0, -nesting)) {
    // Generate tuple type. Use a mean value of 3 elements so the tuple isn't
    // too big.
    int64_t total_width = 0;
    std::vector<TypeAnnotation*> element_types;
    for (int64_t i = 0;
         i < RandomIntWithExpectedValue(
                 3, /*lower_limit=*/options_.generate_empty_tuples ? 0 : 1);
         ++i) {
      TypeAnnotation* element_type = GenerateType(nesting + 1);
      int64_t element_width = GetTypeBitCount(element_type);
      if (total_width + element_width > options_.max_width_aggregate_types) {
        break;
      }
      element_types.push_back(element_type);
      total_width += element_width;
    }
    return MakeTupleType(element_types);
  }
  if (r < 0.2 * std::pow(2.0, -nesting)) {
    // Generate array type.
    TypeAnnotation* element_type = GenerateType(nesting + 1);
    int64_t element_width = GetTypeBitCount(element_type);
    int64_t element_count = RandomIntWithExpectedValue(10, /*lower_limit=*/1);
    if (element_count * element_width > options_.max_width_aggregate_types) {
      element_count = std::max<int64_t>(
          1, options_.max_width_aggregate_types / element_width);
    }
    return MakeArrayType(element_type, element_count);
  }
  return GenerateBitsType();
}

std::optional<TypedExpr> AstGenerator::ChooseEnvValueOptional(
    Env* env, std::function<bool(const TypedExpr&)> take) {
  if (take == nullptr) {
    // Fast path if there's no take function, we don't need to inspect/copy
    // things.
    int64_t index = RandRange(env->size());
    auto it = env->begin();
    std::advance(it, index);
    return it->second;
  }

  std::vector<TypedExpr*> choices;
  for (auto& item : *env) {
    if (take(item.second)) {
      choices.push_back(&item.second);
    }
  }
  if (choices.empty()) {
    return absl::nullopt;
  }
  int64_t index = RandRange(choices.size());
  return *choices[index];
}

absl::StatusOr<TypedExpr> AstGenerator::ChooseEnvValue(
    Env* env, std::function<bool(const TypedExpr&)> take) {
  auto result = ChooseEnvValueOptional(env, take);
  if (!result.has_value()) {
    return RecoverableError(
        "No elements in the environment satisfy the predicate.");
  }
  return result.value();
}

std::vector<TypedExpr> AstGenerator::GatherAllValues(
    Env* env, std::function<bool(const TypedExpr&)> take) {
  std::vector<TypedExpr> values;
  for (auto& item : *env) {
    if (take(item.second)) {
      values.push_back(item.second);
    }
  }
  return values;
}

absl::StatusOr<std::pair<TypedExpr, TypedExpr>>
AstGenerator::ChooseEnvValueBitsPair(Env* env,
                                     std::optional<int64_t> bit_count) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueBits(env, bit_count));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueBits(env, bit_count));
  if (lhs.type == rhs.type) {
    return std::pair{lhs, rhs};
  }
  if (RandomBool()) {
    rhs.expr = module_->Make<Cast>(fake_span_, rhs.expr, lhs.type);
    rhs.type = lhs.type;
  } else {
    lhs.expr = module_->Make<Cast>(fake_span_, lhs.expr, rhs.type);
    lhs.type = rhs.type;
  }
  return std::pair{lhs, rhs};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateUnop(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueBits(env));
  UnopKind op = RandomChoice<UnopKind>({UnopKind::kInvert, UnopKind::kNegate});
  return TypedExpr{module_->Make<Unop>(fake_span_, op, arg.expr), arg.type};
}

// Returns (start, width), resolving indices via DSLX bit slice semantics.
static std::pair<int64_t, int64_t> ResolveBitSliceIndices(
    int64_t bit_count, std::optional<int64_t> start,
    std::optional<int64_t> limit) {
  if (!start.has_value()) {
    start = 0;
  }
  if (!limit.has_value()) {
    limit = bit_count;
  }
  if (*start < 0) {
    start = *start + bit_count;
  }
  if (*limit < 0) {
    limit = *limit + bit_count;
  }
  limit = std::min(std::max(*limit, int64_t{0}), bit_count);
  start = std::min(std::max(*start, int64_t{0}), *limit);
  XLS_CHECK_GE(*start, 0);
  XLS_CHECK_GE(*limit, *start);
  return {*start, *limit - *start};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBitSlice(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueBits(env));
  int64_t bit_count = GetTypeBitCount(arg.type);
  // Slice LHS must be UBits.
  if (!IsUBits(arg.type)) {
    arg.expr = module_->Make<Cast>(fake_span_, arg.expr,
                                   MakeTypeAnnotation(false, bit_count));
  }
  enum class SliceType {
    kBitSlice,
    kWidthSlice,
    kDynamicSlice,
  };
  SliceType which = RandomChoice<SliceType>(
      {SliceType::kBitSlice, SliceType::kWidthSlice, SliceType::kDynamicSlice});
  std::optional<int64_t> start;
  std::optional<int64_t> limit;
  int64_t width = -1;
  while (true) {
    int64_t start_low = (which == SliceType::kWidthSlice) ? 0 : -bit_count - 1;
    bool should_have_start = RandomBool();
    start = should_have_start
                ? absl::make_optional(RandRange(start_low, bit_count + 1))
                : absl::nullopt;
    bool should_have_limit = RandomBool();
    limit = should_have_limit
                ? absl::make_optional(RandRange(-bit_count - 1, bit_count + 1))
                : absl::nullopt;
    width = ResolveBitSliceIndices(bit_count, start, limit).second;
    if (width > 0) {  // Make sure we produce non-zero-width things.
      break;
    }
  }
  XLS_RET_CHECK_GT(width, 0);

  IndexRhs rhs;
  switch (which) {
    case SliceType::kBitSlice: {
      Number* start_num = start.has_value() ? MakeNumber(*start) : nullptr;
      Number* limit_num = limit.has_value() ? MakeNumber(*limit) : nullptr;
      rhs = module_->Make<Slice>(fake_span_, start_num, limit_num);
      break;
    }
    case SliceType::kWidthSlice: {
      int64_t start_int = start.has_value() ? *start : 0;
      rhs = module_->Make<WidthSlice>(fake_span_, MakeNumber(start_int),
                                      MakeTypeAnnotation(false, width));
      break;
    }
    case SliceType::kDynamicSlice: {
      XLS_ASSIGN_OR_RETURN(TypedExpr start, ChooseEnvValueUBits(env));
      rhs = module_->Make<WidthSlice>(fake_span_, start.expr,
                                      MakeTypeAnnotation(false, width));
      break;
    }
  }
  TypeAnnotation* type = MakeTypeAnnotation(false, width);
  auto* expr = module_->Make<Index>(fake_span_, arg.expr, rhs);
  return TypedExpr{expr, type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBitwiseReduction(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(env));
  absl::string_view op = RandomChoice<absl::string_view>(
      {"and_reduce", "or_reduce", "xor_reduce"});
  NameRef* callee = MakeBuiltinNameRef(std::string(op));
  TypeAnnotation* type = MakeTypeAnnotation(false, 1);
  return TypedExpr{module_->Make<Invocation>(fake_span_, callee,
                                             std::vector<Expr*>{arg.expr}),
                   type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCastBitsToArray(Env* env) {
  // Get a random bits-typed element from the environment.
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(env));

  // Next, find factors of the bit count and select one pair.
  int64_t bit_count = GetTypeBitCount(arg.type);
  std::vector<std::pair<int64_t, int64_t>> factors;
  for (int64_t i = 1; i < bit_count + 1; ++i) {
    if (bit_count % i == 0) {
      factors.push_back({i, bit_count / i});
    }
  }

  auto [element_size, array_size] = RandomChoice(absl::MakeConstSpan(factors));
  TypeAnnotation* element_type = MakeTypeAnnotation(false, element_size);
  ArrayTypeAnnotation* outer_array_type =
      MakeArrayType(element_type, array_size);
  Cast* expr = module_->Make<Cast>(fake_span_, arg.expr, outer_array_type);
  return TypedExpr{expr, outer_array_type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBitSliceUpdate(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr start, ChooseEnvValueUBits(env));
  XLS_ASSIGN_OR_RETURN(TypedExpr update_value, ChooseEnvValueUBits(env));

  auto* invocation = module_->Make<Invocation>(
      fake_span_, MakeBuiltinNameRef("bit_slice_update"),
      std::vector<Expr*>{arg.expr, start.expr, update_value.expr});
  return TypedExpr{invocation, arg.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArraySlice(Env* env) {
  // JIT/codegen for array_slice don't currently support zero-sized types
  auto is_not_zst = [this](ArrayTypeAnnotation* array_type) -> bool {
    return this->GetTypeBitCount(array_type) != 0;
  };

  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueArray(env, is_not_zst));

  auto arg_type = dynamic_cast<ArrayTypeAnnotation*>(arg.type);
  XLS_CHECK_NE(arg_type, nullptr)
      << "Postcondition of ChooseEnvValueArray violated";

  XLS_ASSIGN_OR_RETURN(TypedExpr start, ChooseEnvValueUBits(env));

  int64_t slice_width;

  if (RandomBool()) {
    slice_width = RandomIntWithExpectedValue(1.0, /*lower_limit=*/1);
  } else {
    slice_width = RandomIntWithExpectedValue(10.0, /*lower_limit=*/1);
  }

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(
      slice_width * GetTypeBitCount(arg_type->element_type())));

  std::vector<Expr*> width_array_elements = {module_->Make<Index>(
      fake_span_, arg.expr, MakeNumber(0, MakeTypeAnnotation(false, 32)))};
  Array* width_expr = module_->Make<Array>(fake_span_, width_array_elements,
                                           /*has_ellipsis=*/true);
  TypeAnnotation* width_type = module_->Make<ArrayTypeAnnotation>(
      fake_span_, arg_type->element_type(), MakeNumber(slice_width));
  width_expr->set_type_annotation(width_type);

  TypedExpr width{width_expr, width_type};
  auto* invocation = module_->Make<Invocation>(
      fake_span_, MakeBuiltinNameRef("slice"),
      std::vector<Expr*>{arg.expr, start.expr, width.expr});
  return TypedExpr{invocation, width_type};
}

namespace {

enum OpChoice {
  kArray,
  kArrayIndex,
  kArrayUpdate,
  kArraySlice,
  kBinop,
  kBitSlice,
  kBitSliceUpdate,
  kBitwiseReduction,
  kCastToBitsArray,
  kChannelOp,
  kCompareOp,
  kCompareArrayOp,
  kCompareTupleOp,
  kConcat,
  kCountedFor,
  kGate,
  kJoinOp,
  kLogical,
  kMap,
  kNumber,
  kOneHotSelectBuiltin,
  kPartialProduct,
  kPrioritySelectBuiltin,
  kShiftOp,
  kTupleOrIndex,
  kUnop,
  kUnopBuiltin,

  // Sentinel denoting last element of enum.
  kEndSentinel
};

// Returns the relative probability of the given op being generated.
int OpProbability(OpChoice op) {
  switch (op) {
    case kArray:
      return 2;
    case kArrayIndex:
      return 2;
    case kArrayUpdate:
      return 2;
    case kArraySlice:
      return 2;
    case kBinop:
      return 10;
    case kBitSlice:
      return 10;
    case kBitSliceUpdate:
      return 2;
    case kBitwiseReduction:
      return 3;
    case kCastToBitsArray:
      return 1;
    case kChannelOp:
      return 5;
    case kCompareOp:
      return 3;
    case kCompareArrayOp:
      return 2;
    case kCompareTupleOp:
      return 2;
    case kConcat:
      return 5;
    case kCountedFor:
      return 1;
    case kGate:
      return 1;
    case kJoinOp:
      return 5;
    case kLogical:
      return 3;
    case kMap:
      return 1;
    case kNumber:
      return 3;
    case kOneHotSelectBuiltin:
      return 1;
    case kPartialProduct:
      return 1;
    case kPrioritySelectBuiltin:
      return 1;
    case kShiftOp:
      return 3;
    case kTupleOrIndex:
      return 3;
    case kUnop:
      return 10;
    case kUnopBuiltin:
      return 5;
    case kEndSentinel:
      return 0;
  }
  XLS_LOG(FATAL) << "Invalid op choice: " << static_cast<int64_t>(op);
}

std::discrete_distribution<int>& GetOpDistribution(bool generate_proc) {
  static std::discrete_distribution<int> dist = [generate_proc]() {
    static const std::set<int> proc_ops = {int{kChannelOp}, int{kJoinOp}};
    std::vector<int> tmp;
    tmp.reserve(int{kEndSentinel});
    for (int i = 0; i < int{kEndSentinel}; ++i) {
      // When not generating a proc, do not generate proc operations by setting
      // its probability to zero.
      if (!generate_proc && proc_ops.find(i) != proc_ops.end()) {
        tmp.push_back(0);
        continue;
      }
      tmp.push_back(OpProbability(static_cast<OpChoice>(i)));
    }
    return std::discrete_distribution<int>(tmp.begin(), tmp.end());
  }();
  return dist;
}

}  // namespace

absl::StatusOr<TypedExpr> AstGenerator::GenerateExpr(int64_t expr_size,
                                                     int64_t call_depth,
                                                     Env* env) {
  if (!ShouldNest(expr_size, call_depth)) {
    // Should not nest any more, select return values.
    if (options_.generate_proc) {
      return GenerateProcNextFunctionRetval(env);
    }
    return GenerateRetval(env);
  }

  TypedExpr rhs;
  while (true) {
    absl::StatusOr<TypedExpr> generated;

    int choice = GetOpDistribution(options_.generate_proc)(rng_);
    switch (static_cast<OpChoice>(choice)) {
      case kArray:
        generated = GenerateArray(env);
        break;
      case kArrayIndex:
        generated = GenerateArrayIndex(env);
        break;
      case kArrayUpdate:
        generated = GenerateArrayUpdate(env);
        break;
      case kArraySlice:
        generated = GenerateArraySlice(env);
        break;
      case kCountedFor:
        generated = GenerateCountedFor(env);
        break;
      case kTupleOrIndex:
        generated = GenerateTupleOrIndex(env);
        break;
      case kConcat:
        generated = GenerateConcat(env);
        break;
      case kBinop:
        generated = GenerateBinop(env);
        break;
      case kCompareOp:
        generated = GenerateCompare(env);
        break;
      case kCompareArrayOp:
        generated = GenerateCompareArray(env);
        break;
      case kCompareTupleOp:
        generated = GenerateCompareTuple(env);
        break;
      case kShiftOp:
        generated = GenerateShift(env);
        break;
      case kLogical:
        generated = GenerateLogicalOp(env);
        break;
      case kGate:
        if (!options_.emit_gate) {
          continue;
        }
        generated = GenerateGate(env);
        break;
      case kChannelOp:
        generated = GenerateChannelOp(env);
        break;
      case kJoinOp:
        generated = GenerateJoinOp(env);
        break;
      case kMap:
        // TODO(vmirian): 8-22-2022 For initial support of procs, only support a
        // single level calling stack.
        if (options_.generate_proc) {
          continue;
        }
        generated = GenerateMap(call_depth, env);
        break;
      case kUnop:
        generated = GenerateUnop(env);
        break;
      case kUnopBuiltin:
        generated = GenerateUnopBuiltin(env);
        break;
      case kOneHotSelectBuiltin:
        generated = GenerateOneHotSelectBuiltin(env);
        break;
      case kPartialProduct:
        generated = GeneratePartialProductDeterministicGroup(env);
        break;
      case kPrioritySelectBuiltin:
        generated = GeneratePrioritySelectBuiltin(env);
        break;
      case kNumber:
        generated = GenerateNumber();
        break;
      case kBitwiseReduction:
        generated = GenerateBitwiseReduction(env);
        break;
      case kBitSlice:
        generated = GenerateBitSlice(env);
        break;
      case kCastToBitsArray:
        generated = GenerateCastBitsToArray(env);
        break;
      case kBitSliceUpdate:
        generated = GenerateBitSliceUpdate(env);
        break;
      case kEndSentinel:
        XLS_LOG(FATAL) << "Should not have selected end sentinel";
    }

    if (generated.ok()) {
      rhs = generated.value();
      if (IsBits(rhs.type)) {
        XLS_RET_CHECK_LE(GetTypeBitCount(rhs.type),
                         options_.max_width_bits_types)
            << absl::StreamFormat("Bits-typed expression is too wide: %s",
                                  rhs.expr->ToString());
      } else if (IsArray(rhs.type) || IsTuple(rhs.type)) {
        XLS_RET_CHECK_LE(GetTypeBitCount(rhs.type),
                         options_.max_width_aggregate_types)
            << absl::StreamFormat("Aggregate-typed expression is too wide: %s",
                                  rhs.expr->ToString());
      }
      break;
    }

    // We expect the Generate* routines might try to sample things that don't
    // exist in the envs, so we keep going if we see one of those errors.
    if (IsRecoverableError(generated.status())) {
      continue;
    }

    // Any other error is unexpected, though.
    return generated.status();
  }
  std::string identifier = GenSym();

  // What we place into the environment is a NameRef that refers to this RHS
  // value -- this way rules will pick up the expression names instead of
  // picking up the expression ASTs directly (which would cause duplication).
  auto* name_def =
      module_->Make<NameDef>(fake_span_, identifier, /*definer=*/nullptr);
  auto* name_ref = MakeNameRef(name_def);
  (*env)[identifier] = TypedExpr{name_ref, rhs.type};

  // Unpack result tuples from channel operations and place them in environment
  // to be easily accessible creating more interesting behavior.
  // Currently, results from operations that are tuples and start with a token
  // are assumed to be channel operations.
  std::vector<std::pair<NameDef*, TypedExpr>> channel_tuples;
  if (IsTuple(rhs.type)) {
    auto* tuple_type = dynamic_cast<TupleTypeAnnotation*>(rhs.type);
    if (!tuple_type->empty() && IsToken(tuple_type->members()[0])) {
      channel_tuples.resize(tuple_type->members().size());
      for (int64_t index = 0; index < tuple_type->members().size(); ++index) {
        std::string member_identifier = GenSym();
        auto* member_name_def = module_->Make<NameDef>(
            fake_span_, member_identifier, /*definer=*/nullptr);
        auto* member_name_ref = MakeNameRef(member_name_def);
        (*env)[member_identifier] =
            TypedExpr{member_name_ref, tuple_type->members()[index]};
        // Insert in reverse order so the identifier are consecutive when
        // displayed on the output.
        channel_tuples[tuple_type->members().size() - index - 1] =
            std::pair<NameDef*, TypedExpr>{
                member_name_def,
                TypedExpr{module_->Make<TupleIndex>(fake_span_, name_ref,
                                                    MakeNumber(index)),
                          tuple_type->members()[index]}};
      }
    }
  }

  XLS_ASSIGN_OR_RETURN(TypedExpr body,
                       GenerateExpr(expr_size + 1, call_depth, env));

  auto* let = body.expr;
  for (const auto& channel_tuple : channel_tuples) {
    auto* ndt = module_->Make<NameDefTree>(fake_span_, channel_tuple.first);
    let = module_->Make<Let>(fake_span_, /*name_def_tree=*/ndt,
                             /*type=*/channel_tuple.second.type,
                             /*rhs=*/channel_tuple.second.expr,
                             /*body=*/let, /*is_const=*/false);
  }

  auto* ndt = module_->Make<NameDefTree>(fake_span_, name_def);
  let = module_->Make<Let>(fake_span_, /*name_def_tree=*/ndt,
                           /*type=*/rhs.type, /*rhs=*/rhs.expr,
                           /*body=*/let, /*is_const=*/false);
  return TypedExpr{let, body.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateUnopBuiltin(Env* env) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(env));
  enum UnopBuiltin {
    kClz,
    kCtz,
    kRev,
    kOneHot,
  };
  auto to_string = [](UnopBuiltin kind) -> std::string {
    switch (kind) {
      case kClz:
        return "clz";
      case kCtz:
        return "ctz";
      case kRev:
        return "rev";
      case kOneHot:
        return "one_hot";
    }
    XLS_LOG(FATAL) << "Invalid kind: " << kind;
  };

  std::vector<UnopBuiltin> choices = {kClz, kCtz, kRev};
  // Since one_hot adds a bit, only use it when we have head room beneath
  // max_width_bits_types to add another bit.
  if (GetTypeBitCount(arg.type) < options_.max_width_bits_types) {
    choices.push_back(kOneHot);
  }

  Invocation* invocation = nullptr;
  auto which = RandomChoice<UnopBuiltin>(choices);
  NameRef* name_ref = MakeBuiltinNameRef(to_string(which));
  int64_t result_bits = -1;
  switch (which) {
    case kClz:
    case kCtz:
    case kRev:
      invocation = module_->Make<Invocation>(fake_span_, name_ref,
                                             std::vector<Expr*>{arg.expr});
      result_bits = GetTypeBitCount(arg.type);
      break;
    case kOneHot: {
      bool lsb_or_msb = RandomBool();
      invocation = module_->Make<Invocation>(
          fake_span_, name_ref,
          std::vector<Expr*>{arg.expr, MakeBool(lsb_or_msb)});
      result_bits = GetTypeBitCount(arg.type) + 1;
      break;
    }
  }

  TypeAnnotation* result_type = MakeTypeAnnotation(false, result_bits);
  return TypedExpr{invocation, result_type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBody(
    int64_t call_depth, absl::Span<Param* const> params, Env* env) {
  // We need to be able to draw references from the environment, so we never
  // want it to be empty.
  XLS_RET_CHECK(!env->empty());
  return GenerateExpr(/*expr_size=*/0, call_depth, env);
}

// TODO(vmirian): 8-22-2022 Track caller of this function for different control
// paths (e.g. Function or Proc). Perhaps, use stack implementation for state
// transitions.
absl::StatusOr<Function*> AstGenerator::GenerateFunction(
    std::string name, int64_t call_depth,
    std::optional<absl::Span<TypeAnnotation* const>> param_types) {
  Env env;

  std::vector<ParametricBinding*> parametric_bindings;
  std::vector<Param*> params;
  if (param_types.has_value()) {
    for (TypeAnnotation* param_type : param_types.value()) {
      params.push_back(GenerateParam(param_type));
    }
  } else {
    // If we're the main function we have to have at least one parameter,
    // because some Generate* methods expect to be able to draw from a non-empty
    // env.
    //
    // TODO(https://github.com/google/xls/issues/475): Cleanup to make
    // productions that are ok with empty env separate from those which require
    // a populated env.
    //
    // When we're a nested call, 90% of the time make sure we have at least one
    // parameter.  Sometimes it's ok to try out what happens with 0 parameters.
    //
    // (Note we still pick a number of params with an expected value of 4 even
    // when 0 is permitted.)
    int64_t lower_limit = (call_depth == 0 || RandomFloat() >= 0.10) ? 1 : 0;
    params = GenerateParams(RandomIntWithExpectedValue(4, lower_limit));
  }

  // When we're not the main function, 10% of the time put some parametrics on
  // the function.
  if (call_depth != 0 && RandomFloat() >= 0.90) {
    parametric_bindings = GenerateParametricBindings(
        RandomIntWithExpectedValue(2, /*lower_limit=*/1));
  }

  for (Param* param : params) {
    env[param->identifier()] =
        TypedExpr{MakeNameRef(param->name_def()), param->type_annotation()};
  }
  for (ParametricBinding* pb : parametric_bindings) {
    env[pb->identifier()] =
        TypedExpr{MakeNameRef(pb->name_def()), pb->type_annotation()};
  }

  XLS_ASSIGN_OR_RETURN(TypedExpr retval,
                       GenerateBody(call_depth, params, &env));
  if (!options_.generate_empty_tuples) {
    XLS_RET_CHECK(!IsNil(retval.type));
  }
  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Block* block = module_->Make<Block>(fake_span_, retval.expr);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/parametric_bindings,
      /*params=*/params,
      /*return_type=*/retval.type, block, Function::Tag::kNormal,
      /*is_public=*/false);
  name_def->set_definer(f);
  return f;
}

absl::Status AstGenerator::GenerateFunctionInModule(std::string name) {
  XLS_ASSIGN_OR_RETURN(Function * f, GenerateFunction(name));
  for (auto& item : constants_) {
    XLS_RETURN_IF_ERROR(module_->AddTop(item.second));
  }
  for (auto& item : type_defs_) {
    XLS_RETURN_IF_ERROR(module_->AddTop(item));
  }
  for (auto& item : functions_) {
    XLS_RETURN_IF_ERROR(module_->AddTop(item));
  }
  XLS_RETURN_IF_ERROR(module_->AddTop(f));
  return absl::OkStatus();
}

absl::StatusOr<Function*> AstGenerator::GenerateProcConfigFunction(
    std::string name, absl::Span<Param* const> proc_params) {
  std::vector<Param*> params;
  std::vector<Expr*> tuple_members;
  std::vector<TypeAnnotation*> tuple_member_types;
  for (Param* proc_param : proc_params) {
    params.push_back(GenerateParam(proc_param->type_annotation()));
    tuple_members.push_back(MakeNameRef(proc_param->name_def()));
    tuple_member_types.push_back(proc_param->type_annotation());
  }
  TupleTypeAnnotation* ret_tuple_type =
      module_->Make<TupleTypeAnnotation>(fake_span_, tuple_member_types);
  XlsTuple* ret_tuple = module_->Make<XlsTuple>(fake_span_, tuple_members);
  Block* block = module_->Make<Block>(fake_span_, ret_tuple);
  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/params,
      /*return_type=*/ret_tuple_type, block, Function::Tag::kProcConfig,
      /*is_public=*/false);
  name_def->set_definer(f);
  return f;
}

absl::StatusOr<Function*> AstGenerator::GenerateProcNextFunction(
    Env* env, std::string name) {
  // A token is required as the first parameter of the next function.
  NameDef* token_name_def = module_->Make<NameDef>(fake_span_, GenSym(),
                                                   /*definer=*/nullptr);
  Param* token_param = module_->Make<Param>(token_name_def, MakeTokenType());
  std::vector<Param*> params = {token_param};

  // 70% of the time: add a state to the param list.
  if (RandomFloat() >= 0.30) {
    params.insert(params.end(), GenerateParam());
    proc_properties_.state_types.push_back(params.back()->type_annotation());
  }

  for (Param* param : params) {
    (*env)[param->identifier()] =
        TypedExpr{MakeNameRef(param->name_def()), param->type_annotation()};
  }

  XLS_ASSIGN_OR_RETURN(TypedExpr retval, GenerateBody(0, params, env));

  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Block* block = module_->Make<Block>(fake_span_, retval.expr);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/params,
      /*return_type=*/retval.type, block, Function::Tag::kNormal,
      /*is_public=*/false);
  name_def->set_definer(f);
  return f;
}

absl::StatusOr<Proc*> AstGenerator::GenerateProc(std::string name) {
  Env env;

  XLS_ASSIGN_OR_RETURN(Function * next_function,
                       GenerateProcNextFunction(&env, "next"));

  XLS_ASSIGN_OR_RETURN(
      Function * config_function,
      GenerateProcConfigFunction("config", proc_properties_.params));
  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Proc* proc = module_->Make<Proc>(
      fake_span_, name_def, /*config_name_def=*/config_function->name_def(),
      /*next_name_def=*/next_function->name_def(),
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*members=*/proc_properties_.params,
      /*config=*/config_function, /*next=*/next_function,
      /*is_public=*/false);
  name_def->set_definer(proc);
  return proc;
}

absl::Status AstGenerator::GenerateProcInModule(std::string proc_name) {
  XLS_ASSIGN_OR_RETURN(Proc * proc, GenerateProc(proc_name));
  for (auto& item : constants_) {
    XLS_RETURN_IF_ERROR(module_->AddTop(item.second));
  }
  for (auto& item : type_defs_) {
    XLS_RETURN_IF_ERROR(module_->AddTop(item));
  }
  for (auto& item : functions_) {
    XLS_RETURN_IF_ERROR(module_->AddTop(item));
  }
  XLS_RETURN_IF_ERROR(module_->AddTop(proc));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Module>> AstGenerator::Generate(
    std::string top_entity_name, std::string module_name) {
  module_ = std::make_unique<Module>(module_name);
  if (options_.generate_proc) {
    XLS_RETURN_IF_ERROR(GenerateProcInModule(top_entity_name));
  } else {
    XLS_RETURN_IF_ERROR(GenerateFunctionInModule(top_entity_name));
  }
  return std::move(module_);
}

AstGenerator::AstGenerator(AstGeneratorOptions options, std::mt19937* rng)
    : rng_(*XLS_DIE_IF_NULL(rng)),
      options_(options),
      fake_pos_("<fake>", 0, 0),
      fake_span_(fake_pos_, fake_pos_),
      binops_(options.binop_allowlist.has_value()
                  ? options.binop_allowlist.value()
                  : GetBinopSameTypeKinds()) {
  binops_.erase(BinopKind::kDiv);
}

}  // namespace xls::dslx
