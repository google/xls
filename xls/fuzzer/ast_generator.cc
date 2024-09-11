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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/discrete_distribution.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/math_util.h"
#include "xls/common/random_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/symbolized_stacktrace.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/interp_value.h"
#include "xls/fuzzer/ast_generator_options.pb.h"
#include "xls/fuzzer/value_generator.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

namespace {

// Given a collection of delaying operations for the dependencies of a value,
// determine the effective delaying operation we need to propagate along with
// that value.
//
// We return the strongest delaying operation according to the order:
//   Send > Recv > None
LastDelayingOp ComposeDelayingOps(
    absl::Span<const LastDelayingOp> delaying_ops) {
  absl::flat_hash_set<LastDelayingOp> delaying_ops_set(delaying_ops.begin(),
                                                       delaying_ops.end());
  if (delaying_ops_set.contains(LastDelayingOp::kSend)) {
    return LastDelayingOp::kSend;
  }
  if (delaying_ops_set.contains(LastDelayingOp::kRecv)) {
    return LastDelayingOp::kRecv;
  }
  return LastDelayingOp::kNone;
}

LastDelayingOp ComposeDelayingOps(
    std::initializer_list<LastDelayingOp> delaying_ops) {
  return ComposeDelayingOps(
      absl::MakeConstSpan(delaying_ops.begin(), delaying_ops.end()));
}

LastDelayingOp ComposeDelayingOps(LastDelayingOp op1, LastDelayingOp op2) {
  return ComposeDelayingOps({op1, op2});
}

LastDelayingOp ComposeDelayingOps(LastDelayingOp op1, LastDelayingOp op2,
                                  LastDelayingOp op3) {
  return ComposeDelayingOps({op1, op2, op3});
}

}  // namespace

/* static */ absl::StatusOr<AstGeneratorOptions> AstGeneratorOptions::FromProto(
    const AstGeneratorOptionsProto& proto) {
  return AstGeneratorOptions{
      .emit_signed_types = proto.emit_signed_types(),
      .max_width_bits_types = proto.max_width_bits_types(),
      .max_width_aggregate_types = proto.max_width_aggregate_types(),
      .emit_loops = proto.emit_loops(),
      .emit_gate = proto.emit_gate(),
      .generate_proc = proto.generate_proc(),
      .emit_stateless_proc = proto.emit_stateless_proc(),
      .emit_zero_width_bits_types = proto.emit_zero_width_bits_types(),
  };
}

AstGeneratorOptionsProto AstGeneratorOptions::ToProto() const {
  AstGeneratorOptionsProto proto;
  proto.set_emit_signed_types(emit_signed_types);
  proto.set_max_width_bits_types(max_width_bits_types);
  proto.set_max_width_aggregate_types(max_width_aggregate_types);
  proto.set_emit_loops(emit_loops);
  proto.set_emit_gate(emit_gate);
  proto.set_generate_proc(generate_proc);
  proto.set_emit_stateless_proc(emit_stateless_proc);
  proto.set_emit_zero_width_bits_types(emit_zero_width_bits_types);
  return proto;
}

bool AbslParseFlag(std::string_view text,
                   AstGeneratorOptions* ast_generator_options,
                   std::string* error) {
  std::string unescaped_text;
  if (!absl::Base64Unescape(text, &unescaped_text)) {
    *error =
        "Could not parse as an AstGeneratorOptions; not a Base64 encoded "
        "string?";
    return false;
  }
  AstGeneratorOptionsProto proto;
  if (!proto.ParseFromString(unescaped_text)) {
    *error =
        "Could not parse as an AstGeneratorOptions; not a serialized "
        "AstGeneratorOptionsProto?";
    return false;
  }
  absl::StatusOr<AstGeneratorOptions> result =
      AstGeneratorOptions::FromProto(proto);
  if (!result.ok()) {
    *error = result.status().ToString();
    return false;
  }
  *ast_generator_options = *std::move(result);
  return true;
}

std::string AbslUnparseFlag(const AstGeneratorOptions& ast_generator_options) {
  return absl::Base64Escape(
      ast_generator_options.ToProto().SerializeAsString());
}

/* static */ std::tuple<std::vector<Expr*>, std::vector<TypeAnnotation*>,
                        std::vector<LastDelayingOp>>
AstGenerator::Unzip(absl::Span<const TypedExpr> typed_exprs) {
  std::vector<Expr*> exprs;
  std::vector<TypeAnnotation*> types;
  std::vector<LastDelayingOp> delaying_ops;
  for (auto& typed_expr : typed_exprs) {
    exprs.push_back(typed_expr.expr);
    types.push_back(typed_expr.type);
    delaying_ops.push_back(typed_expr.last_delaying_op);
  }
  return std::make_tuple(std::move(exprs), std::move(types),
                         std::move(delaying_ops));
}

/* static */ bool AstGenerator::IsUBits(const TypeAnnotation* t) {
  std::optional<BitVectorMetadata> metadata = ExtractBitVectorMetadata(t);
  return metadata.has_value() && !metadata->is_signed;
}

/* static */ absl::StatusOr<bool> AstGenerator::BitsTypeIsSigned(
    const TypeAnnotation* type) {
  std::optional<BitVectorMetadata> metadata = ExtractBitVectorMetadata(type);
  if (metadata.has_value()) {
    return metadata->is_signed;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Type annotation %s is not a bitvector type", type->ToString()));
}

std::string AstGenerator::GenSym() {
  std::string result = absl::StrCat("x", next_name_index_++);
  VLOG(10) << "generated fresh symbol: " << result << " @ "
           << GetSymbolizedStackTraceAsString();
  return result;
}

absl::StatusOr<int64_t> AstGenerator::BitsTypeGetBitCount(
    TypeAnnotation* type) {
  std::optional<BitVectorMetadata> metadata = ExtractBitVectorMetadata(type);
  if (!metadata.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Type annotation %s is not a bitvector type", type->ToString()));
  }
  if (std::holds_alternative<int64_t>(metadata->bit_count)) {
    return std::get<int64_t>(metadata->bit_count);
  }
  // Implementation note: this method is not static because we want to reuse the
  // GetExprAsUint64() helper, which looks into the constants_ mapping. We could
  // make both of these methods static by leaning harder on AST inspection, but
  // this gives us a shortcut and we expect to typically have an AstGenerator
  // instance in hand for fuzz generation and testing.
  return GetExprAsUint64(std::get<Expr*>(metadata->bit_count));
}

/* static */ bool AstGenerator::IsTypeRef(const TypeAnnotation* t) {
  return dynamic_cast<const TypeRefTypeAnnotation*>(t) != nullptr;
}

/* static */ bool AstGenerator::IsBits(const TypeAnnotation* t) {
  return ExtractBitVectorMetadata(t).has_value();
}

/* static */ bool AstGenerator::IsArray(const TypeAnnotation* t) {
  if (dynamic_cast<const ArrayTypeAnnotation*>(t) != nullptr) {
    return !IsBits(t);
  }
  return false;
}

/* static */ bool AstGenerator::IsTuple(const TypeAnnotation* t) {
  return dynamic_cast<const TupleTypeAnnotation*>(t) != nullptr;
}

/* static */ bool AstGenerator::IsToken(const TypeAnnotation* t) {
  auto* token = dynamic_cast<const BuiltinTypeAnnotation*>(t);
  return token != nullptr && token->builtin_type() == BuiltinType::kToken;
}

/* static */ bool AstGenerator::IsChannel(const TypeAnnotation* t) {
  return dynamic_cast<const ChannelTypeAnnotation*>(t) != nullptr;
}

/* static */ bool AstGenerator::IsNil(const TypeAnnotation* t) {
  if (auto* tuple = dynamic_cast<const TupleTypeAnnotation*>(t);
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

AnnotatedParam AstGenerator::GenerateParam(AnnotatedType type) {
  std::string identifier = GenSym();
  CHECK_NE(type.type, nullptr);
  NameDef* name_def = module_->Make<NameDef>(fake_span_, std::move(identifier),
                                             /*definer=*/nullptr);
  Param* param = module_->Make<Param>(name_def, type.type);
  name_def->set_definer(param);
  return {.param = param,
          .last_delaying_op = type.last_delaying_op,
          .min_stage = type.min_stage};
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
    // arbitrary GenerateNumberWithType() calls.
    //
    // TODO(google/xls#461): We only support 64-bit values being mangled into
    // identifier since Bits conversion to decimal only supports that.
    //
    // Starting from 1 is to ensure that we don't get a 0-bit value.
    int64_t bit_count = absl::Uniform<int64_t>(
        absl::IntervalClosed, bit_gen_, 1,
        std::min(int64_t{65}, options_.max_width_bits_types));
    TypedExpr number =
        GenerateNumberWithType(BitsAndSignedness{bit_count, false});
    ParametricBinding* pb =
        module_->Make<ParametricBinding>(name_def, number.type, number.expr);
    name_def->set_definer(pb);
    pbs.push_back(pb);
  }
  return pbs;
}

BuiltinTypeAnnotation* AstGenerator::MakeTokenType() {
  return module_->Make<BuiltinTypeAnnotation>(
      fake_span_, BuiltinType::kToken,
      module_->GetOrCreateBuiltinNameDef(BuiltinType::kToken));
}

TypeAnnotation* AstGenerator::MakeTypeAnnotation(bool is_signed,
                                                 int64_t width) {
  CHECK_GE(width, 0);
  if (width > 0 && width <= 64) {
    BuiltinType type = GetBuiltinType(is_signed, width).value();
    return module_->Make<BuiltinTypeAnnotation>(
        fake_span_, type, module_->GetOrCreateBuiltinNameDef(type));
  }
  BuiltinType builtin_type = is_signed ? BuiltinType::kSN : BuiltinType::kUN;
  auto* element_type = module_->Make<BuiltinTypeAnnotation>(
      fake_span_, builtin_type,
      module_->GetOrCreateBuiltinNameDef(builtin_type));
  Number* dim = MakeNumber(width);
  return module_->Make<ArrayTypeAnnotation>(fake_span_, element_type, dim);
}

absl::StatusOr<Expr*> AstGenerator::GenerateUmin(TypedExpr arg, int64_t other) {
  Number* rhs = GenerateNumber(other, arg.type);
  Expr* test = MakeGe(arg.expr, rhs);
  return MakeSel(test, rhs, arg.expr);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCompare(Context* ctx) {
  BinopKind op = RandomChoice(GetBinopComparisonKinds(), bit_gen_);
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(&ctx->env));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;
  Binop* binop = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr);
  return TypedExpr{.expr = binop,
                   .type = MakeTypeAnnotation(false, 1),
                   .last_delaying_op = ComposeDelayingOps(lhs.last_delaying_op,
                                                          rhs.last_delaying_op),
                   .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

namespace {

enum class ChannelOpType : std::uint8_t {
  kRecv,
  kRecvNonBlocking,
  kRecvIf,
  kSend,
  kSendIf,
};

struct ChannelOpInfo {
  ChannelDirection channel_direction;
  bool requires_payload;
  bool requires_predicate;
  bool requires_default_value;
};

ChannelOpInfo GetChannelOpInfo(ChannelOpType chan_op) {
  switch (chan_op) {
    case ChannelOpType::kRecv:
      return ChannelOpInfo{.channel_direction = ChannelDirection::kIn,
                           .requires_payload = false,
                           .requires_predicate = false,
                           .requires_default_value = false};
    case ChannelOpType::kRecvNonBlocking:
      return ChannelOpInfo{.channel_direction = ChannelDirection::kIn,
                           .requires_payload = false,
                           .requires_predicate = false,
                           .requires_default_value = true};
    case ChannelOpType::kRecvIf:
      return ChannelOpInfo{.channel_direction = ChannelDirection::kIn,
                           .requires_payload = false,
                           .requires_predicate = true,
                           .requires_default_value = true};
    case ChannelOpType::kSend:
      return ChannelOpInfo{.channel_direction = ChannelDirection::kOut,
                           .requires_payload = true,
                           .requires_predicate = false,
                           .requires_default_value = false};
    case ChannelOpType::kSendIf:
      return ChannelOpInfo{.channel_direction = ChannelDirection::kOut,
                           .requires_payload = true,
                           .requires_predicate = true,
                           .requires_default_value = false};
  }

  LOG(FATAL) << "Invalid ChannelOpType: " << static_cast<int>(chan_op);
}

}  // namespace

absl::StatusOr<TypedExpr> AstGenerator::GenerateChannelOp(Context* ctx) {
  // Equal distribution for channel ops.
  ChannelOpType chan_op_type =
      RandomChoice(absl::MakeConstSpan(
                       {ChannelOpType::kRecv, ChannelOpType::kRecvNonBlocking,
                        ChannelOpType::kRecvIf, ChannelOpType::kSend,
                        ChannelOpType::kSendIf}),
                   bit_gen_);
  ChannelOpInfo chan_op_info = GetChannelOpInfo(chan_op_type);

  int64_t min_stage = 1;

  // If needed, generate a predicate.
  std::optional<TypedExpr> predicate;
  if (chan_op_info.requires_predicate) {
    auto choose_predicate = [this](const TypedExpr& e) -> bool {
      TypeAnnotation* t = e.type;
      return IsUBits(t) && GetTypeBitCount(t) == 1;
    };
    predicate = ChooseEnvValueOptional(&ctx->env, /*take=*/choose_predicate);
    if (!predicate.has_value()) {
      // If there's no natural environment value to use as the predicate,
      // generate a boolean.
      Number* boolean = MakeBool(RandomBool(0.5));
      predicate = TypedExpr{boolean, boolean->type_annotation()};
    }

    int64_t successor_min_stage = predicate->min_stage;
    if (predicate->last_delaying_op == LastDelayingOp::kSend &&
        chan_op_info.channel_direction == ChannelDirection::kIn) {
      // Send depended on by recv - make sure we have a delay.
      successor_min_stage++;
    }
    min_stage = std::max(min_stage, successor_min_stage);
  }

  // The recv_non_blocking has an implicit bool in its return type, resulting in
  // one bit. Therefore, its maximum width must be one less than the maximum
  // width defined in the AST generator options.
  std::optional<int64_t> max_width_bits_types;
  std::optional<int64_t> max_width_aggregate_types;
  if (chan_op_type == ChannelOpType::kRecvNonBlocking) {
    // The recv_non_blocking returns an aggregate type that may contain a bits
    // type. If the max_width_bits_types > max_width_aggregate_types, it would
    // fail the aggregate width bounds check. Therefore, the bits type is
    // bounded within aggregate types maximum width.
    max_width_bits_types = std::min(options_.max_width_bits_types,
                                    options_.max_width_aggregate_types - 1);
    max_width_aggregate_types = options_.max_width_aggregate_types - 1;
  }
  // Generate an arbitrary type for the channel.
  TypeAnnotation* channel_type =
      GenerateType(0, max_width_bits_types, max_width_aggregate_types);

  // If needed, generate a default value.
  std::optional<TypedExpr> default_value;
  if (chan_op_info.requires_default_value) {
    // TODO(meheff): 2023/03/10 Use ChooseEnvValueOptional with randomly
    // generated constant as backup. Will require the ability to generate a
    // random constant of arbitrary type.
    XLS_ASSIGN_OR_RETURN(default_value,
                         ChooseEnvValue(&ctx->env, channel_type));

    int64_t successor_min_stage = default_value->min_stage;
    if (default_value->last_delaying_op == LastDelayingOp::kSend &&
        chan_op_info.channel_direction == ChannelDirection::kIn) {
      // Send depended on by recv - make sure we have a delay.
      successor_min_stage++;
    }
    min_stage = std::max(min_stage, successor_min_stage);
  }

  // If needed, choose a payload from the environment.
  std::optional<TypedExpr> payload;
  if (chan_op_info.requires_payload) {
    // TODO(vmirian): 8-22-2002 Payloads of the type may not be present in the
    // env. Create payload of the type enabling more ops requiring a payload
    // (e.g. send and send_if).
    XLS_ASSIGN_OR_RETURN(payload, ChooseEnvValue(&ctx->env, channel_type));

    int64_t successor_min_stage = payload->min_stage;
    if (payload->last_delaying_op == LastDelayingOp::kSend &&
        chan_op_info.channel_direction == ChannelDirection::kIn) {
      // Send depended on by recv - make sure we have a delay.
      successor_min_stage++;
    }
    min_stage = std::max(min_stage, successor_min_stage);
  }

  // Create the channel.
  // TODO(vmirian): 8-22-2022 If payload type exists, create an array of
  // channels.
  ChannelTypeAnnotation* channel_type_annotation =
      module_->Make<ChannelTypeAnnotation>(fake_span_,
                                           chan_op_info.channel_direction,
                                           channel_type, std::nullopt);
  Param* param = GenerateParam({.type = channel_type_annotation}).param;
  auto to_member = [this](const Param* p) -> absl::StatusOr<ProcMember*> {
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, CloneNode(p->name_def()));
    XLS_ASSIGN_OR_RETURN(
        AstNode * type_annotation,
        CloneAst(p->type_annotation(), &PreserveTypeDefinitionsReplacer));
    return module_->Make<ProcMember>(
        name_def, down_cast<TypeAnnotation*>(type_annotation));
  };
  XLS_ASSIGN_OR_RETURN(ProcMember * member, to_member(param));
  proc_properties_.members.push_back(member);
  proc_properties_.config_params.push_back(param);
  NameRef* chan_expr = module_->Make<NameRef>(fake_span_, param->identifier(),
                                              param->name_def());

  Expr* token_ref = nullptr;
  if (EnvContainsToken(ctx->env) && RandomBool(0.9)) {
    // Choose a random token for the channel op.
    XLS_ASSIGN_OR_RETURN(TypedExpr token,
                         ChooseEnvValue(&ctx->env, MakeTokenType()));
    token_ref = token.expr;

    int64_t successor_min_stage = token.min_stage;
    if (token.last_delaying_op == LastDelayingOp::kSend &&
        chan_op_info.channel_direction == ChannelDirection::kIn) {
      // Send depended on by recv - make sure we have a delay.
      successor_min_stage++;
    }
    min_stage = std::max(min_stage, successor_min_stage);
  } else {
    // Create a new independent token.
    token_ref = module_->Make<Invocation>(
        fake_span_, MakeBuiltinNameRef("join"), std::vector<Expr*>{});
  }
  CHECK(token_ref != nullptr);

  TypeAnnotation* token_type = module_->Make<BuiltinTypeAnnotation>(
      fake_span_, BuiltinType::kToken,
      module_->GetOrCreateBuiltinNameDef(BuiltinType::kToken));
  switch (chan_op_type) {
    case ChannelOpType::kRecv:
      return TypedExpr{.expr = module_->Make<Invocation>(
                           fake_span_, MakeBuiltinNameRef("recv"),
                           std::vector<Expr*>{token_ref, chan_expr}),
                       .type = MakeTupleType({token_type, channel_type}),
                       .last_delaying_op = LastDelayingOp::kRecv,
                       .min_stage = min_stage};
    case ChannelOpType::kRecvNonBlocking:
      return TypedExpr{.expr = module_->Make<Invocation>(
                           fake_span_, MakeBuiltinNameRef("recv_non_blocking"),
                           std::vector<Expr*>{token_ref, chan_expr,
                                              default_value.value().expr}),
                       .type = MakeTupleType({token_type, channel_type,
                                              MakeTypeAnnotation(false, 1)}),
                       .last_delaying_op = LastDelayingOp::kRecv,
                       .min_stage = min_stage};
    case ChannelOpType::kRecvIf:
      return TypedExpr{
          .expr = module_->Make<Invocation>(
              fake_span_, MakeBuiltinNameRef("recv_if"),
              std::vector<Expr*>{token_ref, chan_expr, predicate.value().expr,
                                 default_value.value().expr}),
          .type = MakeTupleType({token_type, channel_type}),
          .last_delaying_op = LastDelayingOp::kRecv,
          .min_stage = min_stage};
    case ChannelOpType::kSend:
      return TypedExpr{
          .expr = module_->Make<Invocation>(
              fake_span_, MakeBuiltinNameRef("send"),
              std::vector<Expr*>{token_ref, chan_expr, payload.value().expr}),
          .type = token_type,
          .last_delaying_op = LastDelayingOp::kSend,
          .min_stage = min_stage};
    case ChannelOpType::kSendIf:
      return TypedExpr{
          .expr = module_->Make<Invocation>(
              fake_span_, MakeBuiltinNameRef("send_if"),
              std::vector<Expr*>{token_ref, chan_expr, predicate.value().expr,
                                 payload.value().expr}),
          .type = token_type,
          .last_delaying_op = LastDelayingOp::kSend,
          .min_stage = min_stage};
  }

  LOG(FATAL) << "Invalid ChannelOpType: " << static_cast<int>(chan_op_type);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateJoinOp(Context* ctx) {
  // Lambda that chooses a token TypedExpr.
  auto token_predicate = [](const TypedExpr& e) -> bool {
    return IsToken(e.type);
  };
  std::vector<TypedExpr> tokens = GatherAllValues(&ctx->env, token_predicate);
  int64_t token_count =
      absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_, 0, tokens.size());
  std::vector<Expr*> tokens_to_join;
  std::vector<LastDelayingOp> delaying_ops;
  tokens_to_join.reserve(token_count);
  delaying_ops.reserve(token_count);
  int64_t min_stage = 1;
  for (int64_t i = 0; i < token_count; ++i) {
    TypedExpr random_token =
        RandomChoice(absl::MakeConstSpan(tokens), bit_gen_);
    tokens_to_join.push_back(random_token.expr);
    delaying_ops.push_back(random_token.last_delaying_op);
    min_stage = std::max(min_stage, random_token.min_stage);
  }
  return TypedExpr{.expr = module_->Make<Invocation>(
                       fake_span_, MakeBuiltinNameRef("join"), tokens_to_join),
                   .type = MakeTokenType(),
                   .last_delaying_op = ComposeDelayingOps(delaying_ops),
                   .min_stage = min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCompareArray(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueArray(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValue(&ctx->env, lhs.type));
  BinopKind op = RandomBool(0.5) ? BinopKind::kEq : BinopKind::kNe;
  return TypedExpr{
      .expr = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
      .type = MakeTypeAnnotation(false, 1),
      .last_delaying_op =
          ComposeDelayingOps(lhs.last_delaying_op, rhs.last_delaying_op),
      .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

class FindTokenTypeVisitor : public AstNodeVisitorWithDefault {
 public:
  FindTokenTypeVisitor() = default;

  bool GetTokenFound() const { return token_found_; }

  absl::Status HandleBuiltinTypeAnnotation(
      const BuiltinTypeAnnotation* builtin_type) override {
    if (!token_found_) {
      token_found_ = builtin_type->builtin_type() == BuiltinType::kToken;
    }
    return absl::OkStatus();
  }

  absl::Status HandleTupleTypeAnnotation(
      const TupleTypeAnnotation* tuple_type) override {
    for (TypeAnnotation* member_type : tuple_type->members()) {
      if (token_found_) {
        break;
      }
      XLS_RETURN_IF_ERROR(member_type->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleArrayTypeAnnotation(
      const ArrayTypeAnnotation* array_type) override {
    return array_type->element_type()->Accept(this);
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* type_ref_type) override {
    return type_ref_type->type_ref()->Accept(this);
  }

  absl::Status HandleTypeRef(const TypeRef* type_ref) override {
    const TypeDefinition& type_def = type_ref->type_definition();
    if (std::holds_alternative<TypeAlias*>(type_def)) {
      return std::get<TypeAlias*>(type_def)->Accept(this);
    }
    if (std::holds_alternative<StructDef*>(type_def)) {
      return std::get<StructDef*>(type_def)->Accept(this);
    }
    if (std::holds_alternative<EnumDef*>(type_def)) {
      return std::get<EnumDef*>(type_def)->Accept(this);
    }
    CHECK(std::holds_alternative<ColonRef*>(type_def));
    return std::get<ColonRef*>(type_def)->Accept(this);
  }

  absl::Status HandleTypeAlias(const TypeAlias* type_alias) override {
    return type_alias->type_annotation().Accept(this);
  }

  absl::Status HandleStructDef(const StructDef* struct_def) override {
    for (const StructMember& member : struct_def->members()) {
      if (token_found_) {
        break;
      }
      XLS_RETURN_IF_ERROR(member.type->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleEnumDef(const EnumDef* enum_def) override {
    return enum_def->type_annotation()->Accept(this);
  }

  absl::Status HandleColonRef(const ColonRef* colon_def) override {
    return absl::InternalError("ColonRef are not supported by the fuzzer.");
  }

 private:
  bool token_found_ = false;
};

/* static */ absl::StatusOr<bool> AstGenerator::ContainsToken(
    const TypeAnnotation* type) {
  FindTokenTypeVisitor token_visitor;
  XLS_RETURN_IF_ERROR(type->Accept(&token_visitor));
  return token_visitor.GetTokenFound();
}

/* static */ bool AstGenerator::ContainsTypeRef(const TypeAnnotation* type) {
  if (IsTypeRef(type)) {
    return true;
  }
  if (auto tuple_type = dynamic_cast<const TupleTypeAnnotation*>(type)) {
    for (TypeAnnotation* member_type : tuple_type->members()) {
      if (ContainsTypeRef(member_type)) {
        return true;
      }
    }
    return false;
  }
  if (auto array_type = dynamic_cast<const ArrayTypeAnnotation*>(type)) {
    return ContainsTypeRef(array_type->element_type());
  }
  if (auto channel_type = dynamic_cast<const ChannelTypeAnnotation*>(type)) {
    return ContainsTypeRef(channel_type->payload());
  }
  CHECK_NE(dynamic_cast<const BuiltinTypeAnnotation*>(type), nullptr);
  return false;
}

absl::StatusOr<TypedExpr> AstGenerator::ChooseEnvValueTupleWithoutToken(
    Env* env, int64_t min_size) {
  auto take = [&](const TypedExpr& e) -> bool {
    if (!IsTuple(e.type)) {
      return false;
    }
    TupleTypeAnnotation* tuple_type =
        dynamic_cast<TupleTypeAnnotation*>(e.type);
    if (tuple_type->size() < min_size) {
      return false;
    }
    absl::StatusOr<bool> contains_token_or = ContainsToken(tuple_type);
    CHECK_OK(contains_token_or.status());
    return !contains_token_or.value();
  };
  return ChooseEnvValue(env, take);
}

absl::StatusOr<TypedExpr> AstGenerator::ChooseEnvValueNotContainingToken(
    Env* env) {
  auto take = [&](const TypedExpr& e) -> bool {
    absl::StatusOr<bool> contains_token_or = ContainsToken(e.type);
    CHECK_OK(contains_token_or.status());
    return !contains_token_or.value();
  };
  return ChooseEnvValue(env, take);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCompareTuple(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs,
                       ChooseEnvValueTupleWithoutToken(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValue(&ctx->env, lhs.type));
  BinopKind op = RandomBool(0.5) ? BinopKind::kEq : BinopKind::kNe;
  return TypedExpr{
      .expr = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
      .type = MakeTypeAnnotation(false, 1),
      .last_delaying_op =
          ComposeDelayingOps(lhs.last_delaying_op, rhs.last_delaying_op),
      .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateExprOfType(
    Context* ctx, TypeAnnotation* type) {
  if (IsTuple(type)) {
    auto tuple_type = dynamic_cast<TupleTypeAnnotation*>(type);
    std::vector<TypedExpr> candidates = GatherAllValues(&ctx->env, tuple_type);
    // Twenty percent of the time, generate a reference if one exists.
    if (!candidates.empty() && RandomBool(0.20)) {
      return RandomChoice(absl::MakeConstSpan(candidates), bit_gen_);
    }
    int64_t min_stage = 1;
    std::vector<TypedExpr> tuple_entries(tuple_type->size());
    for (int64_t index = 0; index < tuple_type->size(); ++index) {
      XLS_ASSIGN_OR_RETURN(
          tuple_entries[index],
          GenerateExprOfType(ctx, tuple_type->members()[index]));
      min_stage = std::max(min_stage, tuple_entries[index].min_stage);
    }
    auto [tuple_values, tuple_types, tuple_delaying_ops] = Unzip(tuple_entries);
    return TypedExpr{
        .expr = module_->Make<XlsTuple>(fake_span_, tuple_values,
                                        /*has_trailing_comma=*/false),
        .type = type,
        .last_delaying_op = ComposeDelayingOps(tuple_delaying_ops),
        .min_stage = min_stage};
  }
  if (IsArray(type)) {
    auto array_type = dynamic_cast<ArrayTypeAnnotation*>(type);
    std::vector<TypedExpr> candidates = GatherAllValues(&ctx->env, array_type);
    // Twenty percent of the time, generate a reference if one exists.
    if (!candidates.empty() && RandomBool(0.20)) {
      return RandomChoice(absl::MakeConstSpan(candidates), bit_gen_);
    }
    int64_t min_stage = 1;
    int64_t array_size = GetArraySize(array_type);
    std::vector<TypedExpr> array_entries(array_size);
    for (int64_t index = 0; index < array_size; ++index) {
      XLS_ASSIGN_OR_RETURN(array_entries[index],
                           GenerateExprOfType(ctx, array_type->element_type()));
      min_stage = std::max(min_stage, array_entries[index].min_stage);
    }
    auto [array_values, array_types, array_delaying_ops] = Unzip(array_entries);
    return TypedExpr{.expr = module_->Make<Array>(fake_span_, array_values,
                                                  /*has_ellipsis=*/false),
                     .type = type,
                     .last_delaying_op = ComposeDelayingOps(array_delaying_ops),
                     .min_stage = min_stage};
  }
  if (auto* type_ref_type = dynamic_cast<const TypeRefTypeAnnotation*>(type)) {
    TypeRef* type_ref = type_ref_type->type_ref();
    const TypeDefinition& type_def = type_ref->type_definition();
    CHECK(std::holds_alternative<TypeAlias*>(type_def));
    TypeAlias* alias = std::get<TypeAlias*>(type_def);
    return GenerateExprOfType(ctx, &alias->type_annotation());
  }
  CHECK(IsBits(type));
  std::vector<TypedExpr> candidates = GatherAllValues(&ctx->env, type);
  // Twenty percent of the time, generate a reference if one exists.
  if (!candidates.empty() && RandomBool(0.20)) {
    return RandomChoice(absl::MakeConstSpan(candidates), bit_gen_);
  }
  return GenerateNumberWithType(
      BitsAndSignedness{GetTypeBitCount(type), BitsTypeIsSigned(type).value()});
}

absl::StatusOr<NameDefTree*> AstGenerator::GenerateMatchArmPattern(
    Context* ctx, const TypeAnnotation* type) {
  if (IsTuple(type)) {
    auto tuple_type = dynamic_cast<const TupleTypeAnnotation*>(type);
    // Ten percent of the time, generate a wildcard pattern.
    if (RandomBool(0.1)) {
      WildcardPattern* wc = module_->Make<WildcardPattern>(fake_span_);
      return module_->Make<NameDefTree>(fake_span_, wc);
    }

    // Ten percent of tuples should have a "rest of tuple", and skip
    // a random # of the rest of the tuple.
    auto insert_rest_of_tuple = RandomBool(0.1);
    auto rest_of_tuple_index =
        RandomIntWithExpectedValue(tuple_type->size() / 2.0, 0);
    auto number_to_skip =
        RandomIntWithExpectedValue(tuple_type->size() / 2.0, 0);
    std::vector<NameDefTree*> tuple_values;
    tuple_values.reserve(tuple_type->size() + 1);
    for (int64_t index = 0; index < tuple_type->size(); ++index) {
      if (rest_of_tuple_index == index && insert_rest_of_tuple) {
        insert_rest_of_tuple = false;

        RestOfTuple* rest = module_->Make<RestOfTuple>(fake_span_);
        tuple_values.push_back(module_->Make<NameDefTree>(fake_span_, rest));

        // Jump forward a random # of elements
        if (number_to_skip > 0) {
          index += number_to_skip;
          // We could keep this entry, or skip it, but for now, skip it.
          continue;
        }
        // The jump forward is 0; we'll keep this entry.
      }

      XLS_ASSIGN_OR_RETURN(
          NameDefTree * pattern,
          GenerateMatchArmPattern(ctx, tuple_type->members()[index]));
      tuple_values.push_back(pattern);
    }
    return module_->Make<NameDefTree>(fake_span_, tuple_values);
  }

  if (IsArray(type)) {
    // For the array type, only name references are supported in the match arm.
    // Reference: https://github.com/google/xls/issues/810.
    auto array_matches = [&type](const TypedExpr& e) -> bool {
      return !ContainsTypeRef(e.type) && e.type->ToString() == type->ToString();
    };
    std::vector<TypedExpr> array_candidates =
        GatherAllValues(&ctx->env, array_matches);
    // Twenty percent of the time, generate a wildcard pattern.
    if (array_candidates.empty() || RandomBool(0.20)) {
      WildcardPattern* wc = module_->Make<WildcardPattern>(fake_span_);
      return module_->Make<NameDefTree>(fake_span_, wc);
    }
    TypedExpr array =
        RandomChoice(absl::MakeConstSpan(array_candidates), bit_gen_);
    NameRef* name_ref = dynamic_cast<NameRef*>(array.expr);
    return module_->Make<NameDefTree>(fake_span_, name_ref);
  }
  if (auto* type_ref_type = dynamic_cast<const TypeRefTypeAnnotation*>(type)) {
    TypeRef* type_ref = type_ref_type->type_ref();
    const TypeDefinition& type_definition = type_ref->type_definition();
    CHECK(std::holds_alternative<TypeAlias*>(type_definition));
    TypeAlias* alias = std::get<TypeAlias*>(type_definition);
    return GenerateMatchArmPattern(ctx, &alias->type_annotation());
  }

  CHECK(IsBits(type));

  // Five percent of the time, generate a wildcard pattern.
  if (RandomBool(0.05)) {
    WildcardPattern* wc = module_->Make<WildcardPattern>(fake_span_);
    return module_->Make<NameDefTree>(fake_span_, wc);
  }

  // Fifteen percent of the time, generate a range. (Note that this is an
  // independent float from the one tested above, so it's a 15% chance not 10%
  // chance.)
  if (RandomBool(0.15)) {
    int64_t bit_count = GetTypeBitCount(type);
    bool is_signed = BitsTypeIsSigned(type).value();
    Bits start_bits;
    TypedExpr start_type_expr = GenerateNumberWithType(
        BitsAndSignedness{bit_count, is_signed}, &start_bits);

    auto start = InterpValue::MakeBits(is_signed, start_bits);
    auto max = InterpValue::MakeMaxValue(is_signed, bit_count);
    bool start_lt_max = start.Lt(max).value().IsTrue();

    TypedExpr limit_type_expr;
    // 30% of the time make a random number, rest of the time make it a
    // non-empty range.
    //
    // TODO(leary): 2023-08-04 If the bit count is too high we don't have an
    // easy RNG available to select out of the remaining range.
    if (RandomBool(0.3) || bit_count >= 64 || !start_lt_max) {
      // Sometimes pick an arbitrary limit in the bitwidth.
      limit_type_expr =
          GenerateNumberWithType(BitsAndSignedness{bit_count, is_signed});
    } else {
      // Other times pick a limit that's >= the start value. Note that this can
      // still be an empty range if we chose the value that's equal to the start
      // value -- this simplifies the logic a bit since then we don't need to
      // worry about whether start+1 exceeds the max value.
      XLS_ASSIGN_OR_RETURN(int64_t start_int64, start.GetBitValueViaSign());
      XLS_ASSIGN_OR_RETURN(int64_t max_int64, max.GetBitValueViaSign());
      int64_t limit = absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_,
                                             start_int64, max_int64);
      limit_type_expr = TypedExpr{MakeNumber(limit, start_type_expr.type),
                                  start_type_expr.type};
    }
    Range* range = module_->Make<Range>(fake_span_, start_type_expr.expr,
                                        limit_type_expr.expr);
    return module_->Make<NameDefTree>(fake_span_, range);
  }

  // Rest of the time we generate a simple number as the pattern to match.
  TypedExpr type_expr = GenerateNumberWithType(
      BitsAndSignedness{GetTypeBitCount(type), BitsTypeIsSigned(type).value()});
  Number* number = dynamic_cast<Number*>(type_expr.expr);
  CHECK_NE(number, nullptr);
  return module_->Make<NameDefTree>(fake_span_, number);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateMatch(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr match,
                       ChooseEnvValueNotContainingToken(&ctx->env));
  LastDelayingOp last_delaying_op = match.last_delaying_op;
  int64_t min_stage = match.min_stage;
  TypeAnnotation* match_return_type = GenerateType();
  // Attempt to create at least one additional match arm aside from the wildcard
  // pattern.
  int64_t max_arm_count =
      absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_, 1, 4);
  std::vector<MatchArm*> match_arms;
  // `match` will flag an error if a syntactically identical pattern is typed
  // twice. Using the `std::string` equivalent of the pattern for comparing
  // syntactically identical patterns. Ref:
  // https://google.github.io/xls/dslx_reference/#redundant-patterns.
  absl::flat_hash_set<std::string> all_match_arms_patterns;
  for (int64_t arm_count = 0; arm_count < max_arm_count; ++arm_count) {
    std::vector<NameDefTree*> match_arm_patterns;
    // Attempt to create at least one pattern.
    int64_t max_pattern_count =
        absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_, 1, 2);
    for (int64_t pattern_count = 0; pattern_count < max_pattern_count;
         ++pattern_count) {
      XLS_ASSIGN_OR_RETURN(NameDefTree * pattern,
                           GenerateMatchArmPattern(ctx, match.type));
      // Early exit when a wildcard pattern is created.
      if (pattern->is_leaf() &&
          std::holds_alternative<WildcardPattern*>(pattern->leaf())) {
        break;
      }
      std::string pattern_str = pattern->ToString();
      if (all_match_arms_patterns.contains(pattern_str)) {
        continue;
      }
      all_match_arms_patterns.insert(pattern_str);
      match_arm_patterns.push_back(pattern);
    }
    // No patterns created on this attempt.
    if (match_arm_patterns.empty()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(TypedExpr ret,
                         GenerateExprOfType(ctx, match_return_type));
    last_delaying_op =
        ComposeDelayingOps(last_delaying_op, ret.last_delaying_op);
    min_stage = std::max(min_stage, ret.min_stage);
    match_arms.push_back(
        module_->Make<MatchArm>(fake_span_, match_arm_patterns, ret.expr));
  }
  // Add wildcard pattern as last match arm.
  XLS_ASSIGN_OR_RETURN(TypedExpr wc_return,
                       GenerateExprOfType(ctx, match_return_type));
  WildcardPattern* wc = module_->Make<WildcardPattern>(fake_span_);
  NameDefTree* wc_pattern = module_->Make<NameDefTree>(fake_span_, wc);
  match_arms.push_back(module_->Make<MatchArm>(
      fake_span_, std::vector<NameDefTree*>{wc_pattern}, wc_return.expr));
  last_delaying_op =
      ComposeDelayingOps(last_delaying_op, wc_return.last_delaying_op);
  min_stage = std::max(min_stage, wc_return.min_stage);
  return TypedExpr{
      .expr = module_->Make<Match>(fake_span_, match.expr, match_arms),
      .type = match_return_type,
      .last_delaying_op = last_delaying_op,
      .min_stage = min_stage,
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateSynthesizableDiv(Context* ctx) {
  // TODO(tedhong): 2022-10-21 When https://github.com/google/xls/issues/746
  // is resolved, remove bitcount constraint.
  XLS_ASSIGN_OR_RETURN(
      TypedExpr lhs,
      ChooseEnvValueBitsInRange(
          &ctx->env, 1, std::min(int64_t{64}, options_.max_width_bits_types)));
  // Divide by an arbitrary literal.
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, BitsTypeGetBitCount(lhs.type));
  Bits divisor = GenerateBits(bit_gen_, bit_count);
  Number* divisor_node = GenerateNumberFromBits(divisor, lhs.type);
  return TypedExpr{.expr = module_->Make<Binop>(fake_span_, BinopKind::kDiv,
                                                lhs.expr, divisor_node),
                   .type = lhs.type,
                   .last_delaying_op = lhs.last_delaying_op,
                   .min_stage = lhs.min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateShift(Context* ctx) {
  BinopKind op = RandomChoice(GetBinopShifts(), bit_gen_);
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueBits(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueUBits(&ctx->env));
  int64_t lhs_bit_count = GetTypeBitCount(lhs.type);
  int64_t rhs_bit_count = GetTypeBitCount(rhs.type);
  if (lhs_bit_count > 0 && rhs_bit_count > 0) {
    if (RandomBool(0.8)) {
      // Clamp the shift rhs to be in range most of the time.  First find the
      // maximum value representable in the RHS type as this imposes a different
      // limit on the magnitude of the shift amount. If the RHS type is 63 bits
      // or wider then just use int max (it's 63 instead of 64 because we're
      // using an signed 64-bit type to hold the limit).
      int64_t max_rhs_value = rhs_bit_count < 63
                                  ? ((int64_t{1} << rhs_bit_count) - 1)
                                  : std::numeric_limits<int64_t>::max();

      int64_t shift_limit = std::min(lhs_bit_count, max_rhs_value);
      int64_t new_upper = absl::Uniform<int64_t>(bit_gen_, 0, shift_limit);
      XLS_ASSIGN_OR_RETURN(rhs.expr, GenerateUmin(rhs, new_upper));
    } else if (RandomBool(0.5)) {
      // Generate a numerical value (Number) as an untyped literal instead of
      // the value we chose above.
      int64_t shift_amount = absl::Uniform<int64_t>(bit_gen_, 0, lhs_bit_count);
      rhs = TypedExpr();
      rhs.expr = MakeNumber(shift_amount);
    }
  }
  return TypedExpr{
      .expr = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
      .type = lhs.type,
      .last_delaying_op =
          ComposeDelayingOps(lhs.last_delaying_op, rhs.last_delaying_op),
      .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

absl::StatusOr<TypedExpr>
AstGenerator::GeneratePartialProductDeterministicGroup(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(&ctx->env));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;
  bool is_signed = RandomBool(0.5);

  std::string op = is_signed ? "smulp" : "umulp";

  CHECK(IsBits(lhs.type));
  CHECK(IsBits(rhs.type));

  TypedExpr lhs_cast, rhs_cast;
  // Don't need a cast if lhs.type matches the sign of the op
  if (is_signed != IsUBits(lhs.type)) {
    lhs_cast = lhs;
  } else {
    lhs_cast.type = MakeTypeAnnotation(is_signed, GetTypeBitCount(lhs.type));
    lhs_cast.expr = module_->Make<Cast>(fake_span_, lhs.expr, lhs_cast.type);
    lhs_cast.last_delaying_op = lhs.last_delaying_op;
    lhs_cast.min_stage = lhs.min_stage;
  }
  // Don't need a cast if rhs.type matches the sign of the op
  if (is_signed != IsUBits(rhs.type)) {
    rhs_cast = rhs;
  } else {
    rhs_cast.type = MakeTypeAnnotation(is_signed, GetTypeBitCount(rhs.type));
    rhs_cast.expr = module_->Make<Cast>(fake_span_, rhs.expr, rhs_cast.type);
    rhs_cast.last_delaying_op = rhs.last_delaying_op;
    rhs_cast.min_stage = rhs.min_stage;
  }

  TypeAnnotation* unsigned_type =
      MakeTypeAnnotation(false, GetTypeBitCount(lhs.type));
  TypeAnnotation* signed_type =
      MakeTypeAnnotation(true, GetTypeBitCount(lhs.type));
  auto mulp = TypedExpr{.expr = module_->Make<Invocation>(
                            fake_span_, MakeBuiltinNameRef(op),
                            std::vector<Expr*>{lhs_cast.expr, rhs_cast.expr}),
                        .type = MakeTupleType({unsigned_type, unsigned_type}),
                        .last_delaying_op = ComposeDelayingOps(
                            {lhs.last_delaying_op, rhs.last_delaying_op}),
                        .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
  std::string mulp_identifier = GenSym();
  auto* mulp_name_def =
      module_->Make<NameDef>(fake_span_, mulp_identifier, /*definer=*/nullptr);
  auto* mulp_name_ref = MakeNameRef(mulp_name_def);
  auto* ndt = module_->Make<NameDefTree>(fake_span_, mulp_name_def);
  auto mulp_lhs = module_->Make<TupleIndex>(fake_span_, mulp_name_ref,
                                            /*index=*/MakeNumber(0));
  auto mulp_rhs = module_->Make<TupleIndex>(fake_span_, mulp_name_ref,
                                            /*index=*/MakeNumber(1));
  Expr* sum =
      module_->Make<Binop>(fake_span_, BinopKind::kAdd, mulp_lhs, mulp_rhs);
  if (is_signed) {  // For smul we have to cast the summation to signed.
    sum = module_->Make<Cast>(fake_span_, sum, signed_type);
  }
  auto* let = module_->Make<Let>(fake_span_, /*name_def_tree=*/ndt,
                                 /*type=*/mulp.type, /*rhs=*/mulp.expr,
                                 /*is_const=*/false);
  auto* body_stmt = module_->Make<Statement>(let);
  auto* block = module_->Make<StatementBlock>(fake_span_,
                                              std::vector<Statement*>{
                                                  body_stmt,
                                                  module_->Make<Statement>(sum),
                                              },
                                              /*trailing_semi=*/false);
  return TypedExpr{.expr = block,
                   .type = lhs_cast.type,
                   .last_delaying_op = mulp.last_delaying_op,
                   .min_stage = mulp.min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBinop(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(auto pair, ChooseEnvValueBitsPair(&ctx->env));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;
  BinopKind op = RandomChoice(GetBinopSameTypeKinds(), bit_gen_);
  if (op == BinopKind::kDiv) {
    return GenerateSynthesizableDiv(ctx);
  }
  if (GetBinopShifts().contains(op)) {
    return GenerateShift(ctx);
  }
  return TypedExpr{
      .expr = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
      .type = lhs.type,
      .last_delaying_op =
          ComposeDelayingOps(lhs.last_delaying_op, rhs.last_delaying_op),
      .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateLogicalOp(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(auto pair,
                       ChooseEnvValueBitsPair(&ctx->env, /*bit_count=*/1));
  TypedExpr lhs = pair.first;
  TypedExpr rhs = pair.second;

  // Pick some operation to do.
  BinopKind op = RandomChoice(
      absl::MakeConstSpan({BinopKind::kAnd, BinopKind::kOr, BinopKind::kXor}),
      bit_gen_);
  return TypedExpr{
      .expr = module_->Make<Binop>(fake_span_, op, lhs.expr, rhs.expr),
      .type = lhs.type,
      .last_delaying_op =
          ComposeDelayingOps(lhs.last_delaying_op, rhs.last_delaying_op),
      .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

Number* AstGenerator::MakeNumber(int64_t value, TypeAnnotation* type) {
  return module_->Make<Number>(fake_span_, absl::StrFormat("%d", value),
                               NumberKind::kOther, type);
}

Number* AstGenerator::MakeNumberFromBits(const Bits& value,
                                         TypeAnnotation* type,
                                         FormatPreference format_preference) {
  return module_->Make<Number>(fake_span_,
                               BitsToString(value, format_preference),
                               NumberKind::kOther, type);
}

Number* AstGenerator::GenerateNumber(int64_t value, TypeAnnotation* type) {
  CHECK_NE(type, nullptr);
  int64_t bit_count = BitsTypeGetBitCount(type).value();
  Bits value_bits;
  if (BitsTypeIsSigned(type).value()) {
    value_bits = SBits(value, bit_count);
  } else {
    value_bits = UBits(value, bit_count);
  }

  return GenerateNumberFromBits(value_bits, type);
}

Number* AstGenerator::GenerateNumberFromBits(const Bits& value,
                                             TypeAnnotation* type) {
  // 50% of the time, when the type is unsigned and it has a single bit,
  // generate the string representation equivalent to the boolean value.
  if (!BitsTypeIsSigned(type).value() && value.bit_count() == 1 &&
      RandomBool(0.5)) {
    return module_->Make<Number>(fake_span_, value.Get(0) ? "true" : "false",
                                 NumberKind::kBool, type);
  }
  // 50% of the time, when the type is unsigned and its bit width is eight,
  // generate a "character" Number.
  if (!BitsTypeIsSigned(type).value() && value.bit_count() == 8 &&
      std::isprint(value.ToBytes()[0]) != 0 && RandomBool(0.5)) {
    return module_->Make<Number>(fake_span_, std::string(1, value.ToBytes()[0]),
                                 NumberKind::kCharacter, type);
  }

  // Most of the time (90%), generate a hexadecimal representation of the
  // literal.
  if (RandomBool(0.9)) {
    return MakeNumberFromBits(value, type, FormatPreference::kHex);
  }

  // 5% of the time (half the remainder): generate a decimal representation if
  // we can. As stated in google/xls#461, decimal values can only be emitted if
  // they fit in an int64_t/uint64_t.
  bool is_signed = BitsTypeIsSigned(type).value();
  bool can_emit_decimal = (is_signed && value.FitsInInt64()) ||
                          (!is_signed && value.FitsInUint64());
  if (RandomBool(0.5) && can_emit_decimal) {
    if (is_signed) {
      return MakeNumberFromBits(value, type, FormatPreference::kSignedDecimal);
    }
    return MakeNumberFromBits(value, type, FormatPreference::kUnsignedDecimal);
  }

  // Otherwise, generate a binary representation of the literal.
  return MakeNumberFromBits(value, type, FormatPreference::kBinary);
}

int64_t AstGenerator::GetTypeBitCount(const TypeAnnotation* type) {
  std::string type_str = type->ToString();
  if (type_str == "uN" || type_str == "sN" || type_str == "bits") {
    // These types are not valid alone, but as the element type of an array
    // (e.g. uN[42]) where they effectively have a width of one bit.
    return 1;
  }

  if (auto* builtin = dynamic_cast<const BuiltinTypeAnnotation*>(type)) {
    return builtin->GetBitCount();
  }
  if (auto* array = dynamic_cast<const ArrayTypeAnnotation*>(type)) {
    return GetArraySize(array) * GetTypeBitCount(array->element_type());
  }
  if (auto* tuple = dynamic_cast<const TupleTypeAnnotation*>(type)) {
    int64_t total = 0;
    for (TypeAnnotation* member_type : tuple->members()) {
      total += GetTypeBitCount(member_type);
    }
    return total;
  }
  if (auto* type_alias = dynamic_cast<const TypeAlias*>(type)) {
    return GetTypeBitCount(&type_alias->type_annotation());
  }

  return type_bit_counts_.at(type_str);
}

absl::StatusOr<uint64_t> AstGenerator::GetExprAsUint64(Expr* expr) {
  const FileTable& file_table = *expr->owner()->file_table();
  if (auto* number = dynamic_cast<Number*>(expr)) {
    return number->GetAsUint64(file_table);
  }
  auto* const_ref = dynamic_cast<ConstRef*>(expr);
  if (const_ref == nullptr) {
    return absl::InvalidArgumentError("Expression is not a number or constant");
  }
  ConstantDef* const_def = constants_[const_ref->identifier()];
  Number* number = dynamic_cast<Number*>(const_def->value());
  XLS_RET_CHECK(number != nullptr) << const_def->ToString();
  return number->GetAsUint64(file_table);
}

int64_t AstGenerator::GetArraySize(const ArrayTypeAnnotation* type) {
  return GetExprAsUint64(type->dim()).value();
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
                                              /*type_annotation=*/nullptr,
                                              GenerateNumber(value, size_type),
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
  if (RandomBool(0.5)) {
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

absl::StatusOr<TypedExpr> AstGenerator::GenerateOneHotSelectBuiltin(
    Context* ctx) {
  // We need to choose a selector with a certain number of bits, then form an
  // array from that many values in the environment.
  constexpr int64_t kMaxBitCount = 8;
  auto choose_value = [this](const TypedExpr& e) -> bool {
    TypeAnnotation* t = e.type;
    return IsUBits(t) && 0 < GetTypeBitCount(t) &&
           GetTypeBitCount(t) <= kMaxBitCount;
  };

  std::optional<TypedExpr> lhs =
      ChooseEnvValueOptional(&ctx->env, /*take=*/choose_value);
  if (!lhs.has_value()) {
    // If there's no natural environment value to use as the LHS, make up a
    // number and number of bits.
    int64_t bits = absl::Uniform<int64_t>(bit_gen_, 1, kMaxBitCount);
    lhs = GenerateNumberWithType(BitsAndSignedness{bits, false});
  }
  LastDelayingOp last_delaying_op = lhs->last_delaying_op;
  int64_t min_stage = lhs->min_stage;

  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueBits(&ctx->env));
  std::vector<Expr*> cases = {rhs.expr};
  int64_t total_operands = GetTypeBitCount(lhs->type);
  last_delaying_op = ComposeDelayingOps(last_delaying_op, rhs.last_delaying_op);
  min_stage = std::max(min_stage, rhs.min_stage);
  for (int64_t i = 0; i < total_operands - 1; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValue(&ctx->env, rhs.type));
    cases.push_back(e.expr);
    last_delaying_op = ComposeDelayingOps(last_delaying_op, e.last_delaying_op);
    min_stage = std::max(min_stage, e.min_stage);
  }

  XLS_RETURN_IF_ERROR(
      VerifyAggregateWidth(cases.size() * GetTypeBitCount(rhs.type)));

  auto* cases_array =
      module_->Make<Array>(fake_span_, cases, /*has_ellipsis=*/false);
  auto* invocation =
      module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("one_hot_sel"),
                                std::vector<Expr*>{lhs->expr, cases_array});
  return TypedExpr{.expr = invocation,
                   .type = rhs.type,
                   .last_delaying_op = last_delaying_op,
                   .min_stage = min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GeneratePrioritySelectBuiltin(
    Context* ctx) {
  XLS_ASSIGN_OR_RETURN(
      TypedExpr selector,
      ChooseEnvValueUBits(&ctx->env, /*bit_count=*/RandomIntWithExpectedValue(
                              5, /*lower_limit=*/1)));

  LastDelayingOp last_delaying_op = selector.last_delaying_op;
  int64_t min_stage = selector.min_stage;

  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueBits(&ctx->env));
  last_delaying_op = ComposeDelayingOps(last_delaying_op, rhs.last_delaying_op);
  min_stage = std::max(min_stage, rhs.min_stage);
  std::vector<Expr*> cases = {rhs.expr};
  int64_t total_operands = GetTypeBitCount(selector.type);
  for (int64_t i = 0; i < total_operands - 1; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValue(&ctx->env, rhs.type));
    cases.push_back(e.expr);
  }

  XLS_RETURN_IF_ERROR(
      VerifyAggregateWidth(cases.size() * GetTypeBitCount(rhs.type)));

  auto* cases_array =
      module_->Make<Array>(fake_span_, cases, /*has_ellipsis=*/false);
  XLS_ASSIGN_OR_RETURN(TypedExpr default_value,
                       ChooseEnvValue(&ctx->env, rhs.type));
  auto* invocation = module_->Make<Invocation>(
      fake_span_, MakeBuiltinNameRef("priority_sel"),
      std::vector<Expr*>{selector.expr, cases_array, default_value.expr});
  return TypedExpr{
      .expr = invocation,
      .type = rhs.type,
      .last_delaying_op = last_delaying_op,
      .min_stage = min_stage,
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateSignExtendBuiltin(
    Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueBits(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr rhs, ChooseEnvValueBits(&ctx->env));
  return TypedExpr{
      .expr =
          module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("signex"),
                                    std::vector<Expr*>{lhs.expr, rhs.expr}),
      .type = rhs.type,
      .last_delaying_op =
          ComposeDelayingOps(lhs.last_delaying_op, rhs.last_delaying_op),
      .min_stage = std::max(lhs.min_stage, rhs.min_stage),
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArrayConcat(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr lhs, ChooseEnvValueArray(&ctx->env));
  auto* lhs_array_type = dynamic_cast<ArrayTypeAnnotation*>(lhs.type);
  XLS_RET_CHECK(lhs_array_type != nullptr);

  auto array_compatible = [&](const TypedExpr& e) -> bool {
    TypeAnnotation* t = e.type;
    if (auto* array = dynamic_cast<ArrayTypeAnnotation*>(t)) {
      return array->element_type() == lhs_array_type->element_type();
    }
    return false;
  };

  XLS_ASSIGN_OR_RETURN(TypedExpr rhs,
                       ChooseEnvValue(&ctx->env, array_compatible));

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
  return TypedExpr{.expr = result,
                   .type = result_type,
                   .last_delaying_op = ComposeDelayingOps(
                       {lhs.last_delaying_op, rhs.last_delaying_op}),
                   .min_stage = std::max(lhs.min_stage, rhs.min_stage)};
}

String* AstGenerator::GenerateString(int64_t char_count) {
  std::string string_literal(char_count, '\0');
  for (int64_t index = 0; index < char_count; ++index) {
    // Codes 32 to 126 are the printable characters. There are 95 printable
    // characters in total. Ref: https://en.wikipedia.org/wiki/ASCII.
    string_literal[index] =
        absl::Uniform<uint8_t>(absl::IntervalClosed, bit_gen_, 32, 126);
  }
  return module_->Make<String>(fake_span_, string_literal);
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArray(Context* ctx) {
  // 5% of the time generate string literals as arrays.
  int64_t byte_count = options_.max_width_aggregate_types / 8;
  if (byte_count > 0 && RandomBool(0.05)) {
    int64_t length =
        absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_, 1, byte_count);
    return TypedExpr{GenerateString(length),
                     MakeArrayType(MakeTypeAnnotation(false, 8), length)};
  }
  // Choose an arbitrary non-token value from the environment, then gather all
  // elements from the environment of that type.
  XLS_ASSIGN_OR_RETURN(TypedExpr value,
                       ChooseEnvValueNotContainingToken(&ctx->env));
  std::vector<TypedExpr> values = GatherAllValues(
      &ctx->env, [&](const TypedExpr& t) { return t.type == value.type; });
  XLS_RET_CHECK(!values.empty());
  if (RandomBool(0.5)) {
    // Half the time extend the set of values by duplicating members. Walk
    // through the vector randomly duplicating members along the way. On average
    // this process will double the size of the array with the distribution
    // falling off exponentially.
    for (int64_t i = 0; i < values.size(); ++i) {
      if (RandomBool(0.5)) {
        values.push_back(RandomChoice(absl::MakeConstSpan(values), bit_gen_));
      }
    }
  }
  std::vector<Expr*> value_exprs;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 0;
  int64_t total_width = 0;
  value_exprs.reserve(values.size());
  for (TypedExpr t : values) {
    value_exprs.push_back(t.expr);
    total_width += GetTypeBitCount(t.type);
    last_delaying_op = ComposeDelayingOps(last_delaying_op, t.last_delaying_op);
    min_stage = std::max(min_stage, t.min_stage);
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

  return TypedExpr{.expr = module_->Make<Array>(fake_span_, value_exprs,
                                                /*has_ellipsis=*/false),
                   .type = result_type,
                   .last_delaying_op = last_delaying_op,
                   .min_stage = min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArrayIndex(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr array, ChooseEnvValueArray(&ctx->env));
  ArrayTypeAnnotation* array_type = down_cast<ArrayTypeAnnotation*>(array.type);
  XLS_ASSIGN_OR_RETURN(TypedExpr index, ChooseEnvValueUBits(&ctx->env));
  int64_t array_size = GetArraySize(array_type);
  // An out-of-bounds array index raises an error in the DSLX interpreter so
  // clamp the index so it is always in-bounds.
  // TODO(https://github.com/google/xls/issues/327) 2021-03-05 Unify OOB
  // behavior across different levels in XLS.
  if (GetTypeBitCount(index.type) >= Bits::MinBitCountUnsigned(array_size)) {
    int64_t index_bound = absl::Uniform<int64_t>(bit_gen_, 0, array_size);
    XLS_ASSIGN_OR_RETURN(index.expr, GenerateUmin(index, index_bound));
  }
  return TypedExpr{
      .expr = module_->Make<Index>(fake_span_, array.expr, index.expr),
      .type = array_type->element_type(),
      .last_delaying_op =
          ComposeDelayingOps(array.last_delaying_op, index.last_delaying_op),
      .min_stage = std::max(array.min_stage, index.min_stage),
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArrayUpdate(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr array, ChooseEnvValueArray(&ctx->env));
  ArrayTypeAnnotation* array_type = down_cast<ArrayTypeAnnotation*>(array.type);
  XLS_ASSIGN_OR_RETURN(TypedExpr index, ChooseEnvValueUBits(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr element,
                       ChooseEnvValue(&ctx->env, array_type->element_type()));
  int64_t array_size = GetArraySize(array_type);
  // An out-of-bounds array update raises an error in the DSLX interpreter so
  // clamp the index so it is always in-bounds.
  // TODO(https://github.com/google/xls/issues/327) 2021-03-05 Unify OOB
  // behavior across different levels in XLS.
  if (GetTypeBitCount(index.type) >= Bits::MinBitCountUnsigned(array_size)) {
    int64_t index_bound = absl::Uniform<int64_t>(bit_gen_, 0, array_size);
    XLS_ASSIGN_OR_RETURN(index.expr, GenerateUmin(index, index_bound));
  }
  return TypedExpr{
      .expr = module_->Make<Invocation>(
          fake_span_, MakeBuiltinNameRef("update"),
          std::vector<Expr*>{array.expr, index.expr, element.expr}),
      .type = array.type,
      .last_delaying_op =
          ComposeDelayingOps(array.last_delaying_op, index.last_delaying_op,
                             element.last_delaying_op),
      .min_stage = std::max(std::max(array.min_stage, index.min_stage),
                            element.min_stage),
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateGate(Context* ctx) {
  XLS_RET_CHECK(ctx != nullptr);
  XLS_ASSIGN_OR_RETURN(TypedExpr p, GenerateCompare(ctx));
  XLS_ASSIGN_OR_RETURN(TypedExpr value, ChooseEnvValueBits(&ctx->env));
  return TypedExpr{
      .expr = module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("gate!"),
                                        std::vector<Expr*>{p.expr, value.expr}),
      .type = value.type,
      .last_delaying_op =
          ComposeDelayingOps(p.last_delaying_op, value.last_delaying_op),
      .min_stage = std::max(p.min_stage, value.min_stage),
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateConcat(Context* ctx) {
  XLS_RET_CHECK(ctx != nullptr);
  if (EnvContainsArray(ctx->env) && RandomBool(0.5)) {
    return GenerateArrayConcat(ctx);
  }

  // Pick the number of operands of the concat. We need at least one value.
  XLS_ASSIGN_OR_RETURN(int64_t count,
                       GenerateNaryOperandCount(ctx, /*lower_limit=*/1));
  std::vector<TypedExpr> operands;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 0;
  int64_t total_width = 0;
  for (int64_t i = 0; i < count; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValueUBits(&ctx->env));
    operands.push_back(e);
    total_width += GetTypeBitCount(e.type);
    last_delaying_op = ComposeDelayingOps(last_delaying_op, e.last_delaying_op);
    min_stage = std::max(min_stage, e.min_stage);
  }

  XLS_RETURN_IF_ERROR(VerifyBitsWidth(total_width));

  TypedExpr e = operands[0];
  Expr* result = e.expr;
  for (int64_t i = 1; i < count; ++i) {
    result = module_->Make<Binop>(fake_span_, BinopKind::kConcat, result,
                                  operands[i].expr);
  }

  TypeAnnotation* return_type = MakeTypeAnnotation(false, total_width);
  return TypedExpr{.expr = result,
                   .type = return_type,
                   .last_delaying_op = last_delaying_op,
                   .min_stage = min_stage};
}

BuiltinTypeAnnotation* AstGenerator::GeneratePrimitiveType(
    std::optional<int64_t> max_width_bits_types) {
  int64_t max_width = options_.max_width_bits_types;
  if (max_width_bits_types.has_value()) {
    max_width = max_width_bits_types.value();
  }
  int64_t integral = absl::Uniform<int64_t>(
      bit_gen_, 0, std::min(kConcreteBuiltinTypeLimit, max_width + 1));
  auto type = static_cast<BuiltinType>(integral);
  return module_->Make<BuiltinTypeAnnotation>(
      fake_span_, type, module_->GetOrCreateBuiltinNameDef(type));
}

TypedExpr AstGenerator::GenerateNumberWithType(
    std::optional<BitsAndSignedness> bas, Bits* out) {
  TypeAnnotation* type;
  if (bas.has_value()) {
    type = MakeTypeAnnotation(bas->signedness, bas->bits);
  } else {
    BuiltinTypeAnnotation* builtin_type = GeneratePrimitiveType();
    type = builtin_type;
  }
  int64_t bit_count = GetTypeBitCount(type);

  Bits value = GenerateBits(bit_gen_, bit_count);
  if (out != nullptr) {
    *out = value;
  }
  return TypedExpr{GenerateNumberFromBits(value, type), type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateRetval(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(int64_t retval_count, GenerateNaryOperandCount(ctx, 0));

  std::vector<TypedExpr> env_params;
  std::vector<TypedExpr> env_non_params;
  for (auto& item : ctx->env) {
    if (auto* name_ref = dynamic_cast<NameRef*>(item.second.expr);
        name_ref != nullptr && name_ref->DefinerIs<Param>()) {
      env_params.push_back(item.second);
    } else {
      env_non_params.push_back(item.second);
    }
  }

  XLS_RET_CHECK(!env_params.empty() || !env_non_params.empty());

  std::vector<TypedExpr> typed_exprs;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 1;
  int64_t total_bit_count = 0;
  for (int64_t i = 0; i < retval_count; ++i) {
    TypedExpr expr;
    if (env_non_params.empty() || (RandomBool(0.1) && !env_params.empty())) {
      expr = RandomChoice(env_params, bit_gen_);
    } else {
      expr = RandomChoice(env_non_params, bit_gen_);
    }

    // See if the value we selected is going to push us over the "aggregate type
    // width" limit.
    if ((total_bit_count + GetTypeBitCount(expr.type) >
         options_.max_width_aggregate_types)) {
      continue;
    }

    typed_exprs.push_back(expr);
    total_bit_count += GetTypeBitCount(expr.type);
    last_delaying_op =
        ComposeDelayingOps(last_delaying_op, expr.last_delaying_op);
    min_stage = std::max(min_stage, expr.min_stage);
  }

  // If only a single return value is selected, most of the time just return it
  // as a non-tuple value.
  if (RandomBool(0.8) && typed_exprs.size() == 1) {
    return typed_exprs[0];
  }

  auto [exprs, types, delaying_ops] = Unzip(typed_exprs);
  auto* tuple =
      module_->Make<XlsTuple>(fake_span_, exprs, /*has_trailing_comma=*/false);
  return TypedExpr{.expr = tuple,
                   .type = MakeTupleType(types),
                   .last_delaying_op = last_delaying_op,
                   .min_stage = min_stage};
}

// The return value for a proc's next function must be of the state type if a
// state is present.
absl::StatusOr<TypedExpr> AstGenerator::GenerateProcNextFunctionRetval(
    Context* ctx) {
  TypedExpr retval;
  if (proc_properties_.state_types.empty()) {
    // Return an empty tuple.
    retval = TypedExpr{
        .expr = module_->Make<XlsTuple>(fake_span_, std::vector<Expr*>{},
                                        /*has_trailing_comma=*/false),
        .type = MakeTupleType(std::vector<TypeAnnotation*>()),
    };
  } else {
    // A state is present, therefore the return value for a proc's next function
    // must be of the state type.
    XLS_ASSIGN_OR_RETURN(
        retval, ChooseEnvValue(&ctx->env, proc_properties_.state_types[0]));
  }
  retval.last_delaying_op = LastDelayingOp::kNone;
  for (auto& [name, value] : ctx->env) {
    if (IsToken(value.type)) {
      retval.min_stage = std::max(retval.min_stage, value.min_stage);
    }
  }
  return retval;
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCountedFor(Context* ctx) {
  // Right now just generates the 'identity' for loop.
  // TODO(meheff): Generate more interesting loop bodies.
  TypeAnnotation* ivar_type = MakeTypeAnnotation(false, 4);
  Number* zero = GenerateNumber(0, ivar_type);
  Number* trips = GenerateNumber(
      absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_, 1, 8), ivar_type);
  Expr* iterable = MakeRange(zero, trips);
  NameDef* x_def = MakeNameDef("x");
  NameDefTree* i_ndt = module_->Make<NameDefTree>(fake_span_, MakeNameDef("i"));
  NameDefTree* x_ndt = module_->Make<NameDefTree>(fake_span_, x_def);
  auto* name_def_tree = module_->Make<NameDefTree>(
      fake_span_, std::vector<NameDefTree*>{i_ndt, x_ndt});
  XLS_ASSIGN_OR_RETURN(TypedExpr e, ChooseEnvValueNotArray(&ctx->env));
  NameRef* body = MakeNameRef(x_def);

  // Randomly decide to use or not-use the type annotation on the loop.
  TupleTypeAnnotation* tree_type = nullptr;
  if (RandomBool(0.5)) {
    tree_type = MakeTupleType({ivar_type, e.type});
  }

  Statement* body_stmt = module_->Make<Statement>(body);
  auto* block = module_->Make<StatementBlock>(
      fake_span_, std::vector<Statement*>{body_stmt}, /*trailing_semi=*/false);
  For* for_ = module_->Make<For>(fake_span_, name_def_tree, tree_type, iterable,
                                 block, /*init=*/e.expr);
  return TypedExpr{for_, e.type};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateTupleOrIndex(Context* ctx) {
  CHECK(ctx != nullptr);
  if (RandomBool(0.5) && EnvContainsTuple(ctx->env)) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e,
                         ChooseEnvValueTuple(&ctx->env, /*min_size=*/1));
    auto* tuple_type = dynamic_cast<TupleTypeAnnotation*>(e.type);
    int64_t i = absl::Uniform<int64_t>(bit_gen_, 0, tuple_type->size());
    Number* index_expr = MakeNumber(i);
    return TypedExpr{
        .expr = module_->Make<TupleIndex>(fake_span_, e.expr, index_expr),
        .type = tuple_type->members()[i],
        .last_delaying_op = e.last_delaying_op,
        .min_stage = e.min_stage,
    };
  }

  std::vector<TypedExpr> members;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 1;
  int64_t total_bit_count = 0;
  XLS_ASSIGN_OR_RETURN(int64_t element_count, GenerateNaryOperandCount(ctx, 0));
  for (int64_t i = 0; i < element_count; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr e,
                         ChooseEnvValueNotContainingToken(&ctx->env));
    members.push_back(e);
    total_bit_count += GetTypeBitCount(e.type);
    last_delaying_op = ComposeDelayingOps(last_delaying_op, e.last_delaying_op);
    min_stage = std::max(min_stage, e.min_stage);
  }

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(total_bit_count));

  auto [exprs, types, delaying_ops] = Unzip(members);
  return TypedExpr{
      .expr = module_->Make<XlsTuple>(fake_span_, exprs,
                                      /*has_trailing_comma=*/false),
      .type = MakeTupleType(types),
      .last_delaying_op = last_delaying_op,
      .min_stage = min_stage,
  };
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateMap(int64_t call_depth,
                                                    Context* ctx) {
  // GenerateFunction(), in turn, can call GenerateMap(), so we need some way of
  // bounding the recursion. To limit explosion, return an recoverable error
  // with exponentially increasing probability depending on the call depth.
  if (RandomBool(1 - pow(10.0, -call_depth))) {
    return RecoverableError("Call depth too deep.");
  }

  std::string map_fn_name = GenSym();

  // Choose a random array from the environment and create a single-argument
  // function which takes an element of that array.
  XLS_ASSIGN_OR_RETURN(TypedExpr array, ChooseEnvValueArray(&ctx->env));
  ArrayTypeAnnotation* array_type = down_cast<ArrayTypeAnnotation*>(array.type);
  XLS_ASSIGN_OR_RETURN(
      AnnotatedFunction map_fn,
      GenerateFunction(map_fn_name, call_depth + 1,
                       /*param_types=*/
                       std::vector<AnnotatedType>({AnnotatedType{
                           .type = array_type->element_type(),
                           .last_delaying_op = array.last_delaying_op,
                           .min_stage = array.min_stage}})));

  // We put the function into the functions_ member so we know to emit it at the
  // top level of the module.
  functions_.push_back(map_fn.function);

  XLS_RETURN_IF_ERROR(
      VerifyAggregateWidth(GetTypeBitCount(map_fn.function->return_type()) *
                           GetArraySize(array_type)));

  TypeAnnotation* return_type =
      MakeArrayType(map_fn.function->return_type(), GetArraySize(array_type));

  NameRef* fn_ref = MakeNameRef(MakeNameDef(map_fn_name));
  auto* invocation =
      module_->Make<Invocation>(fake_span_, MakeBuiltinNameRef("map"),
                                std::vector<Expr*>{array.expr, fn_ref});
  return TypedExpr{.expr = invocation,
                   .type = return_type,
                   .last_delaying_op = map_fn.last_delaying_op,
                   .min_stage = map_fn.min_stage};
}

// TODO(vmirian): 11-16-2022 Add support to override the default parametric
// values.
absl::StatusOr<TypedExpr> AstGenerator::GenerateInvoke(int64_t call_depth,
                                                       Context* ctx) {
  // GenerateFunction(), in turn, can call GenerateInvoke(), so we need some way
  // of bounding the recursion. To limit explosion, return an recoverable error
  // with exponentially increasing probability depending on the call depth.
  if (RandomBool(1 - pow(10.0, -call_depth))) {
    return RecoverableError("Call depth too deep.");
  }

  std::string fn_name = GenSym();

  // When we're a nested call, 90% of the time make sure we have at least one
  // parameter.  Sometimes it's ok to try out what happens with 0 parameters.
  //
  // (Note we still pick a number of params with an expected value of 4 even
  // when 0 is permitted.)
  int64_t num_params =
      RandomIntWithExpectedValue(4,
                                 /*lower_limit=*/(RandomBool(0.90) ? 1 : 0));
  std::vector<Expr*> args;
  std::vector<AnnotatedType> param_types;
  args.reserve(num_params);
  param_types.reserve(num_params);
  for (int64_t i = 0; i < num_params; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr candidate, ChooseEnvValue(&ctx->env));
    args.push_back(candidate.expr);
    param_types.push_back({.type = candidate.type,
                           .last_delaying_op = candidate.last_delaying_op,
                           .min_stage = candidate.min_stage});
  }
  XLS_ASSIGN_OR_RETURN(AnnotatedFunction fn,
                       GenerateFunction(fn_name, call_depth + 1, param_types));

  NameRef* fn_ref = MakeNameRef(MakeNameDef(fn_name));
  auto* invocation = module_->Make<Invocation>(fake_span_, fn_ref, args);
  functions_.push_back(fn.function);

  return TypedExpr{.expr = invocation,
                   .type = fn.function->return_type(),
                   .last_delaying_op = fn.last_delaying_op,
                   .min_stage = fn.min_stage};
}

TypeAnnotation* AstGenerator::GenerateBitsType(
    std::optional<int64_t> max_width_bits_types) {
  int64_t max_width = options_.max_width_bits_types;
  if (max_width_bits_types.has_value()) {
    max_width = max_width_bits_types.value();
  }
  if (max_width <= 64 || RandomBool(0.9)) {
    // Once in a while generate a zero-width bits type.
    if (options_.emit_zero_width_bits_types && RandomBool(1. / 63)) {
      return MakeTypeAnnotation(/*is_signed=*/RandomBool(0.5), 0);
    }
    return GeneratePrimitiveType(max_width_bits_types);
  }
  // Generate a type wider than 64-bits. With smallish probability choose a
  // *really* wide type if the max_width_bits_types supports it, otherwise
  // choose a width up to 128 bits.
  if (max_width > 128 && RandomBool(1. / 9)) {
    max_width = 128;
  }
  return MakeTypeAnnotation(
      /*is_signed=*/RandomBool(0.5),
      /*width=*/absl::Uniform<int64_t>(absl::IntervalClosed, bit_gen_, 65,
                                       max_width));
}

TypeAnnotation* AstGenerator::GenerateType(
    int64_t nesting, std::optional<int64_t> max_width_bits_types,
    std::optional<int64_t> max_width_aggregate_types) {
  int64_t max_width = options_.max_width_aggregate_types;
  if (max_width_aggregate_types.has_value()) {
    max_width = max_width_aggregate_types.value();
  }
  if (RandomBool(0.1 * std::pow(2.0, -nesting))) {
    // Generate tuple type. Use a mean value of 3 elements so the tuple isn't
    // too big.
    int64_t total_width = 0;
    std::vector<TypeAnnotation*> element_types;
    for (int64_t i = 0; i < RandomIntWithExpectedValue(3, 0); ++i) {
      TypeAnnotation* element_type = GenerateType(nesting + 1);
      int64_t element_width = GetTypeBitCount(element_type);
      if (total_width + element_width > max_width) {
        break;
      }
      element_types.push_back(element_type);
      total_width += element_width;
    }
    return MakeTupleType(element_types);
  }
  if (RandomBool(0.1 * std::pow(2.0, -nesting))) {
    // Generate array type.
    TypeAnnotation* element_type =
        GenerateType(nesting + 1, max_width_bits_types, max_width);
    int64_t element_width = GetTypeBitCount(element_type);
    int64_t element_count = RandomIntWithExpectedValue(10, /*lower_limit=*/1);
    if (element_count * element_width > max_width) {
      element_count = std::max<int64_t>(1, max_width / element_width);
    }
    return MakeArrayType(element_type, element_count);
  }
  return GenerateBitsType(max_width_bits_types);
}

std::optional<TypedExpr> AstGenerator::ChooseEnvValueOptional(
    Env* env, const std::function<bool(const TypedExpr&)>& take) {
  if (take == nullptr) {
    // Fast path if there's no take function, we don't need to inspect/copy
    // things.
    if (env->empty()) {
      return std::nullopt;
    }
    int64_t index = absl::Uniform<int64_t>(bit_gen_, 0, env->size());
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
    return std::nullopt;
  }
  return *RandomChoice(absl::MakeConstSpan(choices), bit_gen_);
}

absl::StatusOr<TypedExpr> AstGenerator::ChooseEnvValue(
    Env* env, const std::function<bool(const TypedExpr&)>& take) {
  auto result = ChooseEnvValueOptional(env, take);
  if (!result.has_value()) {
    return RecoverableError(
        "No elements in the environment satisfy the predicate.");
  }
  return result.value();
}

std::vector<TypedExpr> AstGenerator::GatherAllValues(
    Env* env, const std::function<bool(const TypedExpr&)>& take) {
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
  if (RandomBool(0.5)) {
    rhs.expr = module_->Make<Cast>(fake_span_, rhs.expr, lhs.type);
    rhs.type = lhs.type;
  } else {
    lhs.expr = module_->Make<Cast>(fake_span_, lhs.expr, rhs.type);
    lhs.type = rhs.type;
  }
  return std::pair{lhs, rhs};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateUnop(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueBits(&ctx->env));
  UnopKind op = RandomChoice(
      absl::MakeConstSpan({UnopKind::kInvert, UnopKind::kNegate}), bit_gen_);
  return TypedExpr{
      .expr = module_->Make<Unop>(fake_span_, op, arg.expr),
      .type = arg.type,
      .last_delaying_op = arg.last_delaying_op,
      .min_stage = arg.min_stage,
  };
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
  CHECK_GE(*start, 0);
  CHECK_GE(*limit, *start);
  return {*start, *limit - *start};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBitSlice(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueBits(&ctx->env));
  int64_t bit_count = GetTypeBitCount(arg.type);
  // Slice LHS must be UBits.
  if (!IsUBits(arg.type)) {
    arg.expr = module_->Make<Cast>(fake_span_, arg.expr,
                                   MakeTypeAnnotation(false, bit_count));
  }
  enum class SliceType : std::uint8_t {
    kBitSlice,
    kWidthSlice,
    kDynamicSlice,
  };
  SliceType which = RandomChoice(
      absl::MakeConstSpan({SliceType::kBitSlice, SliceType::kWidthSlice,
                           SliceType::kDynamicSlice}),
      bit_gen_);
  std::optional<int64_t> start;
  std::optional<int64_t> limit;
  int64_t width = -1;
  while (true) {
    int64_t start_low = (which == SliceType::kWidthSlice) ? 0 : -bit_count - 1;
    bool should_have_start = RandomBool(0.5);
    start = should_have_start
                ? std::make_optional(absl::Uniform<int64_t>(
                      absl::IntervalClosed, bit_gen_, start_low, bit_count))
                : std::nullopt;
    bool should_have_limit = RandomBool(0.5);
    limit =
        should_have_limit
            ? std::make_optional(absl::Uniform<int64_t>(
                  absl::IntervalClosed, bit_gen_, -bit_count - 1, bit_count))
            : std::nullopt;
    width = ResolveBitSliceIndices(bit_count, start, limit).second;
    // Make sure we produce non-zero-width things.
    if (options_.emit_zero_width_bits_types || width > 0) {
      break;
    }
  }

  LastDelayingOp last_delaying_op = arg.last_delaying_op;
  int64_t min_stage = arg.min_stage;
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
      XLS_ASSIGN_OR_RETURN(TypedExpr start, ChooseEnvValueUBits(&ctx->env));
      rhs = module_->Make<WidthSlice>(fake_span_, start.expr,
                                      MakeTypeAnnotation(false, width));
      last_delaying_op =
          ComposeDelayingOps(last_delaying_op, start.last_delaying_op);
      min_stage = std::max(min_stage, start.min_stage);
      break;
    }
  }
  TypeAnnotation* type = MakeTypeAnnotation(false, width);
  auto* expr = module_->Make<Index>(fake_span_, arg.expr, rhs);
  return TypedExpr{.expr = expr,
                   .type = type,
                   .last_delaying_op = last_delaying_op,
                   .min_stage = min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBitwiseReduction(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(&ctx->env));
  std::string_view op = RandomChoice(
      absl::MakeConstSpan({"and_reduce", "or_reduce", "xor_reduce"}), bit_gen_);
  NameRef* callee = MakeBuiltinNameRef(std::string(op));
  TypeAnnotation* type = MakeTypeAnnotation(false, 1);
  return TypedExpr{.expr = module_->Make<Invocation>(
                       fake_span_, callee, std::vector<Expr*>{arg.expr}),
                   .type = type,
                   .last_delaying_op = arg.last_delaying_op,
                   .min_stage = arg.min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateCastBitsToArray(Context* ctx) {
  // Get a random bits-typed element from the environment.
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(&ctx->env));

  // Casts to arrays result in O(N) IR nodes for N-element arrays, so limit the
  // number of elements in the array to avoid explosion in the number of IR
  // nodes.
  constexpr int64_t kMaxArraySize = 64;

  // Next, find factors of the bit count and select one pair.
  int64_t bit_count = GetTypeBitCount(arg.type);

  if (bit_count == 0) {
    return RecoverableError("Cannot cast to array from zero-width bits type.");
  }

  std::vector<std::pair<int64_t, int64_t>> factors;
  for (int64_t i = 1; i < bit_count + 1; ++i) {
    if (bit_count % i == 0 && bit_count / i <= kMaxArraySize) {
      factors.push_back({i, bit_count / i});
    }
  }

  auto [element_size, array_size] = RandomChoice(factors, bit_gen_);
  TypeAnnotation* element_type = MakeTypeAnnotation(false, element_size);
  ArrayTypeAnnotation* outer_array_type =
      MakeArrayType(element_type, array_size);
  Cast* expr = module_->Make<Cast>(fake_span_, arg.expr, outer_array_type);
  return TypedExpr{.expr = expr,
                   .type = outer_array_type,
                   .last_delaying_op = arg.last_delaying_op,
                   .min_stage = arg.min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBitSliceUpdate(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr start, ChooseEnvValueUBits(&ctx->env));
  XLS_ASSIGN_OR_RETURN(TypedExpr update_value, ChooseEnvValueUBits(&ctx->env));

  auto* invocation = module_->Make<Invocation>(
      fake_span_, MakeBuiltinNameRef("bit_slice_update"),
      std::vector<Expr*>{arg.expr, start.expr, update_value.expr});
  return TypedExpr{
      .expr = invocation,
      .type = arg.type,
      .last_delaying_op =
          ComposeDelayingOps(arg.last_delaying_op, start.last_delaying_op,
                             update_value.last_delaying_op),
      .min_stage = std::max(std::max(arg.min_stage, start.min_stage),
                            update_value.min_stage)};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateArraySlice(Context* ctx) {
  // JIT/codegen for array_slice don't currently support zero-sized types
  auto is_not_zst = [this](ArrayTypeAnnotation* array_type) -> bool {
    return this->GetTypeBitCount(array_type) != 0;
  };

  XLS_ASSIGN_OR_RETURN(TypedExpr arg,
                       ChooseEnvValueArray(&ctx->env, is_not_zst));

  auto arg_type = dynamic_cast<ArrayTypeAnnotation*>(arg.type);
  CHECK_NE(arg_type, nullptr)
      << "Postcondition of ChooseEnvValueArray violated";

  XLS_ASSIGN_OR_RETURN(TypedExpr start, ChooseEnvValueUBits(&ctx->env));

  int64_t slice_width;

  if (RandomBool(0.5)) {
    slice_width = RandomIntWithExpectedValue(1.0, /*lower_limit=*/1);
  } else {
    slice_width = RandomIntWithExpectedValue(10.0, /*lower_limit=*/1);
  }

  XLS_RETURN_IF_ERROR(VerifyAggregateWidth(
      slice_width * GetTypeBitCount(arg_type->element_type())));

  std::vector<Expr*> width_array_elements = {module_->Make<Index>(
      fake_span_, arg.expr, GenerateNumber(0, MakeTypeAnnotation(false, 32)))};
  Array* width_expr = module_->Make<Array>(fake_span_, width_array_elements,
                                           /*has_ellipsis=*/true);
  TypeAnnotation* width_type = module_->Make<ArrayTypeAnnotation>(
      fake_span_, arg_type->element_type(), MakeNumber(slice_width));
  width_expr->set_type_annotation(width_type);

  TypedExpr width{width_expr, width_type};
  auto* invocation = module_->Make<Invocation>(
      fake_span_, MakeBuiltinNameRef("slice"),
      std::vector<Expr*>{arg.expr, start.expr, width.expr});
  return TypedExpr{
      .expr = invocation,
      .type = width_type,
      .last_delaying_op =
          ComposeDelayingOps(arg.last_delaying_op, start.last_delaying_op),
      .min_stage = std::max(arg.min_stage, start.min_stage),
  };
}

absl::StatusOr<int64_t> AstGenerator::GenerateNaryOperandCount(
    Context* ctx, int64_t lower_limit) {
  int64_t result = std::min(RandomIntWithExpectedValue(4, lower_limit),
                            static_cast<int64_t>(ctx->env.size()));
  if (result < lower_limit) {
    return RecoverableError("lower limit not satisfied.");
  }
  return result;
}

namespace {

enum OpChoice : std::uint8_t {
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
  kMatchOp,
  kConcat,
  kCountedFor,
  kGate,
  kInvoke,
  kJoinOp,
  kLogical,
  kMap,
  kNumber,
  kOneHotSelectBuiltin,
  kPartialProduct,
  kPrioritySelectBuiltin,
  kSignExtendBuiltin,
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
    case kMatchOp:
      return 2;
    case kConcat:
      return 5;
    case kCountedFor:
      return 1;
    case kGate:
      return 1;
    case kInvoke:
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
    case kSignExtendBuiltin:
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
  LOG(FATAL) << "Invalid op choice: " << static_cast<int64_t>(op);
}

absl::discrete_distribution<int>& GetOpDistribution(bool generate_proc) {
  auto dist = [&](bool generate_proc) {
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
    return new absl::discrete_distribution<int>(tmp.begin(), tmp.end());
  };
  static absl::discrete_distribution<int>& func_dist = *dist(false);
  static absl::discrete_distribution<int>& proc_dist = *dist(true);
  if (generate_proc) {
    return proc_dist;
  }
  return func_dist;
}

OpChoice ChooseOp(absl::BitGenRef bit_gen, bool generate_proc) {
  return static_cast<OpChoice>(GetOpDistribution(generate_proc)(bit_gen));
}

}  // namespace

absl::StatusOr<TypedExpr> AstGenerator::GenerateExpr(int64_t call_depth,
                                                     Context* ctx) {
  absl::StatusOr<TypedExpr> generated = RecoverableError("Not yet generated.");
  while (IsRecoverableError(generated.status())) {
    switch (ChooseOp(bit_gen_, ctx->is_generating_proc)) {
      case kArray:
        generated = GenerateArray(ctx);
        break;
      case kArrayIndex:
        generated = GenerateArrayIndex(ctx);
        break;
      case kArrayUpdate:
        generated = GenerateArrayUpdate(ctx);
        break;
      case kArraySlice:
        generated = GenerateArraySlice(ctx);
        break;
      case kCountedFor:
        generated = GenerateCountedFor(ctx);
        break;
      case kTupleOrIndex:
        generated = GenerateTupleOrIndex(ctx);
        break;
      case kConcat:
        generated = GenerateConcat(ctx);
        break;
      case kBinop:
        generated = GenerateBinop(ctx);
        break;
      case kCompareOp:
        generated = GenerateCompare(ctx);
        break;
      case kCompareArrayOp:
        generated = GenerateCompareArray(ctx);
        break;
      case kCompareTupleOp:
        generated = GenerateCompareTuple(ctx);
        break;
      case kMatchOp:
        generated = GenerateMatch(ctx);
        break;
      case kShiftOp:
        generated = GenerateShift(ctx);
        break;
      case kLogical:
        generated = GenerateLogicalOp(ctx);
        break;
      case kGate:
        if (!options_.emit_gate) {
          continue;
        }
        generated = GenerateGate(ctx);
        break;
      case kChannelOp:
        generated = GenerateChannelOp(ctx);
        break;
      case kJoinOp:
        generated = GenerateJoinOp(ctx);
        break;
      case kMap:
        generated = GenerateMap(call_depth, ctx);
        break;
      case kInvoke:
        generated = GenerateInvoke(call_depth, ctx);
        break;
      case kUnop:
        generated = GenerateUnop(ctx);
        break;
      case kUnopBuiltin:
        generated = GenerateUnopBuiltin(ctx);
        break;
      case kOneHotSelectBuiltin:
        generated = GenerateOneHotSelectBuiltin(ctx);
        break;
      case kPartialProduct:
        generated = GeneratePartialProductDeterministicGroup(ctx);
        break;
      case kPrioritySelectBuiltin:
        generated = GeneratePrioritySelectBuiltin(ctx);
        break;
      case kSignExtendBuiltin:
        generated = GenerateSignExtendBuiltin(ctx);
        break;
      case kNumber:
        generated = GenerateNumberWithType();
        break;
      case kBitwiseReduction:
        generated = GenerateBitwiseReduction(ctx);
        break;
      case kBitSlice:
        generated = GenerateBitSlice(ctx);
        break;
      case kCastToBitsArray:
        generated = GenerateCastBitsToArray(ctx);
        break;
      case kBitSliceUpdate:
        generated = GenerateBitSliceUpdate(ctx);
        break;
      case kEndSentinel:
        LOG(FATAL) << "Should not have selected end sentinel";
    }
  }

  if (generated.ok()) {
    // Do some opportunistic checking that our result types are staying within
    // requested parameters.
    if (IsBits(generated->type)) {
      XLS_RET_CHECK_LE(GetTypeBitCount(generated->type),
                       options_.max_width_bits_types)
          << absl::StreamFormat("Bits-typed expression is too wide: %s",
                                generated->expr->ToString());
    } else if (IsArray(generated->type) || IsTuple(generated->type)) {
      XLS_RET_CHECK_LE(GetTypeBitCount(generated->type),
                       options_.max_width_aggregate_types)
          << absl::StreamFormat("Aggregate-typed expression is too wide: %s",
                                generated->expr->ToString());
    }
  }
  return generated;
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateUnopBuiltin(Context* ctx) {
  XLS_ASSIGN_OR_RETURN(TypedExpr arg, ChooseEnvValueUBits(&ctx->env));
  enum UnopBuiltin : std::uint8_t {
    kClz,
    kCtz,
    kRev,
    kDecode,
    kEncode,
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
      case kDecode:
        return "decode";
      case kEncode:
        return "encode";
      case kOneHot:
        return "one_hot";
    }
    LOG(FATAL) << "Invalid kind: " << kind;
  };

  std::vector<UnopBuiltin> choices = {kRev, kDecode};
  // Since one_hot, clz, and ctz adds a bit, only use it when we have head room
  // beneath max_width_bits_types to add another bit.
  if (GetTypeBitCount(arg.type) < options_.max_width_bits_types) {
    choices.push_back(kOneHot);
    choices.push_back(kClz);
    choices.push_back(kCtz);
  }
  // Since encode outputs an empty object on inputs <= 1 bit wide, only use it
  // when we have at least 2 bits in the input.
  if (GetTypeBitCount(arg.type) >= 2) {
    choices.push_back(kEncode);
  }

  Invocation* invocation = nullptr;
  auto which = RandomChoice(choices, bit_gen_);
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
    case kDecode: {
      int64_t max_decode_width = options_.max_width_bits_types;
      // decode currently requires we ask for no more bits than `arg` could
      // possibly code for.
      if (GetTypeBitCount(arg.type) < 63) {
        max_decode_width =
            std::min(max_decode_width, int64_t{1} << GetTypeBitCount(arg.type));
      }
      result_bits = std::min(
          max_decode_width,
          RandomIntWithExpectedValue(max_decode_width, /*lower_limit=*/1));
      invocation = module_->Make<Invocation>(
          fake_span_, name_ref, std::vector<Expr*>{arg.expr},
          std::vector<ExprOrType>{MakeTypeAnnotation(false, result_bits)});
      break;
    }
    case kEncode:
      invocation = module_->Make<Invocation>(fake_span_, name_ref,
                                             std::vector<Expr*>{arg.expr});
      result_bits = CeilOfLog2(GetTypeBitCount(arg.type));
      break;
    case kOneHot: {
      bool lsb_or_msb = RandomBool(0.5);
      invocation = module_->Make<Invocation>(
          fake_span_, name_ref,
          std::vector<Expr*>{arg.expr, MakeBool(lsb_or_msb)});
      result_bits = GetTypeBitCount(arg.type) + 1;
      break;
    }
  }

  TypeAnnotation* result_type = MakeTypeAnnotation(false, result_bits);
  return TypedExpr{.expr = invocation,
                   .type = result_type,
                   .last_delaying_op = arg.last_delaying_op,
                   .min_stage = arg.min_stage};
}

absl::StatusOr<TypedExpr> AstGenerator::GenerateBody(int64_t call_depth,
                                                     Context* ctx) {
  // To produce 'interesting' behavior (samples that contain computation/work),
  // the environment of the initial call depth should not be empty, so we can
  // draw references from the environment. The environment can be empty for
  // subsequent call depths.
  XLS_RET_CHECK(call_depth > 0 || (call_depth == 0 && !ctx->env.empty()));

  // Make non-top-level functions smaller; these means were chosen
  // arbitrarily, and can be adjusted as we see fit later.
  int64_t body_size = RandomIntWithExpectedValue(call_depth == 0 ? 22.0 : 2.87,
                                                 /*lower_limit=*/1);

  std::vector<Statement*> statements;
  statements.reserve(body_size + 1);
  for (int64_t i = 0; i < body_size; ++i) {
    XLS_ASSIGN_OR_RETURN(TypedExpr rhs, GenerateExpr(call_depth, ctx));

    // Add the expression into the environment with a unique name.
    std::string identifier = GenSym();

    // What we place into the environment is a NameRef that refers to this RHS
    // value -- this way rules will pick up the expression names instead of
    // picking up the expression ASTs directly (which would cause duplication).
    auto* name_def = module_->Make<NameDef>(fake_span_, identifier, rhs.expr);
    auto* name_ref = MakeNameRef(name_def);

    // For tuples, this generates `let x0: (tuple_type) = tuple_value;`
    // TODO: https://github.com/google/xls/issues/1459 - skip always generating
    // the full tuple assignment (note, it is currently needed for the
    // (token, type) case).
    statements.push_back(module_->Make<Statement>(module_->Make<Let>(
        fake_span_,
        /*name_def_tree=*/module_->Make<NameDefTree>(fake_span_, name_def),
        /*type=*/rhs.type,
        /*rhs=*/rhs.expr,
        /*is_const=*/false)));
    ctx->env[identifier] = TypedExpr{.expr = name_ref,
                                     .type = rhs.type,
                                     .last_delaying_op = rhs.last_delaying_op,
                                     .min_stage = rhs.min_stage};

    if (IsTuple(rhs.type)) {
      GenerateTupleAssignment(name_ref, rhs, ctx, statements);
    }
  }

  // Done building up the body; finish with the retval.
  XLS_ASSIGN_OR_RETURN(TypedExpr retval,
                       ctx->is_generating_proc
                           ? GenerateProcNextFunctionRetval(ctx)
                           : GenerateRetval(ctx));
  statements.push_back(module_->Make<Statement>(retval.expr));

  auto* block = module_->Make<StatementBlock>(fake_span_, statements,
                                              /*trailing_semi=*/false);
  return TypedExpr{.expr = block,
                   .type = retval.type,
                   .last_delaying_op = retval.last_delaying_op,
                   .min_stage = retval.min_stage};
}

void AstGenerator::GenerateTupleAssignment(
    NameRef* name_ref, TypedExpr& rhs, Context* ctx,
    std::vector<Statement*>& statements) {
  auto* tuple_type = dynamic_cast<TupleTypeAnnotation*>(rhs.type);
  if (tuple_type->empty()) {
    return;
  }
  if (IsToken(tuple_type->members()[0])) {
    // Unpack result tuples from channel operations and place them in the
    // environment to be easily accessible, creating more interesting
    // behavior. Currently, results from operations that are tuples and start
    // with a token are assumed to be channel operations.
    for (int64_t index = 0; index < tuple_type->members().size(); ++index) {
      std::string member_identifier = GenSym();
      auto* member_name_def = module_->Make<NameDef>(
          fake_span_, member_identifier, /*definer=*/nullptr);
      auto* member_name_ref = MakeNameRef(member_name_def);
      statements.push_back(module_->Make<Statement>(module_->Make<Let>(
          fake_span_,
          /*name_def_tree=*/
          module_->Make<NameDefTree>(fake_span_, member_name_def),
          /*type=*/tuple_type->members()[index],
          /*rhs=*/
          module_->Make<TupleIndex>(fake_span_, name_ref, MakeNumber(index)),
          /*is_const=*/false)));
      ctx->env[member_identifier] =
          TypedExpr{.expr = member_name_ref,
                    .type = tuple_type->members()[index],
                    .last_delaying_op = rhs.last_delaying_op,
                    .min_stage = rhs.min_stage};
    }
    return;
  }

  // Regular tuple; 50% of the time, destructure it:
  // let (a, b, c): (tuple_type) = tuple_value;
  if (RandomBool(0.5)) {
    TypeAnnotation* rhs_type = rhs.type;
    if (RandomBool(0.5)) {
      // Half the time, let it deduce the type instead of specifying it.
      rhs_type = nullptr;
    }

    // TODO: https://github.com/google/xls/issues/1459 - handle tuples of
    // tuples.
    std::vector<NameDefTree*> name_defs;
    bool has_rest_of_tuple = false;
    for (int64_t index = 0; index < tuple_type->members().size(); ++index) {
      if (RandomBool(0.1)) {
        // Replace this name with a wildcard.
        WildcardPattern* wc = module_->Make<WildcardPattern>(fake_span_);
        name_defs.push_back(module_->Make<NameDefTree>(fake_span_, wc));
        continue;
      }

      if (rhs_type == nullptr && !has_rest_of_tuple && RandomBool(0.1)) {
        has_rest_of_tuple = true;
        // Insert a "rest of tuple", but we might keep this name.
        RestOfTuple* rest = module_->Make<RestOfTuple>(fake_span_);
        name_defs.push_back(module_->Make<NameDefTree>(fake_span_, rest));
        // Also, jump forward a random # of elements
        auto jump_forward = RandomIntWithExpectedValue(
            /*expected_value=*/(tuple_type->members().size() - index) / 2.0,
            /*lower_limit=*/0);
        if (jump_forward > 0) {
          index += jump_forward;
          // We could keep this name, or skip it, but for now, skip it.
          continue;
        }
        // The jump forward is 0; we'll keep this name.
      }

      std::string member_identifier = GenSym();
      auto* member_name_def = module_->Make<NameDef>(
          fake_span_, member_identifier, /*definer=*/nullptr);
      auto* member_name_ref = MakeNameRef(member_name_def);
      ctx->env[member_identifier] =
          TypedExpr{.expr = member_name_ref,
                    .type = tuple_type->members()[index],
                    .last_delaying_op = rhs.last_delaying_op,
                    .min_stage = rhs.min_stage};
      name_defs.push_back(
          module_->Make<NameDefTree>(fake_span_, member_name_def));
    }

    statements.push_back(module_->Make<Statement>(module_->Make<Let>(
        fake_span_,
        /*name_def_tree=*/module_->Make<NameDefTree>(fake_span_, name_defs),
        /*type=*/rhs_type,
        /*rhs=*/rhs.expr,
        /*is_const=*/false)));
  }
}

absl::StatusOr<AnnotatedFunction> AstGenerator::GenerateFunction(
    const std::string& name, int64_t call_depth,
    absl::Span<const AnnotatedType> param_types) {
  Context context{.is_generating_proc = false};

  std::vector<Param*> params;
  std::vector<AnnotatedParam> annotated_params;
  params.reserve(param_types.size());
  annotated_params.reserve(param_types.size());
  for (const AnnotatedType& param_type : param_types) {
    annotated_params.push_back(GenerateParam(param_type));
    params.push_back(annotated_params.back().param);
  }
  for (AnnotatedParam param : annotated_params) {
    context.env[param.param->identifier()] =
        TypedExpr{.expr = MakeNameRef(param.param->name_def()),
                  .type = param.param->type_annotation(),
                  .last_delaying_op = param.last_delaying_op,
                  .min_stage = param.min_stage};
  }

  // When we're not the main function, 10% of the time put some parametrics on
  // the function.
  std::vector<ParametricBinding*> parametric_bindings;
  if (call_depth != 0 && RandomBool(0.10)) {
    parametric_bindings = GenerateParametricBindings(
        RandomIntWithExpectedValue(2, /*lower_limit=*/1));
  }
  for (ParametricBinding* pb : parametric_bindings) {
    context.env[pb->identifier()] =
        TypedExpr{MakeNameRef(pb->name_def()), pb->type_annotation()};
  }

  XLS_ASSIGN_OR_RETURN(TypedExpr retval, GenerateBody(call_depth, &context));
  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Statement* retval_statement = module_->Make<Statement>(retval.expr);
  auto* block = module_->Make<StatementBlock>(
      fake_span_, std::vector<Statement*>{retval_statement},
      /*trailing_semi=*/false);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/parametric_bindings,
      /*params=*/params,
      /*return_type=*/retval.type, block, FunctionTag::kNormal,
      /*is_public=*/false);
  name_def->set_definer(f);

  return AnnotatedFunction{.function = f,
                           .last_delaying_op = retval.last_delaying_op,
                           .min_stage = retval.min_stage};
}

absl::StatusOr<int64_t> AstGenerator::GenerateFunctionInModule(
    const std::string& name) {
  // If we're the main function we have to have at least one parameter,
  // because some Generate* methods expect to be able to draw from a non-empty
  // env.
  //
  // TODO(https://github.com/google/xls/issues/475): Cleanup to make
  // productions that are ok with empty env separate from those which require
  // a populated env.
  int64_t num_params = RandomIntWithExpectedValue(4, /*lower_limit=*/1);
  std::vector<AnnotatedType> param_types(num_params);
  for (int64_t i = 0; i < num_params; ++i) {
    param_types[i] = {.type = GenerateType()};
  }
  XLS_ASSIGN_OR_RETURN(AnnotatedFunction f,
                       GenerateFunction(name, /*call_depth=*/0, param_types));
  for (auto& item : constants_) {
    XLS_RETURN_IF_ERROR(
        module_->AddTop(item.second, /*make_collision_error=*/nullptr));
  }
  for (auto& item : type_aliases_) {
    XLS_RETURN_IF_ERROR(
        module_->AddTop(item, /*make_collision_error=*/nullptr));
  }
  for (auto& item : functions_) {
    XLS_RETURN_IF_ERROR(
        module_->AddTop(item, /*make_collision_error=*/nullptr));
  }
  XLS_RETURN_IF_ERROR(
      module_->AddTop(f.function, /*make_collision_error=*/nullptr));
  return f.min_stage;
}

absl::StatusOr<Function*> AstGenerator::GenerateProcConfigFunction(
    std::string name, absl::Span<Param* const> proc_params) {
  std::vector<Param*> params;
  std::vector<Expr*> tuple_members;
  std::vector<TypeAnnotation*> tuple_member_types;
  for (Param* proc_param : proc_params) {
    params.push_back(module_->Make<Param>(proc_param->name_def(),
                                          proc_param->type_annotation()));
    tuple_members.push_back(MakeNameRef(proc_param->name_def()));
    tuple_member_types.push_back(proc_param->type_annotation());
  }
  TupleTypeAnnotation* ret_tuple_type =
      module_->Make<TupleTypeAnnotation>(fake_span_, tuple_member_types);
  XlsTuple* ret_tuple = module_->Make<XlsTuple>(fake_span_, tuple_members,
                                                /*has_trailing_comma=*/false);
  Statement* ret_stmt = module_->Make<Statement>(ret_tuple);
  auto* block = module_->Make<StatementBlock>(
      fake_span_, std::vector<Statement*>{ret_stmt}, /*trailing_semi=*/false);
  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/params,
      /*return_type=*/ret_tuple_type, block, FunctionTag::kProcConfig,
      /*is_public=*/false);
  name_def->set_definer(f);
  return f;
}

absl::StatusOr<AnnotatedFunction> AstGenerator::GenerateProcNextFunction(
    std::string name) {
  Context context{.is_generating_proc = true};

  std::vector<Param*> params;
  TypeAnnotation* state_param_type = nullptr;
  if (options_.emit_stateless_proc) {
    state_param_type = MakeTupleType({});
  } else {
    state_param_type = GenerateType();
  }
  params.insert(params.end(), GenerateParam({.type = state_param_type}).param);
  proc_properties_.state_types.push_back(params.back()->type_annotation());

  for (Param* param : params) {
    context.env[param->identifier()] =
        TypedExpr{MakeNameRef(param->name_def()), param->type_annotation()};
  }

  XLS_ASSIGN_OR_RETURN(TypedExpr retval, GenerateBody(0, &context));

  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);
  Statement* retval_stmt = module_->Make<Statement>(retval.expr);
  auto* block = module_->Make<StatementBlock>(
      fake_span_, std::vector<Statement*>{retval_stmt},
      /*trailing_semi=*/false);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/params,
      /*return_type=*/retval.type, block, FunctionTag::kProcNext,
      /*is_public=*/false);
  name_def->set_definer(f);

  return AnnotatedFunction{.function = f,
                           .last_delaying_op = retval.last_delaying_op,
                           .min_stage = retval.min_stage};
}

absl::StatusOr<Function*> AstGenerator::GenerateProcInitFunction(
    std::string_view name, TypeAnnotation* return_type) {
  NameDef* name_def = module_->Make<NameDef>(fake_span_, std::string(name),
                                             /*definer=*/nullptr);

  XLS_ASSIGN_OR_RETURN(
      Expr * init_constant,
      GenerateDslxConstant(bit_gen_, module_.get(), return_type));
  Statement* s = module_->Make<Statement>(init_constant);
  auto* b =
      module_->Make<StatementBlock>(fake_span_, std::vector<Statement*>{s},
                                    /*trailing_semi=*/false);
  Function* f = module_->Make<Function>(
      fake_span_, name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(),
      /*params=*/std::vector<Param*>(),
      /*return_type=*/return_type, b, FunctionTag::kProcInit,
      /*is_public=*/false);
  name_def->set_definer(f);
  return f;
}

absl::StatusOr<AnnotatedProc> AstGenerator::GenerateProc(
    const std::string& name) {
  XLS_ASSIGN_OR_RETURN(AnnotatedFunction next_function,
                       GenerateProcNextFunction("next"));

  XLS_ASSIGN_OR_RETURN(
      Function * config_function,
      GenerateProcConfigFunction("config", proc_properties_.config_params));

  CHECK_EQ(proc_properties_.state_types.size(), 1);
  XLS_ASSIGN_OR_RETURN(
      Function * init_fn,
      GenerateProcInitFunction(absl::StrCat(name, ".init"),
                               proc_properties_.state_types[0]));

  NameDef* name_def =
      module_->Make<NameDef>(fake_span_, name, /*definer=*/nullptr);

  std::vector<ProcStmt> proc_stmts;
  proc_stmts.reserve(proc_properties_.members.size());
  for (ProcMember* member : proc_properties_.members) {
    proc_stmts.push_back(member);
  }

  ProcLikeBody body = {
      .stmts = proc_stmts,
      .config = config_function,
      .next = next_function.function,
      .init = init_fn,
      .members = proc_properties_.members,
  };
  Proc* proc = module_->Make<Proc>(
      fake_span_, /*body_span=*/fake_span_, name_def,
      /*parametric_bindings=*/std::vector<ParametricBinding*>(), body,
      /*is_public=*/false);
  name_def->set_definer(proc);
  return AnnotatedProc{.proc = proc, .min_stages = next_function.min_stage};
}

absl::StatusOr<int64_t> AstGenerator::GenerateProcInModule(
    const std::string& proc_name) {
  XLS_ASSIGN_OR_RETURN(AnnotatedProc proc, GenerateProc(proc_name));
  for (auto& item : constants_) {
    XLS_RETURN_IF_ERROR(
        module_->AddTop(item.second, /*make_collision_error=*/nullptr));
  }
  for (auto& item : type_aliases_) {
    XLS_RETURN_IF_ERROR(
        module_->AddTop(item, /*make_collision_error=*/nullptr));
  }
  for (auto& item : functions_) {
    XLS_RETURN_IF_ERROR(
        module_->AddTop(item, /*make_collision_error=*/nullptr));
  }
  XLS_RETURN_IF_ERROR(
      module_->AddTop(proc.proc, /*make_collision_error=*/nullptr));
  return proc.min_stages;
}

absl::StatusOr<AnnotatedModule> AstGenerator::Generate(
    const std::string& top_entity_name, const std::string& module_name) {
  module_ = std::make_unique<Module>(module_name, /*fs_path=*/std::nullopt,
                                     file_table_);
  int64_t min_stages = 1;
  if (options_.generate_proc) {
    XLS_ASSIGN_OR_RETURN(min_stages, GenerateProcInModule(top_entity_name));
  } else {
    XLS_ASSIGN_OR_RETURN(min_stages, GenerateFunctionInModule(top_entity_name));
  }
  return AnnotatedModule{.module = std::move(module_),
                         .min_stages = min_stages};
}

AstGenerator::AstGenerator(AstGeneratorOptions options, absl::BitGenRef bit_gen,
                           FileTable& file_table)
    : bit_gen_(bit_gen),
      options_(options),
      file_table_(file_table),
      fake_pos_(Fileno(0), 0, 0),
      fake_span_(fake_pos_, fake_pos_) {}

}  // namespace xls::dslx
