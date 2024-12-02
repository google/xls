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

#include "xls/dslx/bytecode/bytecode_emitter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/symbolized_stacktrace.h"
#include "xls/common/visitor.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/make_value_format_descriptor.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/format_strings.h"

// TODO(rspringer): 2022-03-01: Verify that, for all valid programs (or at least
// some subset that we test), interpretation terminates with only a single value
// on the stack (I believe that should be the case for any valid program).

namespace xls::dslx {
namespace {

// Determines the owning proc for `node` by walking the parent links.
const Proc* GetContainingProc(const AstNode* node) {
  AstNode* proc_node = node->parent();
  while (dynamic_cast<const Proc*>(proc_node) == nullptr) {
    proc_node = proc_node->parent();
    CHECK(proc_node != nullptr);
  }
  return dynamic_cast<const Proc*>(proc_node);
}

// Find concrete type of channel's payload.
absl::StatusOr<std::unique_ptr<Type>> GetChannelPayloadType(
    const TypeInfo* type_info, const Expr* channel) {
  std::optional<Type*> type = type_info->GetItem(channel);

  if (!type.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Could not retrieve type of channel %s", channel->ToString()));
  }

  ChannelType* channel_type = dynamic_cast<ChannelType*>(type.value());
  if (channel_type == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Channel %s type is not of type channel", channel->ToString()));
  }

  XLS_RET_CHECK(!channel_type->payload_type().IsMeta());
  return channel_type->payload_type().CloneToUnique();
}

absl::StatusOr<Bytecode::ChannelData> CreateChannelData(
    const Expr* channel, const TypeInfo* type_info,
    FormatPreference format_preference) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> channel_payload_type,
                       GetChannelPayloadType(type_info, channel));

  XLS_ASSIGN_OR_RETURN(ValueFormatDescriptor struct_fmt_desc,
                       MakeValueFormatDescriptor(*channel_payload_type.get(),
                                                 format_preference));

  const Proc* proc_node = GetContainingProc(channel);
  std::string_view proc_name = proc_node->identifier();

  return Bytecode::ChannelData(
      absl::StrFormat("%s::%s", proc_name, channel->ToString()),
      std::move(channel_payload_type), std::move(struct_fmt_desc));
}

absl::StatusOr<ValueFormatDescriptor> ExprToValueFormatDescriptor(
    const Expr* expr, const TypeInfo* type_info,
    FormatPreference field_preference) {
  std::optional<Type*> maybe_type = type_info->GetItem(expr);
  XLS_RET_CHECK(maybe_type.has_value());
  XLS_RET_CHECK(maybe_type.value() != nullptr);
  return MakeValueFormatDescriptor(*maybe_type.value(), field_preference);
}

std::optional<ValueFormatDescriptor> GetFormatDescriptorFromNumber(
    const Number* node) {
  std::string text = node->ToStringNoType();
  if (absl::StartsWith(text, "0x")) {
    return ValueFormatDescriptor::MakeLeafValue(FormatPreference::kHex);
  }
  if (absl::StartsWith(text, "0b")) {
    return ValueFormatDescriptor::MakeLeafValue(FormatPreference::kBinary);
  }
  BuiltinTypeAnnotation* builtin_type =
      dynamic_cast<BuiltinTypeAnnotation*>(node->type_annotation());
  if (builtin_type == nullptr) {
    return std::nullopt;
  }
  return ValueFormatDescriptor::MakeLeafValue(
      builtin_type->GetSignedness().value()
          ? FormatPreference::kSignedDecimal
          : FormatPreference::kUnsignedDecimal);
}

}  // namespace

BytecodeEmitter::BytecodeEmitter(
    ImportData* import_data, const TypeInfo* type_info,
    const std::optional<ParametricEnv>& caller_bindings,
    const BytecodeEmitterOptions& options)
    : import_data_(import_data),
      type_info_(type_info),
      caller_bindings_(caller_bindings),
      options_(options) {}

BytecodeEmitter::~BytecodeEmitter() = default;

absl::Status BytecodeEmitter::Init(const Function& f) {
  for (const auto* param : f.params()) {
    namedef_to_slot_[param->name_def()] = next_slotno_++;
  }

  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::Emit(ImportData* import_data, const TypeInfo* type_info,
                      const Function& f,
                      const std::optional<ParametricEnv>& caller_bindings,
                      const BytecodeEmitterOptions& options) {
  return EmitProcNext(import_data, type_info, f, caller_bindings,
                      /*proc_members=*/{}, options);
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::EmitProcNext(
    ImportData* import_data, const TypeInfo* type_info, const Function& f,
    const std::optional<ParametricEnv>& caller_bindings,
    const std::vector<NameDef*>& proc_members,
    const BytecodeEmitterOptions& options) {
  XLS_RET_CHECK(type_info != nullptr);

  BytecodeEmitter emitter(import_data, type_info, caller_bindings, options);
  for (const NameDef* name_def : proc_members) {
    emitter.namedef_to_slot_[name_def] = emitter.next_slotno_++;
  }
  XLS_RETURN_IF_ERROR(emitter.Init(f));
  XLS_RETURN_IF_ERROR(f.body()->AcceptExpr(&emitter));

  return BytecodeFunction::Create(f.owner(), &f, type_info,
                                  std::move(emitter.bytecode_));
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::EmitExpression(
    ImportData* import_data, const TypeInfo* type_info, const Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env,
    const std::optional<ParametricEnv>& caller_bindings,
    const BytecodeEmitterOptions& options) {
  BytecodeEmitter emitter(import_data, type_info, caller_bindings, options);

  XLS_ASSIGN_OR_RETURN(std::vector<const NameDef*> name_defs,
                       CollectReferencedUnder(expr));
  absl::flat_hash_map<std::string, const NameDef*> identifier_to_name_def;
  for (const NameDef* name_def : name_defs) {
    identifier_to_name_def[name_def->identifier()] = name_def;
  }

  for (const auto& [identifier, name_def] : identifier_to_name_def) {
    AstNode* definer = name_def->definer();
    if (dynamic_cast<Function*>(definer) != nullptr ||
        dynamic_cast<Import*>(definer) != nullptr) {
      continue;
    }

    if (!env.contains(identifier)) {
      continue;
    }

    int64_t slot_index = emitter.next_slotno_++;
    emitter.namedef_to_slot_[name_def] = slot_index;
    emitter.Add(Bytecode::MakeLiteral(expr->span(), env.at(identifier)));
    emitter.Add(
        Bytecode::MakeStore(expr->span(), Bytecode::SlotIndex(slot_index)));
  }

  XLS_RETURN_IF_ERROR(expr->AcceptExpr(&emitter));

  return BytecodeFunction::Create(expr->owner(), /*source_fn=*/nullptr,
                                  type_info, std::move(emitter.bytecode_));
}

absl::Status BytecodeEmitter::HandleArray(const Array* node) {
  if (type_info_->IsKnownConstExpr(node)) {
    auto const_expr_or = type_info_->GetConstExpr(node);
    XLS_RET_CHECK_OK(const_expr_or.status());
    VLOG(5) << absl::StreamFormat(
        "BytecodeEmitter::HandleArray; node %s is known constexpr: %s",
        node->ToString(), const_expr_or.value().ToString());
    Add(Bytecode::MakeLiteral(node->span(), const_expr_or.value()));
    return absl::OkStatus();
  }

  size_t num_members = node->members().size();
  for (auto* member : node->members()) {
    XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
  }

  // If we've got an ellipsis, then repeat the last element until we reach the
  // full array size.
  if (node->has_ellipsis()) {
    XLS_RET_CHECK(!node->members().empty());
    XLS_ASSIGN_OR_RETURN(ArrayType * array_type,
                         type_info_->GetItemAs<ArrayType>(node));
    VLOG(5) << "Bytecode::HandleArray; emitting ellipsis for array type: "
            << *array_type;
    const TypeDim& dim = array_type->size();
    XLS_ASSIGN_OR_RETURN(num_members, dim.GetAsInt64());
    int64_t remaining_members = num_members - node->members().size();
    for (int i = 0; i < remaining_members; i++) {
      XLS_RETURN_IF_ERROR(node->members().back()->AcceptExpr(this));
    }
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateArray,
                               Bytecode::NumElements(num_members)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleAttr(const Attr* node) {
  // Will place a struct instance on the stack.
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));

  // Now we need the index of the attr NameRef in the struct def.
  XLS_ASSIGN_OR_RETURN(StructType * struct_type,
                       type_info_->GetItemAs<StructType>(node->lhs()));
  XLS_ASSIGN_OR_RETURN(int64_t member_index,
                       struct_type->GetMemberIndex(node->attr()));

  VLOG(10) << "BytecodeEmitter::HandleAttr; member_index: " << member_index;

  // This indexing literal needs to be unsigned since InterpValue::Index
  // requires an unsigned value.
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeU64(member_index)));
  Add(Bytecode::MakeTupleIndex(node->span()));
  return absl::OkStatus();
}

absl::StatusOr<bool> BytecodeEmitter::IsBitsTypeNodeSigned(
    const AstNode* node) const {
  XLS_RET_CHECK(type_info_ != nullptr);
  std::optional<const Type*> maybe_type = type_info_->GetItem(node);
  XLS_RET_CHECK(maybe_type.has_value()) << "node: " << node->ToString();
  return IsSigned(*maybe_type.value());
}

absl::Status BytecodeEmitter::HandleBinop(const Binop* node) {
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->rhs()->AcceptExpr(this));
  switch (node->binop_kind()) {
    case BinopKind::kAdd: {
      XLS_ASSIGN_OR_RETURN(bool is_signed, IsBitsTypeNodeSigned(node));
      bytecode_.push_back(Bytecode(
          node->span(), is_signed ? Bytecode::Op::kSAdd : Bytecode::Op::kUAdd));
      break;
    }
    case BinopKind::kAnd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kAnd));
      break;
    case BinopKind::kConcat:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kConcat));
      break;
    case BinopKind::kDiv:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kDiv));
      break;
    case BinopKind::kMod:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kMod));
      break;
    case BinopKind::kEq:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kEq));
      break;
    case BinopKind::kGe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kGe));
      break;
    case BinopKind::kGt:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kGt));
      break;
    case BinopKind::kLe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLe));
      break;
    case BinopKind::kLogicalAnd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLogicalAnd));
      break;
    case BinopKind::kLogicalOr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLogicalOr));
      break;
    case BinopKind::kLt:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLt));
      break;
    case BinopKind::kMul: {
      XLS_ASSIGN_OR_RETURN(bool is_signed, IsBitsTypeNodeSigned(node));
      bytecode_.push_back(Bytecode(
          node->span(), is_signed ? Bytecode::Op::kSMul : Bytecode::Op::kUMul));
      break;
    }
    case BinopKind::kNe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kNe));
      break;
    case BinopKind::kOr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kOr));
      break;
    case BinopKind::kShl:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShl));
      break;
    case BinopKind::kShr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShr));
      break;
    case BinopKind::kSub: {
      XLS_ASSIGN_OR_RETURN(bool is_signed, IsBitsTypeNodeSigned(node));
      bytecode_.push_back(Bytecode(
          node->span(), is_signed ? Bytecode::Op::kSSub : Bytecode::Op::kUSub));
      break;
    }
    case BinopKind::kXor:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kXor));
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unimplemented binary operator: ",
                       BinopKindToString(node->binop_kind())));
  }
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleStatementBlock(const StatementBlock* node) {
  VLOG(5) << "BytecodeEmitter::HandleStatementBlock @ "
          << node->span().ToString(file_table()) << " trailing semi? "
          << node->trailing_semi();
  const Expr* last_expression = nullptr;
  for (const Statement* s : node->statements()) {
    // Do not permit expression-statements to have a result on the stack for any
    // subsequent expression-statement.
    if (last_expression != nullptr) {
      Add(Bytecode::MakePop(last_expression->span()));
    }

    VLOG(5) << "BytecodeEmitter::HandleStatement: `" << s->ToString() << "`";
    XLS_RETURN_IF_ERROR(absl::visit(Visitor{
                                        [&](Expr* e) {
                                          last_expression = e;
                                          return e->AcceptExpr(this);
                                        },
                                        [&](Let* let) {
                                          last_expression = nullptr;
                                          return HandleLet(let);
                                        },
                                        [&](ConstAssert*) {
                                          // Nothing to emit, should be
                                          // resolved via type inference.
                                          last_expression = nullptr;
                                          return absl::OkStatus();
                                        },
                                        [&](TypeAlias*) {
                                          // Nothing to emit, should be
                                          // resolved via type inference.
                                          last_expression = nullptr;
                                          return absl::OkStatus();
                                        },
                                        [&](VerbatimNode*) {
                                          return absl::UnimplementedError(
                                              "Should not emit VerbatimNode");
                                        },
                                    },
                                    s->wrapped()));
  }

  if (node->trailing_semi()) {
    if (last_expression == nullptr) {
      Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUnit()));
    } else {
      Add(Bytecode::MakePop(node->span()));
      Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUnit()));
    }
  }
  return absl::OkStatus();
}

static absl::StatusOr<Type*> GetTypeOfNode(const AstNode* node,
                                           const TypeInfo* type_info) {
  std::optional<Type*> maybe_type = type_info->GetItem(node);

  if (!maybe_type.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for node ", node->ToString()));
  }

  return maybe_type.value();
}

static absl::StatusOr<std::unique_ptr<BitsType>> GetTypeOfNodeAsBits(
    const AstNode* node, const TypeInfo* type_info) {
  std::optional<Type*> maybe_type = type_info->GetItem(node);

  if (!maybe_type.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for node ", node->ToString()));
  }

  std::optional<BitsLikeProperties> bits_like =
      GetBitsLike(*maybe_type.value());
  if (!bits_like.has_value()) {
    return absl::InternalError(
        absl::StrCat("Bytecode emitter only supports widening or checked "
                     "casts from/to bits; got ",
                     node->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());
  return std::make_unique<BitsType>(is_signed, bits_like->size);
}

static absl::Status MaybeCheckArrayToBitsCast(const AstNode* node,
                                              const Type* from,
                                              const Type* to) {
  const ArrayType* from_array = dynamic_cast<const ArrayType*>(from);
  bool to_is_bits_like = IsBitsLike(*to);

  if (from_array != nullptr && !to_is_bits_like) {
    return absl::InternalError(absl::StrCat(
        "The only valid array cast is to bits: ", node->ToString()));
  }

  if (from_array == nullptr || !to_is_bits_like) {
    return absl::OkStatus();
  }

  // Bits-constructor acts as a bits type, so we don't need to perform
  // array-oriented cast checks.
  if (IsArrayOfBitsConstructor(*from_array)) {
    return absl::OkStatus();
  }

  // Check casting from an array to bits.
  if (from_array->element_type().GetAllDims().size() != 1) {
    return absl::InternalError(
        "Only casts to/from one-dimensional arrays are supported.");
  }

  XLS_ASSIGN_OR_RETURN(TypeDim bit_count_dim, from_array->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t array_bit_count, bit_count_dim.GetAsInt64());

  XLS_ASSIGN_OR_RETURN(bit_count_dim, to->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bits_bit_count, bit_count_dim.GetAsInt64());

  if (array_bit_count != bits_bit_count) {
    return absl::InternalError(absl::StrFormat(
        "Array-to-bits cast bit counts must match. "
        "Saw %d for \"from\" type `%s` vs %d for \"to\" type `%s`.",
        array_bit_count, from->ToString(), bits_bit_count, to->ToString()));
  }

  return absl::OkStatus();
}

static absl::Status MaybeCheckEnumToBitsCast(const AstNode* node,
                                             const Type* from, const Type* to) {
  const EnumType* from_enum = dynamic_cast<const EnumType*>(from);
  bool to_is_bits_like = IsBitsLike(*to);

  if (from_enum != nullptr && !to_is_bits_like) {
    return absl::InternalError(absl::StrCat(
        "The only valid enum cast is to bits: ", node->ToString()));
  }

  return absl::OkStatus();
}

static absl::Status MaybeCheckBitsToArrayCast(const AstNode* node,
                                              const Type* from,
                                              const Type* to) {
  bool from_is_bits_like = IsBitsLike(*from);
  const ArrayType* to_array = dynamic_cast<const ArrayType*>(to);

  if (to_array != nullptr && !from_is_bits_like) {
    return absl::InternalError(absl::StrCat(
        "The only valid array cast is from bits: ", node->ToString()));
  }

  if (!from_is_bits_like || to_array == nullptr) {
    return absl::OkStatus();
  }

  // Bits-constructor acts as a bits type, so we don't need to perform
  // array-oriented cast checks.
  if (IsArrayOfBitsConstructor(*to_array)) {
    return absl::OkStatus();
  }

  // Casting from bits to an array.
  if (to_array->element_type().GetAllDims().size() != 1) {
    return absl::InternalError(
        "Only casts to/from one-dimensional arrays are supported.");
  }

  VLOG(5) << "from_bits: " << from->ToString()
          << " to_array: " << to_array->ToString();

  XLS_ASSIGN_OR_RETURN(TypeDim bit_count_dim, from->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bits_bit_count, bit_count_dim.GetAsInt64());

  XLS_ASSIGN_OR_RETURN(bit_count_dim, to_array->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t array_bit_count, bit_count_dim.GetAsInt64());

  if (array_bit_count != bits_bit_count) {
    return absl::InternalError(absl::StrFormat(
        "Bits-to-array cast bit counts must match. "
        "bits-type `%s` bit count: %d; array-type bit count for `%s`: %d.",
        from->ToString(), bits_bit_count, to->ToString(), array_bit_count));
  }

  return absl::OkStatus();
}

static absl::Status MaybeCheckBitsToEnumCast(const AstNode* node,
                                             const Type* from, const Type* to) {
  bool from_is_bits_like = IsBitsLike(*from);
  const EnumType* to_enum = dynamic_cast<const EnumType*>(to);

  if (to_enum != nullptr && !from_is_bits_like) {
    return absl::InternalError(absl::StrCat(
        "The only valid enum cast is from bits: ", node->ToString()));
  }

  return absl::OkStatus();
}

static absl::Status CheckSupportedCastTypes(const AstNode* node,
                                            const Type* type) {
  bool is_bits_like = IsBitsLike(*type);
  const ArrayType* as_array_type = dynamic_cast<const ArrayType*>(type);
  const EnumType* as_enum_type = dynamic_cast<const EnumType*>(type);

  if (!is_bits_like && as_array_type == nullptr && as_enum_type == nullptr) {
    return absl::InternalError(
        absl::StrCat("Bytecode emitter only supports casts from/to "
                     "arrays, enums, or bits; got ",
                     node->ToString()));
  }

  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleCast(const Cast* node) {
  VLOG(5) << "BytecodeEmitter::HandleCast @ "
          << node->span().ToString(file_table());

  const Expr* from_expr = node->expr();
  XLS_RETURN_IF_ERROR(from_expr->AcceptExpr(this));

  XLS_ASSIGN_OR_RETURN(Type * from, GetTypeOfNode(from_expr, type_info_));
  XLS_ASSIGN_OR_RETURN(Type * to, GetTypeOfNode(node, type_info_));

  XLS_RETURN_IF_ERROR(CheckSupportedCastTypes(node, from));
  XLS_RETURN_IF_ERROR(CheckSupportedCastTypes(node, to));
  XLS_RETURN_IF_ERROR(MaybeCheckArrayToBitsCast(node, from, to));
  XLS_RETURN_IF_ERROR(MaybeCheckEnumToBitsCast(node, from, to));
  XLS_RETURN_IF_ERROR(MaybeCheckBitsToArrayCast(node, from, to));
  XLS_RETURN_IF_ERROR(MaybeCheckBitsToEnumCast(node, from, to));

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kCast, to->CloneToUnique()));

  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinRecv(const Invocation* node) {
  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];

  XLS_RETURN_IF_ERROR(token->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(channel->AcceptExpr(this));
  // All receives need a predicate. Set to true for unconditional receive..
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUBits(1, 1)));
  XLS_ASSIGN_OR_RETURN(
      Bytecode::ChannelData channel_data,
      CreateChannelData(channel, type_info_, options_.format_preference));
  // Default value which is unused because the predicate is always
  // true. Required because the Recv bytecode has a predicate and default value
  // operand.
  XLS_ASSIGN_OR_RETURN(InterpValue default_value,
                       CreateZeroValueFromType(channel_data.payload_type()));
  Add(Bytecode::MakeLiteral(node->span(), default_value));
  Add(Bytecode::MakeRecv(node->span(), std::move(channel_data)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinJoin(const Invocation* node) {
  // Since we serially execute top-to-bottom, every node is an implicit join.
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeToken()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinToken(const Invocation* node) {
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeToken()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinRecvNonBlocking(
    const Invocation* node) {
  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* default_value = node->args()[2];

  XLS_RETURN_IF_ERROR(token->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(channel->AcceptExpr(this));
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUBits(1, 1)));
  XLS_RETURN_IF_ERROR(default_value->AcceptExpr(this));
  XLS_ASSIGN_OR_RETURN(
      Bytecode::ChannelData channel_data,
      CreateChannelData(channel, type_info_, options_.format_preference));
  Add(Bytecode::MakeRecvNonBlocking(node->span(), std::move(channel_data)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinRecvIf(const Invocation* node) {
  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* condition = node->args()[2];
  Expr* default_value = node->args()[3];

  XLS_RETURN_IF_ERROR(token->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(channel->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(condition->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(default_value->AcceptExpr(this));
  XLS_ASSIGN_OR_RETURN(
      Bytecode::ChannelData channel_data,
      CreateChannelData(channel, type_info_, options_.format_preference));
  Add(Bytecode::MakeRecv(node->span(), std::move(channel_data)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinRecvIfNonBlocking(
    const Invocation* node) {
  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* condition = node->args()[2];
  Expr* default_value = node->args()[3];

  XLS_RETURN_IF_ERROR(token->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(channel->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(condition->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(default_value->AcceptExpr(this));
  XLS_ASSIGN_OR_RETURN(
      Bytecode::ChannelData channel_data,
      CreateChannelData(channel, type_info_, options_.format_preference));
  Add(Bytecode::MakeRecvNonBlocking(node->span(), std::move(channel_data)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinSend(const Invocation* node) {
  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* payload = node->args()[2];

  XLS_RETURN_IF_ERROR(token->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(channel->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(payload->AcceptExpr(this));
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUBits(1, 1)));
  XLS_ASSIGN_OR_RETURN(
      Bytecode::ChannelData channel_data,
      CreateChannelData(channel, type_info_, options_.format_preference));
  Add(Bytecode::MakeSend(node->span(), std::move(channel_data)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinSendIf(const Invocation* node) {
  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* condition = node->args()[2];
  Expr* payload = node->args()[3];

  XLS_RETURN_IF_ERROR(token->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(channel->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(payload->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(condition->AcceptExpr(this));
  XLS_ASSIGN_OR_RETURN(
      Bytecode::ChannelData channel_data,
      CreateChannelData(channel, type_info_, options_.format_preference));
  Add(Bytecode::MakeSend(node->span(), std::move(channel_data)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinDecode(const Invocation* node) {
  VLOG(5) << "BytecodeEmitter::HandleInvocation - Decode @ "
          << node->span().ToString(file_table());

  const Expr* from_expr = node->args().at(0);
  XLS_RETURN_IF_ERROR(from_expr->AcceptExpr(this));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BitsType> to,
                       GetTypeOfNodeAsBits(node, type_info_));

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kDecode, std::move(to)));

  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinCheckedCast(const Invocation* node) {
  VLOG(5) << "BytecodeEmitter::HandleInvocation - CheckedCast @ "
          << node->span().ToString(file_table());

  const Expr* from_expr = node->args().at(0);
  XLS_RETURN_IF_ERROR(from_expr->AcceptExpr(this));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BitsType> to,
                       GetTypeOfNodeAsBits(node, type_info_));

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kCheckedCast, std::move(to)));

  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBuiltinWideningCast(
    const Invocation* node) {
  VLOG(5) << "BytecodeEmitter::HandleInvocation - WideningCast @ "
          << node->span().ToString(file_table());

  const Expr* from_expr = node->args().at(0);
  XLS_RETURN_IF_ERROR(from_expr->AcceptExpr(this));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BitsType> to,
                       GetTypeOfNodeAsBits(node, type_info_));

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kCheckedCast, std::move(to)));

  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleChannelDecl(const ChannelDecl* node) {
  // Channels are created as constexpr values during type deduction/constexpr
  // evaluation, since they're concrete values that need to be shared amongst
  // two actors.
  XLS_ASSIGN_OR_RETURN(InterpValue channel, type_info_->GetConstExpr(node));
  Add(Bytecode::MakeLiteral(node->span(), channel));
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefToEnum(
    const ColonRef* colon_ref, EnumDef* enum_def, const TypeInfo* type_info) {
  // TODO(rspringer): 2022-01-26 We'll need to pull the right type info during
  // ResolveTypeDefToEnum.
  std::string_view attr = colon_ref->attr();
  XLS_ASSIGN_OR_RETURN(Expr * value_expr, enum_def->GetValue(attr));
  XLS_ASSIGN_OR_RETURN(InterpValue result, type_info->GetConstExpr(value_expr));
  XLS_RET_CHECK(result.IsEnum())
      << "expect constexpr for enum value expr `" << value_expr->ToString()
      << "` to evaluate to enum type; got: `" << result.ToString() << "`";
  return result;
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefToValue(
    Module* module, const ColonRef* colon_ref) {
  // TODO(rspringer): We'll need subject resolution to return the appropriate
  // TypeInfo for parametrics.
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data_->GetRootTypeInfo(module));

  std::optional<ModuleMember*> member =
      module->FindMemberWithName(colon_ref->attr());
  if (!member.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Could not find member %s of module %s.",
                        colon_ref->attr(), module->name()));
  }

  if (std::holds_alternative<Function*>(*member.value())) {
    Function* f = std::get<Function*>(*member.value());
    return InterpValue::MakeFunction(InterpValue::UserFnData{module, f});
  }

  XLS_RET_CHECK(std::holds_alternative<ConstantDef*>(*member.value()));
  ConstantDef* constant_def = std::get<ConstantDef*>(*member.value());
  return type_info->GetConstExpr(constant_def->value());
}

absl::Status BytecodeEmitter::HandleColonRef(const ColonRef* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, HandleColonRefInternal(node));

  Add(Bytecode::MakeLiteral(node->span(), value));
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefInternal(
    const ColonRef* node) {
  XLS_ASSIGN_OR_RETURN(
      auto resolved_subject,
      ResolveColonRefSubjectAfterTypeChecking(import_data_, type_info_, node));

  return absl::visit(
      Visitor{
          [&](EnumDef* enum_def) -> absl::StatusOr<InterpValue> {
            const TypeInfo* type_info = type_info_;
            if (enum_def->owner() != type_info_->module()) {
              type_info =
                  import_data_->GetRootTypeInfo(enum_def->owner()).value();
            }
            return HandleColonRefToEnum(node, enum_def, type_info);
          },
          [&](BuiltinNameDef* builtin_name_def) -> absl::StatusOr<InterpValue> {
            return GetBuiltinNameDefColonAttr(builtin_name_def, node->attr());
          },
          [&](ArrayTypeAnnotation* array_type) -> absl::StatusOr<InterpValue> {
            XLS_ASSIGN_OR_RETURN(
                TypeInfo * type_info,
                import_data_->GetRootTypeInfoForNode(array_type));
            XLS_ASSIGN_OR_RETURN(InterpValue value,
                                 type_info->GetConstExpr(array_type->dim()));
            XLS_ASSIGN_OR_RETURN(uint64_t dim_u64, value.GetBitValueUnsigned());
            return GetArrayTypeColonAttr(array_type, dim_u64, node->attr());
          },
          [&](Module* module) -> absl::StatusOr<InterpValue> {
            return HandleColonRefToValue(module, node);
          },
          [&](Impl* impl) -> absl::StatusOr<InterpValue> {
            std::optional<ImplMember> member = impl->GetMember(node->attr());
            XLS_RET_CHECK(member.has_value());
            if (std::holds_alternative<Function*>(*member)) {
              Function* f = std::get<Function*>(*member);
              return InterpValue::MakeFunction(
                  InterpValue::UserFnData{f->owner(), f});
            }
            return type_info_->GetConstExpr(node);
          }},
      resolved_subject);
}

absl::Status BytecodeEmitter::HandleConstAssert(const ConstAssert* node) {
  // Since static assertions are checked to hold at typechecking time, we do not
  // need to check them dynamically via the bytecode execution.
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleConstantArray(const ConstantArray* node) {
  return HandleArray(node);
}

absl::Status BytecodeEmitter::HandleConstRef(const ConstRef* node) {
  return HandleNameRef(node);
}

absl::Status BytecodeEmitter::HandleFor(const For* node) {
  // Here's how a `for` loop is implemented, in some sort of pseudocode:
  //  - Initialize iterable & index
  //  - Initialize the accumulator
  //  - loop_top:
  //  - if index == iterable_sz: jump to end:
  //  - Create the loop carry: the tuple of (index, accumulator).
  //  - Execute the loop body.
  //  - Jump to loop_top:
  //  - end:

  // First, get the size of the iterable array.
  std::optional<Type*> maybe_iterable_type =
      type_info_->GetItem(node->iterable());
  if (!maybe_iterable_type.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find concrete type for loop iterable: ",
                     node->iterable()->ToString()));
  }

  ArrayType* array_type = dynamic_cast<ArrayType*>(maybe_iterable_type.value());
  if (array_type == nullptr) {
    return absl::InternalError(absl::StrCat("Iterable was not of array type: ",
                                            node->iterable()->ToString()));
  }

  TypeDim iterable_size_dim = array_type->size();
  XLS_ASSIGN_OR_RETURN(int64_t iterable_size, iterable_size_dim.GetAsInt64());

  size_t iterable_slot = next_slotno_++;
  XLS_RETURN_IF_ERROR(node->iterable()->AcceptExpr(this));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(iterable_slot)));

  size_t index_slot = next_slotno_++;
  VLOG(10) << "BytecodeEmitter::HandleFor; index_slot: " << index_slot;
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral, InterpValue::MakeU32(0)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(index_slot)));

  // Evaluate the initial accumulator value & leave it on the stack.
  XLS_RETURN_IF_ERROR(node->init()->AcceptExpr(this));

  // Jump destination for the end-of-loop jump to start.
  size_t top_of_loop = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));

  // Loop header: Are we done iterating?
  // Reload the current index and compare against the iterable size.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(index_slot)));
  CHECK_EQ(static_cast<uint32_t>(iterable_size), iterable_size);
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral,
               InterpValue::MakeU32(static_cast<uint32_t>(iterable_size))));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kEq));

  // Cache the location of the top-of-loop jump so we can patch it up later once
  // we actually know the size of the jump.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRelIf,
                               Bytecode::kPlaceholderJumpAmount));
  size_t start_jump_idx = bytecode_.size() - 1;

  // The loop-carry values are a tuple (index, accumulator), but we don't have
  // the index on the stack (so we don't have to drop it once we're out of the
  // loop). We need to plop it on there, then swap the top two values so that
  // the resulting tuple is in the right order.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(iterable_slot)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(index_slot)));
  Add(Bytecode::MakeIndex(node->span()));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSwap));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(2)));
  XLS_RETURN_IF_ERROR(DestructureLet(node->names(), /*type_or_size=*/2));

  // Emit the loop body.
  XLS_RETURN_IF_ERROR(node->body()->AcceptExpr(this));

  // End of loop: increment the loop index and jump back to the beginning.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(index_slot)));
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral, InterpValue::MakeU32(1)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kUAdd));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(index_slot)));
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kJumpRel,
               Bytecode::JumpTarget(top_of_loop - bytecode_.size())));

  // Finally, we're done with the loop!
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));

  // Now we can finally know the relative jump amount for the top-of-loop jump.
  bytecode_.at(start_jump_idx)
      .PatchJumpTarget(bytecode_.size() - start_jump_idx - 1);
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleFunctionRef(const FunctionRef* node) {
  return node->callee()->AcceptExpr(this);
}

absl::Status BytecodeEmitter::HandleZeroMacro(const ZeroMacro* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, type_info_->GetConstExpr(node));
  Add(Bytecode::MakeLiteral(node->span(), std::move(value)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleAllOnesMacro(const AllOnesMacro* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, type_info_->GetConstExpr(node));
  Add(Bytecode::MakeLiteral(node->span(), std::move(value)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleFormatMacro(const FormatMacro* node) {
  for (const Expr* arg : node->args()) {
    XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
  }

  std::vector<FormatPreference> preferences =
      OperandPreferencesFromFormat(node->format());
  XLS_RET_CHECK_EQ(preferences.size(), node->args().size());
  // Replace kDefault values with the format specified in the options if it is
  // not kDefault. This enables user override of default format preference.
  if (options_.format_preference != FormatPreference::kDefault) {
    for (FormatPreference& preference : preferences) {
      preference = preference == FormatPreference::kDefault
                       ? options_.format_preference
                       : preference;
    }
  }
  std::vector<ValueFormatDescriptor> value_fmt_descs;
  for (size_t i = 0; i < node->args().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        ValueFormatDescriptor value_fmt_desc,
        ExprToValueFormatDescriptor(node->args().at(i), type_info_,
                                    preferences.at(i)));
    value_fmt_descs.push_back(std::move(value_fmt_desc));
  }

  Bytecode::TraceData trace_data(
      std::vector<FormatStep>(node->format().begin(), node->format().end()),
      std::move(value_fmt_descs));
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kTrace, std::move(trace_data)));
  return absl::OkStatus();
}

static absl::StatusOr<int64_t> GetValueWidth(const TypeInfo* type_info,
                                             Expr* expr) {
  std::optional<Type*> maybe_type = type_info->GetItem(expr);
  if (!maybe_type.has_value()) {
    return absl::InternalError(
        "Could not find concrete type for slice component.");
  }
  return maybe_type.value()->GetTotalBitCount()->GetAsInt64();
}

absl::Status BytecodeEmitter::HandleIndex(const Index* node) {
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));

  if (std::holds_alternative<Slice*>(node->rhs())) {
    Slice* slice = std::get<Slice*>(node->rhs());
    if (slice->start() == nullptr) {
      int64_t start_width;
      if (slice->limit() == nullptr) {
        // TODO(rspringer): Define a uniform `usize` to avoid specifying magic
        // numbers here. This is the default size used for untyped numbers in
        // the typechecker.
        start_width = 32;
      } else {
        XLS_ASSIGN_OR_RETURN(start_width,
                             GetValueWidth(type_info_, slice->limit()));
      }
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                                   InterpValue::MakeSBits(start_width, 0)));
    } else {
      XLS_RETURN_IF_ERROR(slice->start()->AcceptExpr(this));
    }

    if (slice->limit() == nullptr) {
      std::optional<Type*> maybe_type = type_info_->GetItem(node->lhs());
      if (!maybe_type.has_value()) {
        return absl::InternalError("Could not find concrete type for slice.");
      }
      Type* type = maybe_type.value();
      // These will never fail.
      absl::StatusOr<TypeDim> dim_or = type->GetTotalBitCount();
      absl::StatusOr<int64_t> width_or = dim_or.value().GetAsInt64();

      int64_t limit_width;
      if (slice->start() == nullptr) {
        // TODO(rspringer): Define a uniform `usize` to avoid specifying magic
        // numbers here. This is the default size used for untyped numbers in
        // the typechecker.
        limit_width = 32;
      } else {
        XLS_ASSIGN_OR_RETURN(limit_width,
                             GetValueWidth(type_info_, slice->start()));
      }
      bytecode_.push_back(
          Bytecode(node->span(), Bytecode::Op::kLiteral,
                   InterpValue::MakeSBits(limit_width, width_or.value())));
    } else {
      XLS_RETURN_IF_ERROR(slice->limit()->AcceptExpr(this));
    }
    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSlice));
    return absl::OkStatus();
  }

  if (std::holds_alternative<WidthSlice*>(node->rhs())) {
    WidthSlice* width_slice = std::get<WidthSlice*>(node->rhs());
    XLS_RETURN_IF_ERROR(width_slice->start()->AcceptExpr(this));

    std::optional<Type*> maybe_type = type_info_->GetItem(width_slice->width());
    if (!maybe_type.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Could not find concrete type for slice width parameter \"",
          width_slice->width()->ToString(), "\"."));
    }

    MetaType* type = dynamic_cast<MetaType*>(maybe_type.value());
    XLS_RET_CHECK(type != nullptr) << maybe_type.value()->ToString();
    XLS_RET_CHECK(IsBitsLike(*type->wrapped())) << type->ToString();

    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kWidthSlice,
                                 type->CloneToUnique()));
    return absl::OkStatus();
  }

  // Otherwise, it's a regular [array or tuple] index op.
  Expr* expr = std::get<Expr*>(node->rhs());
  XLS_RETURN_IF_ERROR(expr->AcceptExpr(this));
  Add(Bytecode::MakeIndex(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleInvocation(const Invocation* node) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(node->callee());
      name_ref != nullptr && name_ref->IsBuiltin()) {
    VLOG(10) << "HandleInvocation; builtin name_ref: " << name_ref->ToString();

    if (name_ref->identifier() == "trace!") {
      if (node->args().size() != 1) {
        return absl::InternalError("`trace!` takes a single argument.");
      }

      XLS_RETURN_IF_ERROR(node->args().at(0)->AcceptExpr(this));

      std::vector<FormatStep> steps;
      steps.push_back(
          absl::StrCat("trace of ", node->args()[0]->ToString(), ": "));
      steps.push_back(options_.format_preference);
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kTrace,
                                   Bytecode::TraceData(std::move(steps), {})));
      return absl::OkStatus();
    }

    if (name_ref->identifier() == "decode") {
      return HandleBuiltinDecode(node);
    }

    if (name_ref->identifier() == "widening_cast") {
      return HandleBuiltinWideningCast(node);
    }

    if (name_ref->identifier() == "checked_cast") {
      return HandleBuiltinCheckedCast(node);
    }
    if (name_ref->identifier() == "send") {
      return HandleBuiltinSend(node);
    }
    if (name_ref->identifier() == "send_if") {
      return HandleBuiltinSendIf(node);
    }
    if (name_ref->identifier() == "recv") {
      return HandleBuiltinRecv(node);
    }
    if (name_ref->identifier() == "recv_if") {
      return HandleBuiltinRecvIf(node);
    }
    if (name_ref->identifier() == "recv_non_blocking") {
      return HandleBuiltinRecvNonBlocking(node);
    }
    if (name_ref->identifier() == "recv_if_non_blocking") {
      return HandleBuiltinRecvIfNonBlocking(node);
    }
    if (name_ref->identifier() == "join") {
      return HandleBuiltinJoin(node);
    }
    if (name_ref->identifier() == "token") {
      return HandleBuiltinToken(node);
    }
  }

  for (auto* arg : node->args()) {
    XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
  }

  XLS_RETURN_IF_ERROR(node->callee()->AcceptExpr(this));

  std::optional<ParametricEnv> callee_bindings;
  if (caller_bindings_.has_value()) {
    std::optional<const ParametricEnv*> callee_bindings_ptr =
        type_info_->GetInvocationCalleeBindings(node, caller_bindings_.value());
    if (callee_bindings_ptr.has_value()) {
      callee_bindings = *callee_bindings_ptr.value();
    }
  }

  bytecode_.push_back(Bytecode(
      node->span(), Bytecode::Op::kCall,
      Bytecode::InvocationData{node, caller_bindings_, callee_bindings}));
  return absl::OkStatus();
}

absl::StatusOr<Bytecode::MatchArmItem> BytecodeEmitter::HandleNameDefTreeExpr(
    NameDefTree* tree, Type* type) {
  if (tree->is_leaf()) {
    return absl::visit(
        Visitor{
            [&](NameRef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              using ValueT = std::variant<InterpValue, Bytecode::SlotIndex>;
              XLS_ASSIGN_OR_RETURN(ValueT item_value, HandleNameRefInternal(n));
              if (std::holds_alternative<InterpValue>(item_value)) {
                return Bytecode::MatchArmItem::MakeInterpValue(
                    std::get<InterpValue>(item_value));
              }

              return Bytecode::MatchArmItem::MakeLoad(
                  std::get<Bytecode::SlotIndex>(item_value));
            },
            [&](Number* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              XLS_ASSIGN_OR_RETURN(FormattedInterpValue number,
                                   HandleNumberInternal(n));
              return Bytecode::MatchArmItem::MakeInterpValue(number.value);
            },
            [&](Range* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              XLS_ASSIGN_OR_RETURN(
                  FormattedInterpValue start,
                  HandleNumberInternal(down_cast<Number*>(n->start())));
              XLS_ASSIGN_OR_RETURN(
                  FormattedInterpValue end,
                  HandleNumberInternal(down_cast<Number*>(n->end())));
              return Bytecode::MatchArmItem::MakeRange(start.value, end.value);
            },
            [&](ColonRef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              XLS_ASSIGN_OR_RETURN(InterpValue value,
                                   HandleColonRefInternal(n));
              return Bytecode::MatchArmItem::MakeInterpValue(value);
            },
            [&](NameDef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              int64_t slot_index = next_slotno_++;
              namedef_to_slot_[n] = slot_index;
              return Bytecode::MatchArmItem::MakeStore(
                  Bytecode::SlotIndex(slot_index));
            },
            [&](WildcardPattern* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              return Bytecode::MatchArmItem::MakeWildcard();
            },
            [&](RestOfTuple* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              return Bytecode::MatchArmItem::MakeRestOfTuple();
            },
        },
        tree->leaf());
  }

  // Not a leaf; must be a tuple
  auto* tuple_type = down_cast<TupleType*>(type);
  if (tuple_type == nullptr) {
    return TypeInferenceErrorStatus(
        tree->span(), type, "Pattern expected matched-on type to be a tuple.",
        file_table());
  }

  XLS_ASSIGN_OR_RETURN((auto [number_of_tuple_elements, number_of_names]),
                       GetTupleSizes(tree, tuple_type));

  // TODO: https://github.com/google/xls/issues/1459 - This is at least the
  // 3rd if not 4th time a loop like this has been written. It should be
  // refactored into a common utility function.
  std::vector<Bytecode::MatchArmItem> elements;
  int64_t tuple_index = 0;
  const NameDefTree::Nodes& nodes = tree->nodes();
  for (int64_t name_index = 0; name_index < nodes.size(); ++name_index) {
    NameDefTree* subnode = nodes[name_index];
    if (subnode->IsRestOfTupleLeaf()) {
      // Skip ahead.
      int64_t wildcards_to_insert = number_of_tuple_elements - number_of_names;
      tuple_index += wildcards_to_insert;

      for (int64_t i = 0; i < wildcards_to_insert; ++i) {
        elements.push_back(Bytecode::MatchArmItem::MakeWildcard());
      }
      continue;
    }

    Type& subtype = tuple_type->GetMemberType(tuple_index);
    XLS_ASSIGN_OR_RETURN(Bytecode::MatchArmItem element,
                         HandleNameDefTreeExpr(subnode, &subtype));
    elements.push_back(element);
    tuple_index++;
  }
  return Bytecode::MatchArmItem::MakeTuple(std::move(elements));
}

static int64_t CountElements(std::variant<Type*, int64_t> element) {
  return std::visit(Visitor{[&](Type* type) -> int64_t {
                              const TupleType* tuple_type =
                                  dynamic_cast<const TupleType*>(type);
                              if (tuple_type != nullptr) {
                                return tuple_type->members().size();
                              }
                              return 0;
                            },
                            [&](int64_t size) -> int64_t { return size; }},
                    element);
}

absl::Status BytecodeEmitter::DestructureLet(
    NameDefTree* tree, std::variant<Type*, int64_t> type_or_size) {
  if (tree->is_leaf()) {
    if (std::holds_alternative<WildcardPattern*>(tree->leaf()) ||
        std::holds_alternative<RestOfTuple*>(tree->leaf())) {
      // We can just drop this one.
      Add(Bytecode::MakePop(tree->span()));
      return absl::OkStatus();
    }

    NameDef* name_def = std::get<NameDef*>(tree->leaf());
    if (!namedef_to_slot_.contains(name_def)) {
      namedef_to_slot_.insert({name_def, next_slotno_++});
    }
    int64_t slot = namedef_to_slot_.at(name_def);
    Add(Bytecode::MakeStore(tree->span(), Bytecode::SlotIndex(slot)));
  } else {
    // Pushes each element of the current level of the tuple
    // onto the stack in reverse order, e.g., (a, (b, c)) pushes (b, c) then a
    Add(Bytecode(tree->span(), Bytecode::Op::kExpandTuple));

    // Note: we intentionally don't check validity of the tuple here; that's
    // done by Deduce().

    if (std::holds_alternative<Type*>(type_or_size)) {
      TupleType* tuple_type =
          dynamic_cast<TupleType*>(std::get<Type*>(type_or_size));
      if (tuple_type == nullptr) {
        return absl::InternalError(absl::StrFormat(
            "Type %s is not of type TupleType", tuple_type->ToString()));
      }
    }

    int64_t tuple_index = 0;
    for (int64_t name_index = 0; name_index < tree->nodes().size();
         ++name_index) {
      NameDefTree* node = tree->nodes()[name_index];
      if (node->IsRestOfTupleLeaf()) {
        int64_t number_of_tuple_elements = CountElements(type_or_size);
        // Decrement for the rest-of-tuple
        int64_t number_of_bindings = tree->nodes().size() - 1;

        // Skip ahead to account for the needed remaining elements.
        int64_t difference = number_of_tuple_elements - number_of_bindings;
        tuple_index += difference;

        // Pop unused tuple elements
        for (int64_t pop_count = 0; pop_count < difference; ++pop_count) {
          Add(Bytecode::MakePop(node->span()));
        }
        continue;
      }
      XLS_RETURN_IF_ERROR(std::visit(
          Visitor{[&](Type* type) -> absl::Status {
                    TupleType* tuple_type = down_cast<TupleType*>(type);
                    return DestructureLet(
                        node, &tuple_type->GetMemberType(tuple_index));
                  },
                  [&](int64_t size) -> absl::Status {
                    // If a simple count is given, the tuple can only
                    // contain single elements, so the child count
                    // must be 1.
                    return DestructureLet(node, 1);
                  }},
          type_or_size));
      ++tuple_index;
    }
  }
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleLet(const Let* node) {
  XLS_RETURN_IF_ERROR(node->rhs()->AcceptExpr(this));
  std::optional<Type*> type = type_info_->GetItem(node->rhs());
  if (type.has_value()) {
    return DestructureLet(node->name_def_tree(), type.value());
  }
  return absl::InternalError(absl::StrFormat(
      "@ %s: Could not retrieve type of right-hand side of `let`.",
      node->span().ToString(file_table())));
}

absl::Status BytecodeEmitter::HandleNameRef(const NameRef* node) {
  XLS_ASSIGN_OR_RETURN(auto result, HandleNameRefInternal(node));
  if (std::holds_alternative<InterpValue>(result)) {
    Add(Bytecode::MakeLiteral(node->span(), std::get<InterpValue>(result)));
  } else {
    Add(Bytecode::MakeLoad(node->span(),
                           std::get<Bytecode::SlotIndex>(result)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::variant<InterpValue, Bytecode::SlotIndex>>
BytecodeEmitter::HandleNameRefInternal(const NameRef* node) {
  if (std::holds_alternative<BuiltinNameDef*>(node->name_def())) {
    // Builtins don't have NameDefs, so we can't use slots to store them. It's
    // simpler to just emit a literal InterpValue, anyway.
    BuiltinNameDef* builtin_def = std::get<BuiltinNameDef*>(node->name_def());
    XLS_ASSIGN_OR_RETURN(Builtin builtin,
                         BuiltinFromString(builtin_def->identifier()));
    return InterpValue::MakeFunction(builtin);
  }

  // Emit function and constant refs directly so that they can be stack elements
  // without having to load slots with them.
  const NameDef* name_def = std::get<const NameDef*>(node->name_def());
  if (auto* f = dynamic_cast<Function*>(name_def->definer()); f != nullptr) {
    return InterpValue::MakeFunction(InterpValue::UserFnData{f->owner(), f});
  }

  if (auto* cd = dynamic_cast<ConstantDef*>(name_def->definer());
      cd != nullptr) {
    return type_info_->GetConstExpr(cd->value());
  }

  // The value is either a local name or a parametric name.
  if (namedef_to_slot_.contains(name_def)) {
    return Bytecode::SlotIndex(namedef_to_slot_.at(name_def));
  }

  if (caller_bindings_.has_value()) {
    absl::flat_hash_map<std::string, InterpValue> bindings_map =
        caller_bindings_.value().ToMap();
    if (bindings_map.contains(name_def->identifier())) {
      return caller_bindings_.value().ToMap().at(name_def->identifier());
    }
  }

  return absl::InternalError(absl::StrCat(
      "BytecodeEmitter could not find slot or binding for name: ",
      name_def->ToString(), " @ ", name_def->span().ToString(file_table()),
      " stack: ", GetSymbolizedStackTraceAsString()));
}

absl::Status BytecodeEmitter::HandleNumber(const Number* node) {
  XLS_ASSIGN_OR_RETURN(FormattedInterpValue value, HandleNumberInternal(node));
  Add(Bytecode::MakeLiteral(node->span(), value.value,
                            value.format_descriptor));
  return absl::OkStatus();
}

absl::StatusOr<BytecodeEmitter::FormattedInterpValue>
BytecodeEmitter::HandleNumberInternal(const Number* node) {
  std::optional<Type*> type_or = type_info_->GetItem(node);
  if (!type_or.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for number: ", node->ToString()));
  }

  std::optional<BitsLikeProperties> bits_like = GetBitsLike(*type_or.value());

  XLS_RET_CHECK(bits_like.has_value())
      << "Not bits-like; number:" << node->ToString();

  XLS_ASSIGN_OR_RETURN(int64_t dim_val, bits_like->size.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());

  XLS_ASSIGN_OR_RETURN(Bits bits_value, node->GetBits(dim_val, file_table()));
  return FormattedInterpValue{
      .value = InterpValue::MakeBits(is_signed, bits_value),
      .format_descriptor = GetFormatDescriptorFromNumber(node)};
}

absl::Status BytecodeEmitter::HandleRange(const Range* node) {
  XLS_RETURN_IF_ERROR(node->start()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->end()->AcceptExpr(this));
  Add(Bytecode::MakeRange(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleSpawn(const Spawn* node) {
  XLS_ASSIGN_OR_RETURN(Proc * proc, ResolveProc(node->callee(), type_info_));
  for (const Expr* arg : node->config()->args()) {
    XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
  }
  XLS_RET_CHECK_EQ(node->next()->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(InterpValue initial_state,
                       type_info_->GetConstExpr(node->next()->args()[0]));

  // The whole Proc is parameterized, not the individual invocations
  // (config/next), so we can use either invocation to get the bindings.
  const ParametricEnv caller_bindings =
      caller_bindings_.has_value() ? caller_bindings_.value() : ParametricEnv();
  std::optional<const ParametricEnv*> maybe_callee_bindings =
      type_info_->GetInvocationCalleeBindings(node->config(), caller_bindings);
  std::optional<ParametricEnv> final_bindings = std::nullopt;
  if (maybe_callee_bindings.has_value()) {
    final_bindings = *maybe_callee_bindings.value();
  }

  Bytecode::SpawnFunctions spawn_functions = {.config = node->config(),
                                              .next = node->next()};
  Bytecode::SpawnData spawn_data{spawn_functions, proc, initial_state,
                                 caller_bindings, final_bindings};
  Add(Bytecode::MakeSpawn(node->span(), spawn_data));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleString(const String* node) {
  std::vector<InterpValue> u8s;
  for (unsigned char c : node->text()) {
    u8s.push_back(InterpValue::MakeUBits(/*bit_count=*/8, static_cast<int>(c)));
  }

  Add(Bytecode::MakeLiteral(node->span(),
                            InterpValue::MakeArray(std::move(u8s)).value()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleStructInstance(const StructInstance* node) {
  XLS_ASSIGN_OR_RETURN(StructType * struct_type,
                       type_info_->GetItemAs<StructType>(node));

  const StructDef& struct_def = struct_type->nominal_type();
  for (const std::pair<std::string, Expr*>& member :
       node->GetOrderedMembers(&struct_def)) {
    XLS_RETURN_IF_ERROR(member.second->AcceptExpr(this));
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(struct_def.size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleSplatStructInstance(
    const SplatStructInstance* node) {
  // For each field in the struct:
  //   If it's specified in the SplatStructInstance, then use it,
  //   otherwise extract it from the base struct.
  XLS_ASSIGN_OR_RETURN(StructType * struct_type,
                       type_info_->GetItemAs<StructType>(node));

  XLS_ASSIGN_OR_RETURN(std::vector<std::string> member_names,
                       struct_type->GetMemberNames());

  absl::flat_hash_map<std::string, Expr*> new_members;
  for (const std::pair<std::string, Expr*>& new_member : node->members()) {
    new_members[new_member.first] = new_member.second;
  }

  // To extract values from the base struct, we need to visit it and extract the
  // appropriate attr (i.e., index into the representative tuple).
  // Referencing a base struct doesn't have side effects, so we can just
  // visit it as often as we need.

  // Member names is ordered.
  for (int i = 0; i < member_names.size(); i++) {
    std::string& member_name = member_names.at(i);
    if (new_members.contains(member_name)) {
      XLS_RETURN_IF_ERROR(new_members.at(member_name)->AcceptExpr(this));
    } else {
      XLS_RETURN_IF_ERROR(node->splatted()->AcceptExpr(this));
      Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeU64(i)));
      Add(Bytecode::MakeIndex(node->span()));
    }
  }

  const StructDef& struct_def = struct_type->nominal_type();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(struct_def.size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleTupleIndex(const TupleIndex* node) {
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->index()->AcceptExpr(this));
  Add(Bytecode::MakeIndex(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleUnop(const Unop* node) {
  XLS_RETURN_IF_ERROR(node->operand()->AcceptExpr(this));
  switch (node->unop_kind()) {
    case UnopKind::kInvert:
      Add(Bytecode::MakeInvert(node->span()));
      break;
    case UnopKind::kNegate:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kNegate));
      break;
  }
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleVerbatimNode(const VerbatimNode* node) {
  return absl::UnimplementedError("Should not emit VerbatimNode");
}

absl::Status BytecodeEmitter::HandleXlsTuple(const XlsTuple* node) {
  for (auto* member : node->members()) {
    XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
  }

  // Pop the N elements and push the result as a single value.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(node->members().size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleConditional(const Conditional* node) {
  // Structure is:
  //
  //  $test
  //  jump_if =>consequent
  // alternate:
  //  $alternate
  //  jump =>join
  // consequent:
  //  jump_dest
  //  $consequent
  // join:
  //  jump_dest
  XLS_RETURN_IF_ERROR(node->test()->AcceptExpr(this));
  size_t jump_if_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRelIf,
                               Bytecode::kPlaceholderJumpAmount));
  XLS_RETURN_IF_ERROR(ToExprNode(node->alternate())->AcceptExpr(this));
  size_t jump_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRel,
                               Bytecode::kPlaceholderJumpAmount));
  size_t consequent_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));
  XLS_RETURN_IF_ERROR(node->consequent()->AcceptExpr(this));
  size_t jumpdest_index = bytecode_.size();

  // Now patch up the jumps since we now know the relative PCs we're jumping to.
  // TODO(leary): 2021-12-10 Keep track of the stack depth so we can validate it
  // at the jump destination.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));
  bytecode_.at(jump_if_index).PatchJumpTarget(consequent_index - jump_if_index);
  bytecode_.at(jump_index).PatchJumpTarget(jumpdest_index - jump_index);
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleUnrollFor(const UnrollFor* node) {
  std::optional<const Expr*> unrolled = type_info_->GetUnrolledLoop(
      node, caller_bindings_.has_value() ? *caller_bindings_ : ParametricEnv());
  if (unrolled.has_value()) {
    return (*unrolled)->AcceptExpr(this);
  }
  return absl::UnimplementedError(
      "UnrollFor nodes aren't interpretable/emittable until after they have "
      "been unrolled.");
}

absl::Status BytecodeEmitter::HandleMatch(const Match* node) {
  // Structure is: (value-to-match is at TOS0)
  //
  //  <value-to-match present>
  //  dup                       # Duplicate the value to match so the test can
  //                            # consume it and we can go to the next one if it
  //                            # fails.
  //  arm0_matcher_value        # The value to match against.
  //  ne                        # If it's not a match...
  //  jump_rel_if =>arm1_match  # Jump to the next arm!
  //  pop                       # Otherwise, we don't need the matchee any more,
  //                            # we found what arm we want to eval.
  //  $arm0_expr                # Eval the arm RHS.
  //  jump =>done               # Done evaluating the match, goto done.
  // arm1_match:                # Next match arm test...
  //  jump_dest                 # Is something we jump to.
  //  dup                       # It also needs to copy in case its test fails.
  //  arm1_matcher_value        # etc.
  //  ne
  //  jump_rel_if => wild_arm_match
  //  pop
  //  $arm1_expr
  //  jump =>done
  // wild_arm_match:            # Final arm must be wildcard.
  //  pop                       # Where we don't care about the matched value.
  //  $wild_arm_expr            # We just eval the RHS.
  // done:
  //  jump_dest                 # TOS0 holds result, matched value is gone.
  XLS_RETURN_IF_ERROR(node->matched()->AcceptExpr(this));

  const std::vector<MatchArm*>& arms = node->arms();
  if (arms.empty()) {
    return absl::InternalError("At least one match arm is required.");
  }

  std::vector<size_t> arm_offsets;
  std::vector<size_t> jumps_to_next;
  std::vector<size_t> jumps_to_done;

  std::optional<Type*> type = type_info_->GetItem(node->matched());
  if (!type.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for matched value: ",
                     node->matched()->ToString()));
  }
  for (size_t arm_idx = 0; arm_idx < node->arms().size(); ++arm_idx) {
    auto outer_scope_slots = namedef_to_slot_;
    absl::Cleanup cleanup = [this, &outer_scope_slots]() {
      namedef_to_slot_ = outer_scope_slots;
    };

    MatchArm* arm = node->arms()[arm_idx];
    arm_offsets.push_back(bytecode_.size());
    // We don't jump to the first arm test, but we jump to all the subsequent
    // ones.
    if (arm_offsets.size() > 1) {
      Add(Bytecode::MakeJumpDest(node->span()));
    }

    const std::vector<NameDefTree*>& patterns = arm->patterns();
    // First, prime the stack with all the copies of the matchee we'll need.
    for (int pattern_idx = 0; pattern_idx < patterns.size(); pattern_idx++) {
      Add(Bytecode::MakeDup(node->matched()->span()));
    }

    for (int pattern_idx = 0; pattern_idx < patterns.size(); pattern_idx++) {
      // Then we match each arm. We OR with the prev. result (if there is one)
      // and swap to the next copy of the matchee, unless this is the last
      // pattern.
      NameDefTree* ndt = arm->patterns()[pattern_idx];
      XLS_ASSIGN_OR_RETURN(Bytecode::MatchArmItem arm_item,
                           HandleNameDefTreeExpr(ndt, type.value()));
      Add(Bytecode::MakeMatchArm(ndt->span(), arm_item));

      if (pattern_idx != 0) {
        Add(Bytecode::MakeLogicalOr(ndt->span()));
      }
      if (pattern_idx != patterns.size() - 1) {
        Add(Bytecode::MakeSwap(ndt->span()));
      }
    }
    Add(Bytecode::MakeInvert(arm->span()));
    jumps_to_next.push_back(bytecode_.size());
    Add(Bytecode::MakeJumpRelIf(arm->span(), Bytecode::kPlaceholderJumpAmount));
    // Pop the value being matched since now we're going to produce the final
    // result.
    Add(Bytecode::MakePop(arm->span()));

    // The arm matched: calculate the resulting expression and jump
    // unconditionally to "done".
    XLS_RETURN_IF_ERROR(arm->expr()->AcceptExpr(this));
    jumps_to_done.push_back(bytecode_.size());
    Add(Bytecode::MakeJumpRel(arm->span(), Bytecode::kPlaceholderJumpAmount));
  }

  // Finally, handle the case where no arm matched, which is a failure.
  // Theoretically, we could reduce bytecode size by omitting this when the last
  // arm is strictly wildcards, but it doesn't seem worth the effort.
  arm_offsets.push_back(bytecode_.size());
  Add(Bytecode::MakeJumpDest(node->span()));

  std::vector<FormatStep> steps;
  steps.push_back("The value was not matched: value: ");
  steps.push_back(FormatPreference::kDefault);
  Add(Bytecode(node->span(), Bytecode::Op::kFail,
               Bytecode::TraceData(steps, {})));

  size_t done_offset = bytecode_.size();
  Add(Bytecode::MakeJumpDest(node->span()));

  // One extra for the fall-through failure case.
  CHECK_EQ(node->arms().size() + 1, arm_offsets.size());
  CHECK_EQ(node->arms().size(), jumps_to_next.size());
  for (size_t i = 0; i < jumps_to_next.size(); ++i) {
    size_t jump_offset = jumps_to_next[i];
    size_t next_offset = arm_offsets[i + 1];
    VLOG(5) << "Patching jump offset " << jump_offset
            << " to jump to next_offset " << next_offset;
    bytecode_.at(jump_offset).PatchJumpTarget(next_offset - jump_offset);
  }

  for (size_t offset : jumps_to_done) {
    VLOG(5) << "Patching jump-to-done offset " << offset
            << " to jump to done_offset " << done_offset;
    bytecode_.at(offset).PatchJumpTarget(done_offset - offset);
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
