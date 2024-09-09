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

#include "xls/dslx/type_system/type_info_to_proto.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/proto_adaptor_utils.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_expression.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_info.pb.h"
#include "xls/ir/bits.h"

namespace xls::dslx {
namespace {

// Converts the AstNodeKind (C++ enum class) to its protobuf form.
AstNodeKindProto ToProto(AstNodeKind kind) {
  switch (kind) {
    case AstNodeKind::kConstAssert:
      return AST_NODE_KIND_CONST_ASSERT;
    case AstNodeKind::kJoin:
      return AST_NODE_KIND_JOIN;
    case AstNodeKind::kTypeAnnotation:
      return AST_NODE_KIND_TYPE_ANNOTATION;
    case AstNodeKind::kModule:
      return AST_NODE_KIND_MODULE;
    case AstNodeKind::kNameDef:
      return AST_NODE_KIND_NAME_DEF;
    case AstNodeKind::kNameRef:
      return AST_NODE_KIND_NAME_REF;
    case AstNodeKind::kConstRef:
      return AST_NODE_KIND_CONST_REF;
    case AstNodeKind::kBuiltinNameDef:
      return AST_NODE_KIND_BUILTIN_NAME_DEF;
    case AstNodeKind::kConditional:
      return AST_NODE_KIND_CONDITIONAL;
    case AstNodeKind::kTypeAlias:
      return AST_NODE_KIND_TYPE_ALIAS;
    case AstNodeKind::kTypeRef:
      return AST_NODE_KIND_TYPE_REF;
    case AstNodeKind::kNumber:
      return AST_NODE_KIND_NUMBER;
    case AstNodeKind::kImport:
      return AST_NODE_KIND_IMPORT;
    case AstNodeKind::kUnop:
      return AST_NODE_KIND_UNOP;
    case AstNodeKind::kBinop:
      return AST_NODE_KIND_BINOP;
    case AstNodeKind::kColonRef:
      return AST_NODE_KIND_COLON_REF;
    case AstNodeKind::kParam:
      return AST_NODE_KIND_PARAM;
    case AstNodeKind::kFunction:
      return AST_NODE_KIND_FUNCTION;
    case AstNodeKind::kProc:
      return AST_NODE_KIND_PROC;
    case AstNodeKind::kArray:
      return AST_NODE_KIND_ARRAY;
    case AstNodeKind::kString:
      return AST_NODE_KIND_STRING;
    case AstNodeKind::kStructInstance:
      return AST_NODE_KIND_STRUCT_INSTANCE;
    case AstNodeKind::kNameDefTree:
      return AST_NODE_KIND_NAME_DEF_TREE;
    case AstNodeKind::kSplatStructInstance:
      return AST_NODE_KIND_SPLAT_STRUCT_INSTANCE;
    case AstNodeKind::kIndex:
      return AST_NODE_KIND_INDEX;
    case AstNodeKind::kRange:
      return AST_NODE_KIND_RANGE;
    case AstNodeKind::kRecv:
      return AST_NODE_KIND_RECV;
    case AstNodeKind::kRecvIf:
      return AST_NODE_KIND_RECV_IF;
    case AstNodeKind::kRecvIfNonBlocking:
      return AST_NODE_KIND_RECV_IF_NON_BLOCKING;
    case AstNodeKind::kRecvNonBlocking:
      return AST_NODE_KIND_RECV_NON_BLOCKING;
    case AstNodeKind::kSend:
      return AST_NODE_KIND_SEND;
    case AstNodeKind::kSendIf:
      return AST_NODE_KIND_SEND_IF;
    case AstNodeKind::kTestFunction:
      return AST_NODE_KIND_TEST_FUNCTION;
    case AstNodeKind::kTestProc:
      return AST_NODE_KIND_TEST_PROC;
    case AstNodeKind::kWildcardPattern:
      return AST_NODE_KIND_WILDCARD_PATTERN;
    case AstNodeKind::kWidthSlice:
      return AST_NODE_KIND_WIDTH_SLICE;
    case AstNodeKind::kMatchArm:
      return AST_NODE_KIND_MATCH_ARM;
    case AstNodeKind::kMatch:
      return AST_NODE_KIND_MATCH;
    case AstNodeKind::kAttr:
      return AST_NODE_KIND_ATTR;
    case AstNodeKind::kInstantiation:
      return AST_NODE_KIND_INSTANTIATION;
    case AstNodeKind::kInvocation:
      return AST_NODE_KIND_INVOCATION;
    case AstNodeKind::kSpawn:
      return AST_NODE_KIND_SPAWN;
    case AstNodeKind::kFormatMacro:
      return AST_NODE_KIND_FORMAT_MACRO;
    case AstNodeKind::kZeroMacro:
      return AST_NODE_KIND_ZERO_MACRO;
    case AstNodeKind::kAllOnesMacro:
      return AST_NODE_KIND_ALL_ONES_MACRO;
    case AstNodeKind::kSlice:
      return AST_NODE_KIND_SLICE;
    case AstNodeKind::kEnumDef:
      return AST_NODE_KIND_ENUM_DEF;
    case AstNodeKind::kStructDef:
      return AST_NODE_KIND_STRUCT_DEF;
    case AstNodeKind::kQuickCheck:
      return AST_NODE_KIND_QUICK_CHECK;
    case AstNodeKind::kXlsTuple:
      return AST_NODE_KIND_XLS_TUPLE;
    case AstNodeKind::kFor:
      return AST_NODE_KIND_FOR;
    case AstNodeKind::kCast:
      return AST_NODE_KIND_CAST;
    case AstNodeKind::kConstantDef:
      return AST_NODE_KIND_CONSTANT_DEF;
    case AstNodeKind::kLet:
      return AST_NODE_KIND_LET;
    case AstNodeKind::kChannelDecl:
      return AST_NODE_KIND_CHANNEL_DECL;
    case AstNodeKind::kParametricBinding:
      return AST_NODE_KIND_PARAMETRIC_BINDING;
    case AstNodeKind::kTupleIndex:
      return AST_NODE_KIND_TUPLE_INDEX;
    case AstNodeKind::kStatementBlock:
      return AST_NODE_KIND_STATEMENT_BLOCK;
    case AstNodeKind::kStatement:
      return AST_NODE_KIND_STATEMENT;
    case AstNodeKind::kUnrollFor:
      return AST_NODE_KIND_UNROLL_FOR;
    case AstNodeKind::kProcMember:
      return AST_NODE_KIND_PROC_MEMBER;
    case AstNodeKind::kRestOfTuple:
      return AST_NODE_KIND_REST_OF_TUPLE;
  }
  // Fatal since enum class values should not be out of range.
  LOG(FATAL) << "Out of range AstNodeKind: " << static_cast<int64_t>(kind);
}

PosProto ToProto(const Pos& pos, const FileTable& file_table) {
  PosProto proto;
  proto.set_filename(pos.GetFilename(file_table));
  proto.set_lineno(static_cast<int32_t>(pos.lineno()));
  proto.set_colno(static_cast<int32_t>(pos.colno()));
  return proto;
}

SpanProto ToProto(const Span& span, const FileTable& file_table) {
  SpanProto proto;
  *proto.mutable_start() = ToProto(span.start(), file_table);
  *proto.mutable_limit() = ToProto(span.limit(), file_table);
  return proto;
}

// Helper that turns a span of u8s into a std::string so it can be easily
// inserted into protobuf bytes fields.
std::string U8sToString(absl::Span<const uint8_t> bs) {
  return std::string(reinterpret_cast<const char*>(bs.data()), bs.size());
}

absl::StatusOr<InterpValueProto> ToProto(const InterpValue& v) {
  InterpValueProto proto;
  if (v.IsBits()) {
    BitsValueProto* bvp = proto.mutable_bits();
    bvp->set_is_signed(v.IsSBits());
    bvp->set_bit_count(static_cast<int32_t>(v.GetBitCount().value()));
    // Bits::ToBytes is in little-endian format. The proto stores data in
    // big-endian.
    std::vector<uint8_t> bytes = v.GetBitsOrDie().ToBytes();
    std::reverse(bytes.begin(), bytes.end());
    *bvp->mutable_data() = U8sToString(bytes);
  } else {
    return absl::UnimplementedError(
        "TypeInfoProto: convert InterpValue to proto: " + v.ToString());
  }
  return proto;
}

absl::StatusOr<ParametricExpressionProto> ToProto(const ParametricExpression& e,
                                                  const FileTable& file_table) {
  ParametricExpressionProto proto;
  if (const auto* s = dynamic_cast<const ParametricSymbol*>(&e)) {
    ParametricSymbolProto* psproto = proto.mutable_symbol();
    psproto->set_identifier(s->identifier());
    *psproto->mutable_span() = ToProto(s->span(), file_table);
    return proto;
  }
  if (const auto* s = dynamic_cast<const ParametricConstant*>(&e)) {
    ParametricConstantProto* pp = proto.mutable_constant();
    XLS_ASSIGN_OR_RETURN(*pp->mutable_constant(), ToProto(s->value()));
    return proto;
  }
  if (const auto* p = dynamic_cast<const ParametricMul*>(&e)) {
    ParametricMulProto* pp = proto.mutable_mul();
    XLS_ASSIGN_OR_RETURN(*pp->mutable_lhs(), ToProto(p->lhs(), file_table));
    XLS_ASSIGN_OR_RETURN(*pp->mutable_rhs(), ToProto(p->rhs(), file_table));
    return proto;
  }
  if (const auto* p = dynamic_cast<const ParametricAdd*>(&e)) {
    ParametricAddProto* pp = proto.mutable_add();
    XLS_ASSIGN_OR_RETURN(*pp->mutable_lhs(), ToProto(p->lhs(), file_table));
    XLS_ASSIGN_OR_RETURN(*pp->mutable_rhs(), ToProto(p->rhs(), file_table));
    return proto;
  }
  return absl::UnimplementedError(
      "TypeInfoToProto: convert ParametricExpression to proto: " +
      e.ToString());
}

absl::StatusOr<TypeDimProto> ToProto(const TypeDim& ctd,
                                     const FileTable& file_table) {
  TypeDimProto proto;
  if (std::holds_alternative<InterpValue>(ctd.value())) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_interp_value(),
                         ToProto(std::get<InterpValue>(ctd.value())));
  } else {
    auto& p = std::get<TypeDim::OwnedParametric>(ctd.value());
    XLS_ASSIGN_OR_RETURN(*proto.mutable_parametric(), ToProto(*p, file_table));
  }
  return proto;
}

absl::StatusOr<BitsTypeProto> ToProto(const BitsType& bits_type,
                                      const FileTable& file_table) {
  BitsTypeProto proto;
  proto.set_is_signed(bits_type.is_signed());
  XLS_ASSIGN_OR_RETURN(*proto.mutable_dim(),
                       ToProto(bits_type.size(), file_table));
  return proto;
}

absl::StatusOr<BitsConstructorTypeProto> ToProto(
    const BitsConstructorType& bits_type, const FileTable& file_table) {
  BitsConstructorTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_is_signed(),
                       ToProto(bits_type.is_signed(), file_table));
  return proto;
}

// Forward decl since this is co-recursive with the ToProto() for Type
// subtypes.
absl::StatusOr<TypeProto> ToProto(const Type& type,
                                  const FileTable& file_table);

absl::StatusOr<FunctionTypeProto> ToProto(const FunctionType& fn_type,
                                          const FileTable& file_table) {
  FunctionTypeProto proto;
  for (const std::unique_ptr<Type>& param : fn_type.params()) {
    XLS_ASSIGN_OR_RETURN(*proto.add_params(), ToProto(*param, file_table));
  }
  XLS_ASSIGN_OR_RETURN(*proto.mutable_return_type(),
                       ToProto(fn_type.return_type(), file_table));
  return proto;
}

absl::StatusOr<TupleTypeProto> ToProto(const TupleType& tuple_type,
                                       const FileTable& file_table) {
  TupleTypeProto proto;
  for (const std::unique_ptr<Type>& member : tuple_type.members()) {
    XLS_ASSIGN_OR_RETURN(*proto.add_members(), ToProto(*member, file_table));
  }
  return proto;
}

absl::StatusOr<ArrayTypeProto> ToProto(const ArrayType& array_type,
                                       const FileTable& file_table) {
  ArrayTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_size(),
                       ToProto(array_type.size(), file_table));
  XLS_ASSIGN_OR_RETURN(*proto.mutable_element_type(),
                       ToProto(array_type.element_type(), file_table));
  return proto;
}

absl::StatusOr<StructDefProto> ToProto(const StructDef& struct_def,
                                       const FileTable& file_table) {
  StructDefProto proto;
  *proto.mutable_span() = ToProto(struct_def.span(), file_table);
  proto.set_identifier(struct_def.identifier());
  proto.set_is_public(struct_def.is_public());
  for (int64_t i = 0; i < struct_def.size(); ++i) {
    proto.add_member_names(ToProtoString(struct_def.GetMemberName(i)));
  }
  return proto;
}

absl::StatusOr<StructTypeProto> ToProto(const StructType& struct_type,
                                        const FileTable& file_table) {
  StructTypeProto proto;
  for (const std::unique_ptr<Type>& member : struct_type.members()) {
    XLS_ASSIGN_OR_RETURN(*proto.add_members(), ToProto(*member, file_table));
  }
  XLS_ASSIGN_OR_RETURN(*proto.mutable_struct_def(),
                       ToProto(struct_type.nominal_type(), file_table));
  return proto;
}

absl::StatusOr<EnumDefProto> ToProto(const EnumDef& enum_def,
                                     const FileTable& file_table) {
  EnumDefProto proto;
  *proto.mutable_span() = ToProto(enum_def.span(), file_table);
  proto.set_identifier(enum_def.identifier());
  proto.set_is_public(enum_def.is_public());
  for (int64_t i = 0; i < enum_def.values().size(); ++i) {
    proto.add_member_names(ToProtoString(enum_def.GetMemberName(i)));
  }
  return proto;
}

absl::StatusOr<EnumTypeProto> ToProto(const EnumType& enum_type,
                                      const FileTable& file_table) {
  VLOG(5) << "Converting EnumType to proto: " << enum_type.ToString();
  EnumTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_enum_def(),
                       ToProto(enum_type.nominal_type(), file_table));
  XLS_ASSIGN_OR_RETURN(*proto.mutable_size(),
                       ToProto(enum_type.size(), file_table));
  proto.set_is_signed(enum_type.is_signed());
  VLOG(5) << "- proto: " << proto.ShortDebugString();
  return proto;
}

absl::StatusOr<MetaTypeProto> ToProto(const MetaType& meta_type,
                                      const FileTable& file_table) {
  VLOG(5) << "Converting MetaType to proto: " << meta_type.ToString();
  MetaTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_wrapped(),
                       ToProto(*meta_type.wrapped(), file_table));
  VLOG(5) << "- proto: " << proto.ShortDebugString();
  return proto;
}

ChannelDirectionProto extracted() {
  return ChannelDirectionProto::CHANNEL_DIRECTION_OUT;
}
ChannelDirectionProto ToProto(ChannelDirection d) {
  switch (d) {
    case ChannelDirection::kIn:
      return ChannelDirectionProto::CHANNEL_DIRECTION_IN;
    case ChannelDirection::kOut:
      return extracted();
  }
  LOG(FATAL) << "Invalid ChannelDirection: " << static_cast<int64_t>(d);
}

absl::StatusOr<ChannelTypeProto> ToProto(const ChannelType& channel_type,
                                         const FileTable& file_table) {
  VLOG(5) << "Converting ChannelType to proto: " << channel_type.ToString();
  ChannelTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_payload(),
                       ToProto(channel_type.payload_type(), file_table));
  proto.set_direction(ToProto(channel_type.direction()));
  VLOG(5) << "- proto: " << proto.ShortDebugString();
  return proto;
}

absl::StatusOr<TypeProto> ToProto(const Type& type,
                                  const FileTable& file_table) {
  TypeProto proto;
  if (const auto* bits = dynamic_cast<const BitsType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_bits_type(),
                         ToProto(*bits, file_table));
  } else if (const auto* bc = dynamic_cast<const BitsConstructorType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_bits_constructor_type(),
                         ToProto(*bc, file_table));
  } else if (const auto* fn = dynamic_cast<const FunctionType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_fn_type(), ToProto(*fn, file_table));
  } else if (const auto* tuple = dynamic_cast<const TupleType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_tuple_type(),
                         ToProto(*tuple, file_table));
  } else if (const auto* array = dynamic_cast<const ArrayType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_array_type(),
                         ToProto(*array, file_table));
  } else if (const auto* struct_type = dynamic_cast<const StructType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_struct_type(),
                         ToProto(*struct_type, file_table));
  } else if (const auto* enum_type = dynamic_cast<const EnumType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_enum_type(),
                         ToProto(*enum_type, file_table));
  } else if (const auto* meta_type = dynamic_cast<const MetaType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_meta_type(),
                         ToProto(*meta_type, file_table));
  } else if (const auto* channel_type =
                 dynamic_cast<const ChannelType*>(&type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_channel_type(),
                         ToProto(*channel_type, file_table));
  } else if (dynamic_cast<const TokenType*>(&type) != nullptr) {
    proto.mutable_token_type();
  } else {
    return absl::UnimplementedError("TypeInfoToProto: convert Type to proto: " +
                                    type.ToString());
  }
  return proto;
}

absl::StatusOr<AstNodeTypeInfoProto> ToProto(const AstNode& node,
                                             const Type& type) {
  const FileTable& file_table = *node.owner()->file_table();
  AstNodeTypeInfoProto proto;
  proto.set_kind(ToProto(node.kind()));
  if (std::optional<Span> maybe_span = node.GetSpan()) {
    *proto.mutable_span() = ToProto(maybe_span.value(), file_table);
  }
  XLS_ASSIGN_OR_RETURN(*proto.mutable_type(), ToProto(type, file_table));
  return proto;
}

std::string ToHumanString(const SpanProto& s) {
  return absl::StrFormat("%d:%d-%d:%d", s.start().lineno(), s.start().colno(),
                         s.limit().lineno(), s.limit().colno());
}

std::string ToHumanString(AstNodeKindProto kind) {
  return AstNodeKindProto_Name(kind).substr(
      std::string_view("AST_NODE_KIND_").size());
}

Pos FromProto(const PosProto& p, FileTable& file_table) {
  Fileno fileno = file_table.GetOrCreate(p.filename());
  return Pos(fileno, p.lineno(), p.colno());
}

Span FromProto(const SpanProto& p, FileTable& file_table) {
  return Span(FromProto(p.start(), file_table),
              FromProto(p.limit(), file_table));
}

absl::Span<const uint8_t> ToU8Span(const std::string& s) {
  return absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()),
                                   s.size());
}

absl::StatusOr<InterpValue> FromProto(const InterpValueProto& ivp) {
  switch (ivp.value_oneof_case()) {
    case InterpValueProto::ValueOneofCase::kBits: {
      std::vector<uint8_t> bytes;
      for (uint8_t i8 : ToU8Span(ivp.bits().data())) {
        bytes.push_back(i8);
      }
      // Bits::FromBytes expects data in little-endian format.
      std::reverse(bytes.begin(), bytes.end());
      return InterpValue::MakeBits(
          ivp.bits().is_signed(),
          Bits::FromBytes(bytes, ivp.bits().bit_count()));
    }
    default:
      break;
  }

  return absl::UnimplementedError(
      "TypeInfoFromProto: not yet implemented for "
      "InterpValueProto->InterpValue conversion: " +
      ivp.ShortDebugString());
}

std::unique_ptr<ParametricSymbol> FromProto(const ParametricSymbolProto& proto,
                                            FileTable& file_table) {
  return std::make_unique<ParametricSymbol>(
      proto.identifier(), FromProto(proto.span(), file_table));
}

absl::StatusOr<std::unique_ptr<ParametricExpression>> FromProto(
    const ParametricExpressionProto& proto, FileTable& file_table) {
  switch (proto.expr_oneof_case()) {
    case ParametricExpressionProto::ExprOneofCase::kSymbol: {
      return FromProto(proto.symbol(), file_table);
    }
    default:
      break;
  }
  return absl::UnimplementedError(
      "TypeInfoFromProto: not yet implemented for "
      "ParametricExpressionProto->ParametricExpression "
      "conversion: " +
      proto.ShortDebugString());
}

absl::StatusOr<TypeDim> FromProto(const TypeDimProto& ctdp,
                                  FileTable& file_table) {
  switch (ctdp.dim_oneof_case()) {
    case TypeDimProto::DimOneofCase::kInterpValue: {
      XLS_ASSIGN_OR_RETURN(InterpValue iv, FromProto(ctdp.interp_value()));
      return TypeDim(std::move(iv));
    }
    case TypeDimProto::DimOneofCase::kParametric: {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ParametricExpression> p,
                           FromProto(ctdp.parametric(), file_table));
      return TypeDim(std::move(p));
    }
    default:
      return absl::UnimplementedError(
          "TypeInfoFromProto: not yet implemented for "
          "TypeDimProto->TypeDim "
          "conversion: " +
          ctdp.ShortDebugString());
  }
}

absl::StatusOr<std::unique_ptr<Type>> FromProto(const TypeProto& ctp,
                                                const ImportData& import_data,
                                                FileTable& file_table) {
  VLOG(5) << "Converting TypeProto to C++: " << ctp.ShortDebugString();
  switch (ctp.type_oneof_case()) {
    case TypeProto::TypeOneofCase::kBitsType: {
      XLS_ASSIGN_OR_RETURN(TypeDim dim,
                           FromProto(ctp.bits_type().dim(), file_table));
      return std::make_unique<BitsType>(ctp.bits_type().is_signed(),
                                        std::move(dim));
    }
    case TypeProto::TypeOneofCase::kTupleType: {
      std::vector<std::unique_ptr<Type>> members;
      for (const TypeProto& member : ctp.tuple_type().members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> ct,
                             FromProto(member, import_data, file_table));
        members.push_back(std::move(ct));
      }
      return std::make_unique<TupleType>(std::move(members));
    }
    case TypeProto::TypeOneofCase::kArrayType: {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> element_type,
          FromProto(ctp.array_type().element_type(), import_data, file_table));
      XLS_ASSIGN_OR_RETURN(TypeDim size,
                           FromProto(ctp.array_type().size(), file_table));
      return std::make_unique<ArrayType>(std::move(element_type),
                                         std::move(size));
    }
    case TypeProto::TypeOneofCase::kEnumType: {
      const EnumTypeProto& etp = ctp.enum_type();
      const EnumDefProto& enum_def_proto = etp.enum_def();
      XLS_ASSIGN_OR_RETURN(TypeDim size,
                           FromProto(ctp.enum_type().size(), file_table));
      XLS_ASSIGN_OR_RETURN(const EnumDef* enum_def,
                           import_data.FindEnumDef(
                               FromProto(enum_def_proto.span(), file_table)));
      std::vector<InterpValue> members;
      for (const InterpValueProto& value : etp.members()) {
        XLS_ASSIGN_OR_RETURN(InterpValue member, FromProto(value));
        members.push_back(member);
      }

      return std::make_unique<EnumType>(*enum_def, std::move(size),
                                        /*is_signed=*/etp.is_signed(), members);
    }
    case TypeProto::TypeOneofCase::kFnType: {
      const FunctionTypeProto& ftp = ctp.fn_type();
      std::vector<std::unique_ptr<Type>> params;
      for (const TypeProto& param : ftp.params()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> ct,
                             FromProto(param, import_data, file_table));
        params.push_back(std::move(ct));
      }
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> rt,
          FromProto(ftp.return_type(), import_data, file_table));
      return std::make_unique<FunctionType>(std::move(params), std::move(rt));
    }
    case TypeProto::TypeOneofCase::kTokenType: {
      return std::make_unique<TokenType>();
    }
    case TypeProto::TypeOneofCase::kStructType: {
      const StructTypeProto& stp = ctp.struct_type();
      const StructDefProto& struct_def_proto = stp.struct_def();
      XLS_ASSIGN_OR_RETURN(const StructDef* struct_def,
                           import_data.FindStructDef(
                               FromProto(struct_def_proto.span(), file_table)));
      std::vector<std::unique_ptr<Type>> members;
      for (const TypeProto& member_proto : stp.members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> member,
                             FromProto(member_proto, import_data, file_table));
        members.push_back(std::move(member));
      }
      return std::make_unique<StructType>(std::move(members), *struct_def);
    }
    case TypeProto::TypeOneofCase::kMetaType: {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<Type> wrapped,
          FromProto(ctp.meta_type().wrapped(), import_data, file_table));
      return std::make_unique<MetaType>(std::move(wrapped));
    }
    default:
      return absl::UnimplementedError(
          "TypeInfoFromProto: not yet implemented for "
          "TypeProto->Type "
          "conversion: " +
          ctp.ShortDebugString());
  }
}

absl::StatusOr<std::string> ToHumanString(const TypeProto& ctp,
                                          const ImportData& import_data,
                                          FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> ct,
                       FromProto(ctp, import_data, file_table));
  return ct->ToString();
}

absl::StatusOr<AstNodeKind> FromProto(AstNodeKindProto p) {
  switch (p) {
    case AST_NODE_KIND_JOIN:
      return AstNodeKind::kJoin;
    case AST_NODE_KIND_TYPE_ANNOTATION:
      return AstNodeKind::kTypeAnnotation;
    case AST_NODE_KIND_MODULE:
      return AstNodeKind::kModule;
    case AST_NODE_KIND_NAME_DEF:
      return AstNodeKind::kNameDef;
    case AST_NODE_KIND_NAME_REF:
      return AstNodeKind::kNameRef;
    case AST_NODE_KIND_CONST_REF:
      return AstNodeKind::kConstRef;
    case AST_NODE_KIND_BUILTIN_NAME_DEF:
      return AstNodeKind::kBuiltinNameDef;
    case AST_NODE_KIND_CONDITIONAL:
      return AstNodeKind::kConditional;
    case AST_NODE_KIND_TYPE_ALIAS:
      return AstNodeKind::kTypeAlias;
    case AST_NODE_KIND_TYPE_REF:
      return AstNodeKind::kTypeRef;
    case AST_NODE_KIND_NUMBER:
      return AstNodeKind::kNumber;
    case AST_NODE_KIND_IMPORT:
      return AstNodeKind::kImport;
    case AST_NODE_KIND_UNOP:
      return AstNodeKind::kUnop;
    case AST_NODE_KIND_BINOP:
      return AstNodeKind::kBinop;
    case AST_NODE_KIND_COLON_REF:
      return AstNodeKind::kColonRef;
    case AST_NODE_KIND_PARAM:
      return AstNodeKind::kParam;
    case AST_NODE_KIND_FUNCTION:
      return AstNodeKind::kFunction;
    case AST_NODE_KIND_PROC:
      return AstNodeKind::kProc;
    case AST_NODE_KIND_ARRAY:
      return AstNodeKind::kArray;
    case AST_NODE_KIND_STRING:
      return AstNodeKind::kString;
    case AST_NODE_KIND_STRUCT_INSTANCE:
      return AstNodeKind::kStructInstance;
    case AST_NODE_KIND_NAME_DEF_TREE:
      return AstNodeKind::kNameDefTree;
    case AST_NODE_KIND_SPLAT_STRUCT_INSTANCE:
      return AstNodeKind::kSplatStructInstance;
    case AST_NODE_KIND_INDEX:
      return AstNodeKind::kIndex;
    case AST_NODE_KIND_RANGE:
      return AstNodeKind::kRange;
    case AST_NODE_KIND_RECV:
      return AstNodeKind::kRecv;
    case AST_NODE_KIND_RECV_NON_BLOCKING:
      return AstNodeKind::kRecvNonBlocking;
    case AST_NODE_KIND_RECV_IF:
      return AstNodeKind::kRecvIf;
    case AST_NODE_KIND_RECV_IF_NON_BLOCKING:
      return AstNodeKind::kRecvIfNonBlocking;
    case AST_NODE_KIND_SEND:
      return AstNodeKind::kSend;
    case AST_NODE_KIND_SEND_IF:
      return AstNodeKind::kSendIf;
    case AST_NODE_KIND_TEST_FUNCTION:
      return AstNodeKind::kTestFunction;
    case AST_NODE_KIND_TEST_PROC:
      return AstNodeKind::kTestProc;
    case AST_NODE_KIND_WILDCARD_PATTERN:
      return AstNodeKind::kWildcardPattern;
    case AST_NODE_KIND_WIDTH_SLICE:
      return AstNodeKind::kWidthSlice;
    case AST_NODE_KIND_MATCH_ARM:
      return AstNodeKind::kMatchArm;
    case AST_NODE_KIND_MATCH:
      return AstNodeKind::kMatch;
    case AST_NODE_KIND_ATTR:
      return AstNodeKind::kAttr;
    case AST_NODE_KIND_INSTANTIATION:
      return AstNodeKind::kInstantiation;
    case AST_NODE_KIND_INVOCATION:
      return AstNodeKind::kInvocation;
    case AST_NODE_KIND_SPAWN:
      return AstNodeKind::kSpawn;
    case AST_NODE_KIND_FORMAT_MACRO:
      return AstNodeKind::kFormatMacro;
    case AST_NODE_KIND_SLICE:
      return AstNodeKind::kSlice;
    case AST_NODE_KIND_ENUM_DEF:
      return AstNodeKind::kEnumDef;
    case AST_NODE_KIND_STRUCT_DEF:
      return AstNodeKind::kStructDef;
    case AST_NODE_KIND_QUICK_CHECK:
      return AstNodeKind::kQuickCheck;
    case AST_NODE_KIND_XLS_TUPLE:
      return AstNodeKind::kXlsTuple;
    case AST_NODE_KIND_FOR:
      return AstNodeKind::kFor;
    case AST_NODE_KIND_CAST:
      return AstNodeKind::kCast;
    case AST_NODE_KIND_CONSTANT_DEF:
      return AstNodeKind::kConstantDef;
    case AST_NODE_KIND_LET:
      return AstNodeKind::kLet;
    case AST_NODE_KIND_CHANNEL_DECL:
      return AstNodeKind::kChannelDecl;
    case AST_NODE_KIND_PARAMETRIC_BINDING:
      return AstNodeKind::kParametricBinding;
    case AST_NODE_KIND_TUPLE_INDEX:
      return AstNodeKind::kTupleIndex;
    case AST_NODE_KIND_STATEMENT_BLOCK:
      return AstNodeKind::kStatementBlock;
    case AST_NODE_KIND_UNROLL_FOR:
      return AstNodeKind::kUnrollFor;
    case AST_NODE_KIND_STATEMENT:
      return AstNodeKind::kStatement;
    case AST_NODE_KIND_ZERO_MACRO:
      return AstNodeKind::kZeroMacro;
    case AST_NODE_KIND_ALL_ONES_MACRO:
      return AstNodeKind::kAllOnesMacro;
    case AST_NODE_KIND_CONST_ASSERT:
      return AstNodeKind::kConstAssert;
    case AST_NODE_KIND_PROC_MEMBER:
      return AstNodeKind::kProcMember;
    case AST_NODE_KIND_REST_OF_TUPLE:
      return AstNodeKind::kRestOfTuple;
    // Note: since this is a proto enum there are sentinel values defined in
    // addition to the "real" above. Return an invalid argument error.
    case AST_NODE_KIND_INVALID:
    case AstNodeKindProto_INT_MIN_SENTINEL_DO_NOT_USE_:
    case AstNodeKindProto_INT_MAX_SENTINEL_DO_NOT_USE_:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown AstNodeKindProto: ", p));
  }
  ABSL_UNREACHABLE();
}

}  // namespace

absl::StatusOr<std::string> ToHumanString(const AstNodeTypeInfoProto& antip,
                                          const ImportData& import_data,
                                          FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(std::string type_str,
                       ToHumanString(antip.type(), import_data, file_table));
  XLS_ASSIGN_OR_RETURN(AstNodeKind kind, FromProto(antip.kind()));
  XLS_ASSIGN_OR_RETURN(
      const AstNode* n,
      import_data.FindNode(kind, FromProto(antip.span(), file_table)));
  std::string node_str = n == nullptr ? std::string("") : n->ToString();
  return absl::StrFormat("%s: %s :: `%s` :: %s", ToHumanString(antip.span()),
                         ToHumanString(antip.kind()), node_str, type_str);
}

absl::StatusOr<TypeInfoProto> TypeInfoToProto(const TypeInfo& type_info) {
  TypeInfoProto tip;
  // We have to sort the items in a stable way before adding them to the
  // repeated field in the proto.
  struct Item {
    Span span;
    AstNodeKind kind;
    const AstNode* node;
    const Type* type;
  };
  std::vector<Item> items;
  for (const auto& [node, type] : type_info.dict()) {
    items.push_back(
        Item{node->GetSpan().value(), node->kind(), node, type.get()});
  }
  std::sort(items.begin(), items.end(), [](const Item& lhs, const Item& rhs) {
    return std::make_tuple(lhs.span.start(), lhs.span.limit(),
                           static_cast<int>(lhs.kind)) <
           std::make_tuple(rhs.span.start(), rhs.span.limit(),
                           static_cast<int>(rhs.kind));
  });

  for (const Item& item : items) {
    XLS_ASSIGN_OR_RETURN(*tip.add_nodes(), ToProto(*item.node, *item.type));
  }
  return tip;
}

absl::StatusOr<std::string> ToHumanString(const TypeInfoProto& tip,
                                          const ImportData& import_data,
                                          FileTable& file_table) {
  std::vector<std::string> lines;
  for (int64_t i = 0; i < tip.nodes_size(); ++i) {
    XLS_ASSIGN_OR_RETURN(std::string node_str,
                         ToHumanString(tip.nodes(static_cast<int32_t>(i)),
                                       import_data, file_table));
    lines.push_back(std::move(node_str));
  }
  return absl::StrJoin(lines, "\n");
}

}  // namespace xls::dslx
