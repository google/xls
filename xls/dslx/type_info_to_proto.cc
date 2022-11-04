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

#include "xls/dslx/type_info_to_proto.h"

#include "xls/common/proto_adaptor_utils.h"

namespace xls::dslx {
namespace {

// Converts the AstNodeKind (C++ enum class) to its protobuf form.
AstNodeKindProto ToProto(AstNodeKind kind) {
  switch (kind) {
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
    case AstNodeKind::kTernary:
      return AST_NODE_KIND_TERNARY;
    case AstNodeKind::kTypeDef:
      return AST_NODE_KIND_TYPE_DEF;
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
    case AstNodeKind::kBlock:
      return AST_NODE_KIND_BLOCK;
    case AstNodeKind::kUnrollFor:
      return AST_NODE_KIND_UNROLL_FOR;
  }
  // Fatal since enum class values should not be out of range.
  XLS_LOG(FATAL) << "Out of range AstNodeKind: " << static_cast<int64_t>(kind);
}

PosProto ToProto(const Pos& pos) {
  PosProto proto;
  proto.set_filename(pos.filename());
  proto.set_lineno(pos.lineno());
  proto.set_colno(pos.colno());
  return proto;
}

SpanProto ToProto(const Span& span) {
  SpanProto proto;
  *proto.mutable_start() = ToProto(span.start());
  *proto.mutable_limit() = ToProto(span.limit());
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
    bvp->set_bit_count(v.GetBitCount().value());
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

absl::StatusOr<ParametricExpressionProto> ToProto(
    const ParametricExpression& e) {
  ParametricExpressionProto proto;
  if (const auto* s = dynamic_cast<const ParametricSymbol*>(&e)) {
    ParametricSymbolProto* psproto = proto.mutable_symbol();
    psproto->set_identifier(s->identifier());
    *psproto->mutable_span() = ToProto(s->span());
    return proto;
  }
  return absl::UnimplementedError(
      "TypeInfoToProto: convert ParametricExpression to proto: " +
      e.ToString());
}

absl::StatusOr<ConcreteTypeDimProto> ToProto(const ConcreteTypeDim& ctd) {
  ConcreteTypeDimProto proto;
  if (std::holds_alternative<InterpValue>(ctd.value())) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_interp_value(),
                         ToProto(std::get<InterpValue>(ctd.value())));
  } else {
    auto& p = std::get<ConcreteTypeDim::OwnedParametric>(ctd.value());
    XLS_ASSIGN_OR_RETURN(*proto.mutable_parametric(), ToProto(*p));
  }
  return proto;
}

absl::StatusOr<BitsTypeProto> ToProto(const BitsType& bits_type) {
  BitsTypeProto proto;
  proto.set_is_signed(bits_type.is_signed());
  XLS_ASSIGN_OR_RETURN(*proto.mutable_dim(), ToProto(bits_type.size()));
  return proto;
}

// Forward decl since this is co-recursive with the ToProto() for ConcreteType
// subtypes.
absl::StatusOr<ConcreteTypeProto> ToProto(const ConcreteType& concrete_type);

absl::StatusOr<FunctionTypeProto> ToProto(const FunctionType& fn_type) {
  FunctionTypeProto proto;
  for (const std::unique_ptr<ConcreteType>& param : fn_type.params()) {
    XLS_ASSIGN_OR_RETURN(*proto.add_params(), ToProto(*param));
  }
  XLS_ASSIGN_OR_RETURN(*proto.mutable_return_type(),
                       ToProto(fn_type.return_type()));
  return proto;
}

absl::StatusOr<TupleTypeProto> ToProto(const TupleType& tuple_type) {
  TupleTypeProto proto;
  for (const std::unique_ptr<ConcreteType>& member : tuple_type.members()) {
    XLS_ASSIGN_OR_RETURN(*proto.add_members(), ToProto(*member));
  }
  return proto;
}

absl::StatusOr<ArrayTypeProto> ToProto(const ArrayType& array_type) {
  ArrayTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_size(), ToProto(array_type.size()));
  XLS_ASSIGN_OR_RETURN(*proto.mutable_element_type(),
                       ToProto(array_type.element_type()));
  return proto;
}

absl::StatusOr<StructDefProto> ToProto(const StructDef& struct_def) {
  StructDefProto proto;
  *proto.mutable_span() = ToProto(struct_def.span());
  proto.set_identifier(struct_def.identifier());
  proto.set_is_public(struct_def.is_public());
  for (int64_t i = 0; i < struct_def.size(); ++i) {
    proto.add_member_names(ToProtoString(struct_def.GetMemberName(i)));
  }
  return proto;
}

absl::StatusOr<StructTypeProto> ToProto(const StructType& struct_type) {
  StructTypeProto proto;
  for (const std::unique_ptr<ConcreteType>& member : struct_type.members()) {
    XLS_ASSIGN_OR_RETURN(*proto.add_members(), ToProto(*member));
  }
  XLS_ASSIGN_OR_RETURN(*proto.mutable_struct_def(),
                       ToProto(struct_type.nominal_type()));
  return proto;
}

absl::StatusOr<EnumDefProto> ToProto(const EnumDef& enum_def) {
  EnumDefProto proto;
  *proto.mutable_span() = ToProto(enum_def.span());
  proto.set_identifier(enum_def.identifier());
  proto.set_is_public(enum_def.is_public());
  for (int64_t i = 0; i < enum_def.values().size(); ++i) {
    proto.add_member_names(ToProtoString(enum_def.GetMemberName(i)));
  }
  return proto;
}

absl::StatusOr<EnumTypeProto> ToProto(const EnumType& enum_type) {
  XLS_VLOG(5) << "Converting EnumType to proto: " << enum_type.ToString();
  EnumTypeProto proto;
  XLS_ASSIGN_OR_RETURN(*proto.mutable_enum_def(),
                       ToProto(enum_type.nominal_type()));
  XLS_ASSIGN_OR_RETURN(*proto.mutable_size(), ToProto(enum_type.size()));
  proto.set_is_signed(enum_type.signedness());
  XLS_VLOG(5) << "- proto: " << proto.ShortDebugString();
  return proto;
}

absl::StatusOr<ConcreteTypeProto> ToProto(const ConcreteType& concrete_type) {
  ConcreteTypeProto proto;
  if (const auto* bits = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_bits_type(), ToProto(*bits));
  } else if (const auto* fn =
                 dynamic_cast<const FunctionType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_fn_type(), ToProto(*fn));
  } else if (const auto* tuple =
                 dynamic_cast<const TupleType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_tuple_type(), ToProto(*tuple));
  } else if (const auto* array =
                 dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_array_type(), ToProto(*array));
  } else if (const auto* struct_type =
                 dynamic_cast<const StructType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_struct_type(), ToProto(*struct_type));
  } else if (const auto* enum_type =
                 dynamic_cast<const EnumType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_enum_type(), ToProto(*enum_type));
  } else if (dynamic_cast<const TokenType*>(&concrete_type) != nullptr) {
    proto.mutable_token_type();
  } else {
    return absl::UnimplementedError(
        "TypeInfoToProto: convert ConcreteType to proto: " +
        concrete_type.ToString());
  }
  return proto;
}

absl::StatusOr<AstNodeTypeInfoProto> ToProto(
    const AstNode& node, const ConcreteType& concrete_type) {
  AstNodeTypeInfoProto proto;
  proto.set_kind(ToProto(node.kind()));
  if (std::optional<Span> maybe_span = node.GetSpan()) {
    *proto.mutable_span() = ToProto(maybe_span.value());
  }
  XLS_ASSIGN_OR_RETURN(*proto.mutable_type(), ToProto(concrete_type));
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

Pos FromProto(const PosProto& p) {
  return Pos(p.filename(), p.lineno(), p.colno());
}

Span FromProto(const SpanProto& p) {
  return Span(FromProto(p.start()), FromProto(p.limit()));
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

std::unique_ptr<ParametricSymbol> FromProto(
    const ParametricSymbolProto& proto) {
  return std::make_unique<ParametricSymbol>(proto.identifier(),
                                            FromProto(proto.span()));
}

absl::StatusOr<std::unique_ptr<ParametricExpression>> FromProto(
    const ParametricExpressionProto& proto) {
  switch (proto.expr_oneof_case()) {
    case ParametricExpressionProto::ExprOneofCase::kSymbol: {
      return FromProto(proto.symbol());
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

absl::StatusOr<ConcreteTypeDim> FromProto(const ConcreteTypeDimProto& ctdp) {
  switch (ctdp.dim_oneof_case()) {
    case ConcreteTypeDimProto::DimOneofCase::kInterpValue: {
      XLS_ASSIGN_OR_RETURN(InterpValue iv, FromProto(ctdp.interp_value()));
      return ConcreteTypeDim(std::move(iv));
    }
    case ConcreteTypeDimProto::DimOneofCase::kParametric: {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ParametricExpression> p,
                           FromProto(ctdp.parametric()));
      return ConcreteTypeDim(std::move(p));
    }
    default:
      return absl::UnimplementedError(
          "TypeInfoFromProto: not yet implemented for "
          "ConcreteTypeDimProto->ConcreteTypeDim "
          "conversion: " +
          ctdp.ShortDebugString());
  }
}

absl::StatusOr<std::unique_ptr<ConcreteType>> FromProto(
    const ConcreteTypeProto& ctp, const ImportData& import_data) {
  XLS_VLOG(5) << "Converting ConcreteTypeProto to C++: "
              << ctp.ShortDebugString();
  switch (ctp.concrete_type_oneof_case()) {
    case ConcreteTypeProto::ConcreteTypeOneofCase::kBitsType: {
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim dim,
                           FromProto(ctp.bits_type().dim()));
      return std::make_unique<BitsType>(ctp.bits_type().is_signed(),
                                        std::move(dim));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kTupleType: {
      std::vector<std::unique_ptr<ConcreteType>> members;
      for (const ConcreteTypeProto& member : ctp.tuple_type().members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> ct,
                             FromProto(member, import_data));
        members.push_back(std::move(ct));
      }
      return std::make_unique<TupleType>(std::move(members));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kArrayType: {
      XLS_ASSIGN_OR_RETURN(
          std::unique_ptr<ConcreteType> element_type,
          FromProto(ctp.array_type().element_type(), import_data));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim size,
                           FromProto(ctp.array_type().size()));
      return std::make_unique<ArrayType>(std::move(element_type),
                                         std::move(size));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kEnumType: {
      const EnumTypeProto& etp = ctp.enum_type();
      const EnumDefProto& enum_def_proto = etp.enum_def();
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim size,
                           FromProto(ctp.enum_type().size()));
      XLS_ASSIGN_OR_RETURN(
          const EnumDef* enum_def,
          import_data.FindEnumDef(FromProto(enum_def_proto.span())));
      std::vector<InterpValue> members;
      for (const InterpValueProto& value : etp.members()) {
        XLS_ASSIGN_OR_RETURN(InterpValue member, FromProto(value));
        members.push_back(member);
      }

      return std::make_unique<EnumType>(*enum_def, std::move(size),
                                        /*is_signed=*/etp.is_signed(), members);
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kFnType: {
      const FunctionTypeProto& ftp = ctp.fn_type();
      std::vector<std::unique_ptr<ConcreteType>> params;
      for (const ConcreteTypeProto& param : ftp.params()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> ct,
                             FromProto(param, import_data));
        params.push_back(std::move(ct));
      }
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rt,
                           FromProto(ftp.return_type(), import_data));
      return std::make_unique<FunctionType>(std::move(params), std::move(rt));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kTokenType: {
      return std::make_unique<TokenType>();
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kStructType: {
      const StructTypeProto& stp = ctp.struct_type();
      const StructDefProto& struct_def_proto = stp.struct_def();
      XLS_ASSIGN_OR_RETURN(
          const StructDef* struct_def,
          import_data.FindStructDef(FromProto(struct_def_proto.span())));
      std::vector<std::unique_ptr<ConcreteType>> members;
      for (const ConcreteTypeProto& member_proto : stp.members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> member,
                             FromProto(member_proto, import_data));
        members.push_back(std::move(member));
      }
      return std::make_unique<StructType>(std::move(members), *struct_def);
    }
    default:
      return absl::UnimplementedError(
          "TypeInfoFromProto: not yet implemented for "
          "ConcreteTypeProto->ConcreteType "
          "conversion: " +
          ctp.ShortDebugString());
  }
}

absl::StatusOr<std::string> ToHumanString(const ConcreteTypeProto& ctp,
                                          const ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> ct,
                       FromProto(ctp, import_data));
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
    case AST_NODE_KIND_TERNARY:
      return AstNodeKind::kTernary;
    case AST_NODE_KIND_TYPE_DEF:
      return AstNodeKind::kTypeDef;
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
    case AST_NODE_KIND_BLOCK:
      return AstNodeKind::kBlock;
    case AST_NODE_KIND_UNROLL_FOR:
      return AstNodeKind::kUnrollFor;
    case AST_NODE_KIND_INVALID:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown AstNodeKindProto: ", p));
}

}  // namespace

absl::StatusOr<std::string> ToHumanString(const AstNodeTypeInfoProto& antip,
                                          const ImportData& import_data) {
  XLS_ASSIGN_OR_RETURN(std::string type_str,
                       ToHumanString(antip.type(), import_data));
  XLS_ASSIGN_OR_RETURN(AstNodeKind kind, FromProto(antip.kind()));
  XLS_ASSIGN_OR_RETURN(const AstNode* n,
                       import_data.FindNode(kind, FromProto(antip.span())));
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
    const ConcreteType* type;
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
                                          const ImportData& import_data) {
  std::vector<std::string> lines;
  for (int64_t i = 0; i < tip.nodes_size(); ++i) {
    XLS_ASSIGN_OR_RETURN(std::string node_str,
                         ToHumanString(tip.nodes(i), import_data));
    lines.push_back(std::move(node_str));
  }
  return absl::StrJoin(lines, "\n");
}

}  // namespace xls::dslx
