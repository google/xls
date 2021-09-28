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
    case AstNodeKind::kRecv:
      return AST_NODE_KIND_RECV;
    case AstNodeKind::kSend:
      return AST_NODE_KIND_SEND;
    case AstNodeKind::kTestFunction:
      return AST_NODE_KIND_TEST_FUNCTION;
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
    case AstNodeKind::kWhile:
      return AST_NODE_KIND_WHILE;
    case AstNodeKind::kCast:
      return AST_NODE_KIND_CAST;
    case AstNodeKind::kNext:
      return AST_NODE_KIND_NEXT;
    case AstNodeKind::kCarry:
      return AST_NODE_KIND_CARRY;
    case AstNodeKind::kConstantDef:
      return AST_NODE_KIND_CONSTANT_DEF;
    case AstNodeKind::kLet:
      return AST_NODE_KIND_LET;
    case AstNodeKind::kChannelDecl:
      return AST_NODE_KIND_CHANNEL_DECL;
    case AstNodeKind::kParametricBinding:
      return AST_NODE_KIND_PARAMETRIC_BINDING;
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
    *bvp->mutable_data() = U8sToString(v.GetBitsOrDie().ToBytes());
  } else {
    return absl::UnimplementedError("Convert InterpValue to proto: " +
                                    v.ToString());
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
  return absl::UnimplementedError("Convert ParametricExpression to proto: " +
                                  e.ToString());
}

absl::StatusOr<ConcreteTypeDimProto> ToProto(const ConcreteTypeDim& ctd) {
  ConcreteTypeDimProto proto;
  if (absl::holds_alternative<InterpValue>(ctd.value())) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_interp_value(),
                         ToProto(absl::get<InterpValue>(ctd.value())));
  } else {
    auto& p = absl::get<ConcreteTypeDim::OwnedParametric>(ctd.value());
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

absl::StatusOr<ConcreteTypeProto> ToProto(const ConcreteType& concrete_type) {
  ConcreteTypeProto proto;
  if (const auto* bits = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_bits(), ToProto(*bits));
  } else if (const auto* fn =
                 dynamic_cast<const FunctionType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_fn(), ToProto(*fn));
  } else if (const auto* tuple =
                 dynamic_cast<const TupleType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_tuple(), ToProto(*tuple));
  } else if (const auto* array =
                 dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_array(), ToProto(*array));
  } else if (const auto* struct_type =
                 dynamic_cast<const StructType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_structure(), ToProto(*struct_type));
  } else if (const auto* token_type =
                 dynamic_cast<const TokenType*>(&concrete_type)) {
    proto.mutable_token();
  } else {
    return absl::UnimplementedError("Convert ConcreteType to proto: " +
                                    concrete_type.ToString());
  }
  return proto;
}

absl::StatusOr<AstNodeTypeInfoProto> ToProto(
    const AstNode& node, const ConcreteType& concrete_type) {
  AstNodeTypeInfoProto proto;
  proto.set_kind(ToProto(node.kind()));
  if (absl::optional<Span> maybe_span = node.GetSpan()) {
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
      return InterpValue::MakeBits(
          ivp.bits().is_signed(),
          Bits::FromBytes(ToU8Span(ivp.bits().data()), ivp.bits().bit_count()));
    }
    default:
      break;
  }

  return absl::UnimplementedError(
      "Not yet implemented for InterpValueProto->InterpValue conversion: " +
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
      "Not yet implemented for ParametricExpressionProto->ParametricExpression "
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
          "Not yet implemented for ConcreteTypeDimProto->ConcreteTypeDim "
          "conversion: " +
          ctdp.ShortDebugString());
  }
}

absl::StatusOr<std::unique_ptr<ConcreteType>> FromProto(
    const ConcreteTypeProto& ctp, const Module& m) {
  switch (ctp.concrete_type_oneof_case()) {
    case ConcreteTypeProto::ConcreteTypeOneofCase::kBits: {
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim dim, FromProto(ctp.bits().dim()));
      return std::make_unique<BitsType>(ctp.bits().is_signed(), std::move(dim));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kTuple: {
      std::vector<std::unique_ptr<ConcreteType>> members;
      for (const ConcreteTypeProto& member : ctp.tuple().members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> ct,
                             FromProto(member, m));
        members.push_back(std::move(ct));
      }
      return std::make_unique<TupleType>(std::move(members));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kArray: {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> element_type,
                           FromProto(ctp.array().element_type(), m));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim size, FromProto(ctp.array().size()));
      return std::make_unique<ArrayType>(std::move(element_type),
                                         std::move(size));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kFn: {
      const FunctionTypeProto& ftp = ctp.fn();
      std::vector<std::unique_ptr<ConcreteType>> params;
      for (const ConcreteTypeProto& param : ftp.params()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> ct,
                             FromProto(param, m));
        params.push_back(std::move(ct));
      }
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> rt,
                           FromProto(ftp.return_type(), m));
      return std::make_unique<FunctionType>(std::move(params), std::move(rt));
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kToken: {
      return std::make_unique<TokenType>();
    }
    case ConcreteTypeProto::ConcreteTypeOneofCase::kStructure: {
      const StructTypeProto& stp = ctp.structure();
      const StructDefProto& struct_def_proto = stp.struct_def();
      const StructDef* struct_def =
          m.FindStructDef(FromProto(struct_def_proto.span()));
      if (struct_def == nullptr) {
        return absl::NotFoundError(
            absl::StrFormat("Structure definition not found in module %s: %s",
                            m.name(), struct_def_proto.identifier()));
      }
      std::vector<std::unique_ptr<ConcreteType>> members;
      for (const ConcreteTypeProto& member_proto : stp.members()) {
        XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> member,
                             FromProto(member_proto, m));
        members.push_back(std::move(member));
      }
      return std::make_unique<StructType>(std::move(members), *struct_def);
    }
    default:
      return absl::UnimplementedError(
          "Not yet implemented for ConcreteTypeProto->ConcreteType "
          "conversion: " +
          ctp.ShortDebugString());
  }
}

absl::StatusOr<std::string> ToHumanString(const ConcreteTypeProto& ctp,
                                          const Module& m) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> ct, FromProto(ctp, m));
  return ct->ToString();
}

absl::StatusOr<AstNodeKind> FromProto(AstNodeKindProto p) {
  switch (p) {
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
    case AST_NODE_KIND_RECV:
      return AstNodeKind::kRecv;
    case AST_NODE_KIND_SEND:
      return AstNodeKind::kSend;
    case AST_NODE_KIND_TEST_FUNCTION:
      return AstNodeKind::kTestFunction;
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
    case AST_NODE_KIND_WHILE:
      return AstNodeKind::kWhile;
    case AST_NODE_KIND_CAST:
      return AstNodeKind::kCast;
    case AST_NODE_KIND_NEXT:
      return AstNodeKind::kNext;
    case AST_NODE_KIND_CARRY:
      return AstNodeKind::kCarry;
    case AST_NODE_KIND_CONSTANT_DEF:
      return AstNodeKind::kConstantDef;
    case AST_NODE_KIND_LET:
      return AstNodeKind::kLet;
    case AST_NODE_KIND_CHANNEL_DECL:
      return AstNodeKind::kChannelDecl;
    case AST_NODE_KIND_PARAMETRIC_BINDING:
      return AstNodeKind::kParametricBinding;
    case AST_NODE_KIND_INVALID:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown AstNodeKindProto: ", p));
}

}  // namespace

absl::StatusOr<std::string> ToHumanString(const AstNodeTypeInfoProto& antip,
                                          const Module& m) {
  XLS_ASSIGN_OR_RETURN(std::string type_str, ToHumanString(antip.type(), m));
  XLS_ASSIGN_OR_RETURN(AstNodeKind kind, FromProto(antip.kind()));
  const AstNode* n = m.FindNode(kind, FromProto(antip.span()));
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

}  // namespace xls::dslx
