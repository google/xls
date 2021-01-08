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

#include "xls/dslx/ir_converter.h"

#include "absl/strings/str_replace.h"
#include "absl/types/variant.h"
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/ir/lsb_or_msb.h"

namespace xls::dslx {

using IrOp = xls::Op;
using IrLiteral = xls::Value;

/* static */ std::string IrConverter::ToString(const IrValue& value) {
  if (absl::holds_alternative<BValue>(value)) {
    return absl::StrFormat("%p", absl::get<BValue>(value).node());
  }
  return absl::StrFormat("%p", absl::get<CValue>(value).value.node());
}

IrConverter::IrConverter(const std::shared_ptr<Package>& package,
                         Module* module,
                         const std::shared_ptr<TypeInfo>& type_info,
                         bool emit_positions)
    : package_(package),
      module_(module),
      type_info_(type_info),
      emit_positions_(emit_positions),
      // TODO(leary): 2019-07-19 Create a way to get the file path from the
      // module.
      fileno_(package->GetOrCreateFileno("fake_file.x")) {
  XLS_VLOG(5) << "Constructed IR converter: " << this;
}

void IrConverter::InstantiateFunctionBuilder(absl::string_view mangled_name) {
  XLS_CHECK(function_builder_ == nullptr);
  function_builder_ =
      std::make_shared<FunctionBuilder>(mangled_name, package_.get());
}

void IrConverter::AddConstantDep(ConstantDef* constant_def) {
  XLS_VLOG(2) << "Adding consatnt dep: " << constant_def->ToString();
  constant_deps_.push_back(constant_def);
}

absl::StatusOr<BValue> IrConverter::DefAlias(AstNode* from, AstNode* to) {
  auto it = node_to_ir_.find(from);
  if (it == node_to_ir_.end()) {
    return absl::InternalError(absl::StrFormat(
        "Could not find AST node for aliasing: %s", from->ToString()));
  }
  IrValue value = it->second;
  XLS_VLOG(5) << absl::StreamFormat("Aliased node '%s' to be same as '%s': %s",
                                    to->ToString(), from->ToString(),
                                    ToString(value));
  node_to_ir_[to] = std::move(value);
  if (auto* name_def = dynamic_cast<NameDef*>(to)) {
    // Name the aliased node based on the identifier in the NameDef.
    if (absl::holds_alternative<BValue>(node_to_ir_.at(from))) {
      BValue ir_node = absl::get<BValue>(node_to_ir_.at(from));
      ir_node.SetName(name_def->identifier());
    }
  }
  return Use(to);
}

BValue IrConverter::Def(
    AstNode* node,
    const std::function<BValue(absl::optional<SourceLocation>)>& ir_func) {
  absl::optional<SourceLocation> loc;
  absl::optional<Span> span = node->GetSpan();
  if (emit_positions_ && span.has_value()) {
    const Pos& start_pos = span->start();
    Lineno lineno(start_pos.lineno());
    Colno colno(start_pos.colno());
    // TODO(leary): 2020-12-20 Figure out the fileno based on the module owner
    // of node.
    loc.emplace(fileno_, lineno, colno);
  }

  BValue result = ir_func(loc);
  XLS_VLOG(4) << absl::StreamFormat("Define node '%s' (%s) to be %s @ %s",
                                    node->ToString(), node->GetNodeTypeName(),
                                    ToString(result), SpanToString(span));
  SetNodeToIr(node, result);
  return result;
}

IrConverter::CValue IrConverter::DefConst(
    AstNode* node, IrLiteral ir_value,
    const std::function<BValue(absl::optional<SourceLocation>)>& ir_func) {
  absl::optional<SourceLocation> loc;
  absl::optional<Span> span = node->GetSpan();
  if (emit_positions_ && span.has_value()) {
    const Pos& start_pos = span->start();
    Lineno lineno(start_pos.lineno());
    Colno colno(start_pos.colno());
    // TODO(leary): 2020-12-20 Figure out the fileno based on the module owner
    // of node.
    loc.emplace(fileno_, lineno, colno);
  }

  BValue result = ir_func(loc);
  XLS_VLOG(4) << absl::StreamFormat("Define node '%s' (%s) to be %s @ %s",
                                    node->ToString(), node->GetNodeTypeName(),
                                    ToString(result), SpanToString(span));
  CValue c_value{ir_value, result};
  SetNodeToIr(node, c_value);
  return c_value;
}

absl::StatusOr<BValue> IrConverter::Use(AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Exception resolving %s node: %s",
                        node->GetNodeTypeName(), node->ToString()));
  }
  const IrValue& ir_value = it->second;
  XLS_VLOG(5) << absl::StreamFormat("Using node '%s' (%p) as IR value %s.",
                                    node->ToString(), node, ToString(ir_value));
  if (absl::holds_alternative<BValue>(ir_value)) {
    return absl::get<BValue>(ir_value);
  }
  XLS_RET_CHECK(absl::holds_alternative<CValue>(ir_value));
  return absl::get<CValue>(ir_value).value;
}

void IrConverter::SetNodeToIr(AstNode* node, IrValue value) {
  XLS_VLOG(5) << absl::StreamFormat("Setting node '%s' (%p) to IR value %s.",
                                    node->ToString(), node, ToString(value));
  node_to_ir_[node] = value;
}
absl::optional<IrConverter::IrValue> IrConverter::GetNodeToIr(
    AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::nullopt;
  }
  return it->second;
}

absl::Status IrConverter::HandleUnop(Unop* node) {
  XLS_ASSIGN_OR_RETURN(BValue operand, Use(node->operand()));
  switch (node->kind()) {
    case UnopKind::kNegate: {
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->AddUnOp(IrOp::kNeg, operand, loc);
      });
      return absl::OkStatus();
    }
    case UnopKind::kInvert: {
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->AddUnOp(IrOp::kNot, operand, loc);
      });
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid UnopKind: ", static_cast<int64>(node->kind())));
}

absl::Status IrConverter::HandleConcat(Binop* node, BValue lhs, BValue rhs) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> output_type,
                       ResolveType(node));
  std::vector<BValue> pieces = {lhs, rhs};
  if (dynamic_cast<BitsType*>(output_type.get()) != nullptr) {
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->Concat(pieces, loc);
    });
    return absl::OkStatus();
  }

  // Fallthrough case should be an ArrayType.
  auto* array_output_type = dynamic_cast<ArrayType*>(output_type.get());
  XLS_RET_CHECK(array_output_type != nullptr);
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->ArrayConcat(pieces, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBinop(Binop* node) {
  absl::optional<const ConcreteType*> lhs_type =
      type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* bits_type = dynamic_cast<const BitsType*>(lhs_type.value());
  bool signed_input = bits_type != nullptr && bits_type->is_signed();
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->rhs()));
  std::function<BValue(absl::optional<SourceLocation>)> ir_func;

  switch (node->kind()) {
    case BinopKind::kConcat:
      // Concat is handled out of line since it makes different IR ops for bits
      // and array kinds.
      return HandleConcat(node, lhs, rhs);
    // Arithmetic.
    case BinopKind::kAdd:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Add(lhs, rhs, loc);
      };
      break;
    case BinopKind::kSub:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Subtract(lhs, rhs, loc);
      };
      break;
    case BinopKind::kMul:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        if (signed_input) {
          return function_builder_->SMul(lhs, rhs, loc);
        }
        return function_builder_->UMul(lhs, rhs, loc);
      };
      break;
    case BinopKind::kDiv:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->UDiv(lhs, rhs, loc);
      };
      break;
    // Comparisons.
    case BinopKind::kEq:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Eq(lhs, rhs, loc);
      };
      break;
    case BinopKind::kNe:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Ne(lhs, rhs, loc);
      };
      break;
    case BinopKind::kGe:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        if (signed_input) {
          return function_builder_->SGe(lhs, rhs, loc);
        }
        return function_builder_->UGe(lhs, rhs, loc);
      };
      break;
    case BinopKind::kGt:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        if (signed_input) {
          return function_builder_->SGt(lhs, rhs, loc);
        }
        return function_builder_->UGt(lhs, rhs, loc);
      };
      break;
    case BinopKind::kLe:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        if (signed_input) {
          return function_builder_->SLe(lhs, rhs, loc);
        }
        return function_builder_->ULe(lhs, rhs, loc);
      };
      break;
    case BinopKind::kLt:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        if (signed_input) {
          return function_builder_->SLt(lhs, rhs, loc);
        }
        return function_builder_->ULt(lhs, rhs, loc);
      };
      break;
    // Shifts.
    case BinopKind::kShrl:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Shrl(lhs, rhs, loc);
      };
      break;
    case BinopKind::kShll:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Shll(lhs, rhs, loc);
      };
      break;
    case BinopKind::kShra:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Shra(lhs, rhs, loc);
      };
      break;
    // Bitwise.
    case BinopKind::kXor:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Xor(lhs, rhs, loc);
      };
      break;
    case BinopKind::kAnd:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->And(lhs, rhs, loc);
      };
      break;
    case BinopKind::kOr:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Or(lhs, rhs, loc);
      };
      break;
    // Logical.
    case BinopKind::kLogicalAnd:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->And(lhs, rhs, loc);
      };
      break;
    case BinopKind::kLogicalOr:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Or(lhs, rhs, loc);
      };
      break;
  }
  Def(node, ir_func);
  return absl::OkStatus();
}

absl::Status IrConverter::HandleAttr(Attr* node) {
  absl::optional<const ConcreteType*> lhs_type =
      type_info()->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* tuple_type = dynamic_cast<const TupleType*>(lhs_type.value());
  const std::string& identifier = node->attr()->identifier();
  XLS_ASSIGN_OR_RETURN(int64 index, tuple_type->GetMemberIndex(identifier));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));
  BValue ir = Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->TupleIndex(lhs, index, loc);
  });
  // Give the tuple-index instruction a meaningful name based on the identifier.
  if (lhs.HasAssignedName()) {
    ir.SetName(absl::StrCat(lhs.GetName(), "_", identifier));
  } else {
    ir.SetName(identifier);
  }
  return absl::OkStatus();
}

absl::Status IrConverter::HandleTernary(Ternary* node) {
  XLS_ASSIGN_OR_RETURN(BValue arg0, Use(node->test()));
  XLS_ASSIGN_OR_RETURN(BValue arg1, Use(node->consequent()));
  XLS_ASSIGN_OR_RETURN(BValue arg2, Use(node->alternate()));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Select(arg0, arg1, arg2, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinAndReduce(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->AndReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinBitSlice(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(Bits start_bits, GetConstBits(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(uint64 start, start_bits.ToUint64());
  XLS_ASSIGN_OR_RETURN(Bits width_bits, GetConstBits(node->args()[2]));
  XLS_ASSIGN_OR_RETURN(uint64 width, width_bits.ToUint64());
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->BitSlice(arg, start, width, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinClz(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Clz(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinCtz(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Ctz(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinOneHot(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue input, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(Bits lsb_prio, GetConstBits(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(uint64 lsb_prio_value, lsb_prio.ToUint64());

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->OneHot(
        input, lsb_prio_value ? LsbOrMsb::kLsb : LsbOrMsb::kMsb, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinOneHotSel(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue selector, Use(node->args()[0]));

  const Expr* cases_arg = node->args()[1];
  std::vector<BValue> cases;
  const auto* array = dynamic_cast<const Array*>(cases_arg);
  XLS_RET_CHECK_NE(array, nullptr);
  for (const auto& sel_case : array->members()) {
    XLS_ASSIGN_OR_RETURN(BValue bvalue_case, Use(sel_case));
    cases.push_back(bvalue_case);
  }

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->OneHotSelect(
        selector, cases, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinOrReduce(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->OrReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinRev(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Reverse(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinSignex(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));

  // Remember - it's the _type_ of the RHS of a signex that gives the new bit
  // count, not the value!
  auto* bit_count = dynamic_cast<Number*>(node->args()[1]);
  XLS_RET_CHECK_NE(bit_count, nullptr);
  XLS_RET_CHECK(bit_count->type());
  auto* type_annot = dynamic_cast<BuiltinTypeAnnotation*>(bit_count->type());
  int64 new_bit_count = type_annot->GetBitCount();

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->SignExtend(arg, new_bit_count, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinUpdate(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue index, Use(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(BValue new_value, Use(node->args()[2]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->ArrayUpdate(arg, new_value, {index}, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleBuiltinXorReduce(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->XorReduce(arg, loc);
  });
  return absl::OkStatus();
}


absl::StatusOr<ConcreteTypeDim> IrConverter::ResolveDim(ConcreteTypeDim dim) {
  while (
      absl::holds_alternative<ConcreteTypeDim::OwnedParametric>(dim.value())) {
    ParametricExpression& original =
        *absl::get<ConcreteTypeDim::OwnedParametric>(dim.value());
    ParametricExpression::Evaluated evaluated = original.Evaluate(
        ToParametricEnv(SymbolicBindings(symbolic_binding_map_)));
    dim = ConcreteTypeDim(std::move(evaluated));
  }
  return dim;
}

absl::StatusOr<std::unique_ptr<ConcreteType>> IrConverter::ResolveType(
    AstNode* node) {
  absl::optional<const ConcreteType*> t = type_info_->GetItem(node);
  if (!t.has_value()) {
    return ConversionErrorStatus(
        node->GetSpan(),
        absl::StrFormat(
            "Failed to convert IR because type was missing for AST node: %s",
            node->ToString()));
  }

  return t.value()->MapSize(
      [this](ConcreteTypeDim dim) { return ResolveDim(dim); });
}

absl::StatusOr<Bits> IrConverter::GetConstBits(AstNode* node) const {
  absl::optional<IrValue> ir_value = GetNodeToIr(node);
  if (!ir_value.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "AST node had no associated IR value: %s", node->ToString()));
  }
  if (!absl::holds_alternative<CValue>(*ir_value)) {
    return absl::InternalError(absl::StrFormat(
        "AST node had a non-const IR value: %s", node->ToString()));
  }
  return absl::get<CValue>(*ir_value).ir_value.GetBitsWithStatus();
}

absl::Status IrConverter::HandleConstantArray(ConstantArray* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  auto* array_type = dynamic_cast<ArrayType*>(type.get());

  std::vector<IrLiteral> values;
  for (Expr* n : node->members()) {
    // All elements are invariants of the given ConstantArray node.
    XLS_RET_CHECK(IsConstant(n));
    auto ir_value = GetNodeToIr(n);
    XLS_RET_CHECK(ir_value.has_value());
    XLS_RET_CHECK(absl::holds_alternative<CValue>(ir_value.value()));
    values.push_back(absl::get<CValue>(ir_value.value()).ir_value);
  }
  if (node->has_ellipsis()) {
    while (values.size() < absl::get<int64>(array_type->size().value())) {
      values.push_back(values.back());
    }
  }
  Value ir_value = IrLiteral::Array(std::move(values)).value();
  DefConst(node, ir_value, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Literal(ir_value, loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<std::string> MangleDslxName(
    absl::string_view function_name,
    const absl::btree_set<std::string>& free_keys, Module* module,
    SymbolicBindings* symbolic_bindings) {
  absl::btree_set<std::string> symbolic_bindings_keys;
  std::vector<int64> symbolic_bindings_values;
  if (symbolic_bindings != nullptr) {
    for (const SymbolicBinding& item : symbolic_bindings->bindings()) {
      symbolic_bindings_keys.insert(item.identifier);
      symbolic_bindings_values.push_back(item.value);
    }
  }
  absl::btree_set<std::string> difference;
  absl::c_set_difference(free_keys, symbolic_bindings_keys,
                         std::inserter(difference, difference.begin()));
  if (!difference.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Not enough symbolic bindings to convert function "
                        "'%s'; need {%s} got {%s}",
                        function_name, absl::StrJoin(free_keys, ", "),
                        absl::StrJoin(symbolic_bindings_keys, ", ")));
  }

  std::string module_name = absl::StrReplaceAll(module->name(), {{".", "_"}});
  if (symbolic_bindings == nullptr) {
    return absl::StrFormat("__%s__%s", module_name, function_name);
  }
  std::string suffix = absl::StrJoin(symbolic_bindings_values, "_");
  return absl::StrFormat("__%s__%s__%s", module_name, function_name, suffix);
}

absl::Status ConversionErrorStatus(const absl::optional<Span>& span,
                                   absl::string_view message) {
  return absl::InternalError(
      absl::StrFormat("ConversionErrorStatus: %s %s",
                      span ? span->ToString() : "<no span>", message));
}

absl::StatusOr<xls::Type*> IrConverter::ResolveTypeToIr(AstNode* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_type,
                       ResolveType(node));
  return TypeToIr(*concrete_type);
}

absl::StatusOr<xls::Type*> IrConverter::TypeToIr(
    const ConcreteType& concrete_type) {
  XLS_VLOG(4) << "Converting concrete type to IR: " << concrete_type;
  if (auto* array_type = dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(xls::Type * element_type,
                         TypeToIr(array_type->element_type()));
    int64 element_count = absl::get<int64>(array_type->size().value());
    xls::Type* result = package_->GetArrayType(element_count, element_type);
    XLS_VLOG(4) << "Converted type to IR; concrete type: " << concrete_type
                << " ir: " << result->ToString()
                << " element_count: " << element_count;
    return result;
  }
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    int64 bit_count = absl::get<int64>(bits_type->size().value());
    return package_->GetBitsType(bit_count);
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    int64 bit_count = absl::get<int64>(enum_type->size().value());
    return package_->GetBitsType(bit_count);
  }
  auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type);
  XLS_RET_CHECK(tuple_type != nullptr) << concrete_type;
  std::vector<xls::Type*> members;
  for (const ConcreteType* m : tuple_type->GetUnnamedMembers()) {
    XLS_ASSIGN_OR_RETURN(xls::Type * type, TypeToIr(*m));
    members.push_back(type);
  }
  return package_->GetTupleType(std::move(members));
}

}  // namespace xls::dslx
