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
#include "xls/dslx/deduce_ctx.h"

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
  using IrFunc = std::function<BValue(FunctionBuilder&,
                                      absl::optional<SourceLocation>, BValue)>;
  switch (node->kind()) {
    case UnopKind::kNegate: {
      IrFunc add_neg = [](FunctionBuilder& fb,
                          absl::optional<SourceLocation> loc,
                          BValue x) { return fb.AddUnOp(IrOp::kNeg, x, loc); };
      Def(node, add_neg, operand);
      return absl::OkStatus();
    }
    case UnopKind::kInvert: {
      IrFunc add_not = [](FunctionBuilder& fb,
                          absl::optional<SourceLocation> loc,
                          BValue x) { return fb.AddUnOp(IrOp::kNot, x, loc); };
      Def(node, add_not, operand);
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid UnopKind: ", static_cast<int64>(node->kind())));
}

absl::Status IrConverter::HandleAttr(Attr* node) {
  absl::optional<const ConcreteType*> lhs_type =
      type_info()->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* tuple_type = dynamic_cast<const TupleType*>(lhs_type.value());
  const std::string& identifier = node->attr()->identifier();
  XLS_ASSIGN_OR_RETURN(int64 index, tuple_type->GetMemberIndex(identifier));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));
  using IrFunc = std::function<BValue(
      FunctionBuilder&, absl::optional<SourceLocation>, BValue, int64)>;
  IrFunc ir_func = [](FunctionBuilder& fb, absl::optional<SourceLocation> loc,
                      BValue lhs,
                      int64 rhs) { return fb.TupleIndex(lhs, rhs, loc); };
  BValue ir = Def(node, ir_func, lhs, index);
  // Give the tuple-index instruction a meaningful name based on the identifier.
  if (lhs.HasAssignedName()) {
    ir.SetName(absl::StrCat(lhs.GetName(), "_", identifier));
  } else {
    ir.SetName(identifier);
  }
  return absl::OkStatus();
}

absl::Status IrConverter::HandleTernary(Ternary* node) {
  using IrFunc =
      std::function<BValue(FunctionBuilder&, absl::optional<SourceLocation>,
                           BValue, BValue, BValue)>;
  IrFunc ir_func = [](FunctionBuilder& fb, absl::optional<SourceLocation> loc,
                      BValue arg0, BValue arg1,
                      BValue arg2) { return fb.Select(arg0, arg1, arg2, loc); };
  XLS_ASSIGN_OR_RETURN(BValue arg0, Use(node->test()));
  XLS_ASSIGN_OR_RETURN(BValue arg1, Use(node->consequent()));
  XLS_ASSIGN_OR_RETURN(BValue arg2, Use(node->alternate()));
  Def(node, ir_func, arg0, arg1, arg2);
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
  return absl::get<CValue>(*ir_value).integral;
}

absl::Status IrConverter::HandleConstantArray(ConstantArray* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  auto* array_type = dynamic_cast<ArrayType*>(type.get());

  std::vector<IrLiteral> values;
  for (Expr* n : node->members()) {
    XLS_ASSIGN_OR_RETURN(Bits e, GetConstBits(n));
    values.push_back(IrLiteral(std::move(e)));
  }
  if (node->has_ellipsis()) {
    while (values.size() < absl::get<int64>(array_type->size().value())) {
      values.push_back(values.back());
    }
  }
  using IrFunc = std::function<BValue(
      FunctionBuilder&, absl::optional<SourceLocation>, IrLiteral)>;
  IrFunc ir_func = [](FunctionBuilder& self, absl::optional<SourceLocation> loc,
                      IrLiteral literal) { return self.Literal(literal, loc); };
  Def(node, ir_func, IrLiteral::Array(std::move(values)).value());
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

}  // namespace xls::dslx
