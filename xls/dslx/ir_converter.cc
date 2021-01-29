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
#include "xls/dslx/cpp_extract_conversion_order.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/interpreter.h"
#include "xls/ir/lsb_or_msb.h"

namespace xls::dslx {
namespace internal {

// Helper that dispatches to the appropriate handler in the AST
class IrConverterVisitor : public AstNodeVisitor {
 public:
  explicit IrConverterVisitor(IrConverter* converter) : converter_(converter) {}

  // Causes node "n" to accept this visitor (basic double-dispatch).
  absl::Status Visit(AstNode* n) {
    XLS_VLOG(5) << this << " visiting: `" << n->ToString() << "` ("
                << n->GetNodeTypeName() << ")";
    return n->Accept(this);
  }

  // Causes all children of "node" to accept this visitor.
  absl::Status VisitChildren(AstNode* node) {
    for (AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(Visit(child));
    }
    return absl::OkStatus();
  }

  // A macro used for AST types where we want to visit all children, then call
  // the IrConverter handler (i.e. postorder traversal).
#define TRAVERSE_DISPATCH(__type)                      \
  absl::Status Handle##__type(__type* node) override { \
    XLS_RETURN_IF_ERROR(VisitChildren(node));          \
    return converter_->Handle##__type(node);           \
  }

  TRAVERSE_DISPATCH(Unop)
  TRAVERSE_DISPATCH(Binop)
  TRAVERSE_DISPATCH(Ternary)
  TRAVERSE_DISPATCH(XlsTuple)

  absl::Status HandleFor(For* node) override {
    auto visit = [this](AstNode* n) { return Visit(n); };
    auto visit_converter = [](IrConverter* converter,
                              AstNode* n) -> absl::Status {
      IrConverterVisitor visitor(converter);
      return visitor.Visit(n);
    };
    return converter_->HandleFor(node, visit, visit_converter);
  }

  // A macro used for AST types where we don't want to visit any children, just
  // call the IrConverter handler.
#define NO_TRAVERSE_DISPATCH(__type)                   \
  absl::Status Handle##__type(__type* node) override { \
    return converter_->Handle##__type(node);           \
  }

  NO_TRAVERSE_DISPATCH(Param)
  NO_TRAVERSE_DISPATCH(NameRef)
  NO_TRAVERSE_DISPATCH(ConstRef)
  NO_TRAVERSE_DISPATCH(Number)

  // A macro used for AST types where we don't want to visit any children, just
  // call the IrConverter handler (which accepts a "visit" callback).
#define NO_TRAVERSE_DISPATCH_VISIT(__type)                \
  absl::Status Handle##__type(__type* node) override {    \
    auto visit = [this](AstNode* n) { return Visit(n); }; \
    return converter_->Handle##__type(node, visit);       \
  }

  NO_TRAVERSE_DISPATCH_VISIT(Attr)
  NO_TRAVERSE_DISPATCH_VISIT(Array)
  NO_TRAVERSE_DISPATCH_VISIT(ConstantArray)
  NO_TRAVERSE_DISPATCH_VISIT(Cast)
  NO_TRAVERSE_DISPATCH_VISIT(ColonRef)
  NO_TRAVERSE_DISPATCH_VISIT(ConstantDef)
  NO_TRAVERSE_DISPATCH_VISIT(Index)
  NO_TRAVERSE_DISPATCH_VISIT(Invocation)
  NO_TRAVERSE_DISPATCH_VISIT(Let)
  NO_TRAVERSE_DISPATCH_VISIT(Match)
  NO_TRAVERSE_DISPATCH_VISIT(SplatStructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(StructInstance)

  // A macro used for AST types that we never expect to visit (if we do we
  // provide an error message noting it was unexpected).
#define INVALID(__type) \
  absl::Status Handle##__type(__type* node) { return Invalid(node); }

  // These are always custom-visited (i.e. traversed to in a specialized way
  // from their parent nodes).
  INVALID(NameDefTree)
  INVALID(ParametricBinding)
  INVALID(MatchArm)
  INVALID(WildcardPattern)
  INVALID(WidthSlice)
  INVALID(Slice)
  INVALID(NameDef)
  INVALID(TypeRef)
  INVALID(ArrayTypeAnnotation)
  INVALID(BuiltinTypeAnnotation)
  INVALID(TupleTypeAnnotation)
  INVALID(TypeRefTypeAnnotation)
  INVALID(TestFunction)

  // The visitor operates within a function, so none of these should be visible.
  INVALID(BuiltinNameDef)
  INVALID(EnumDef)
  INVALID(Import)
  INVALID(Function)
  INVALID(TypeDef)
  INVALID(Proc)
  INVALID(Module)
  INVALID(QuickCheck)
  INVALID(StructDef)

  // Unsupported for IR emission.
  INVALID(While)
  INVALID(Next)
  INVALID(Carry)

 private:
  // Called when we visit a node we don't expect to observe in the traversal.
  absl::Status Invalid(AstNode* node) {
    return absl::UnimplementedError(absl::StrFormat(
        "AST node unsupported for IR conversion: %s @ %s",
        node->GetNodeTypeName(), SpanToString(node->GetSpan())));
  }

  // The converter object we call back to for node handling.
  IrConverter* converter_;
};

/* static */ std::string IrConverter::ToString(const IrValue& value) {
  if (absl::holds_alternative<BValue>(value)) {
    return absl::StrFormat("%p", absl::get<BValue>(value).node());
  }
  return absl::StrFormat("%p", absl::get<CValue>(value).value.node());
}

IrConverter::IrConverter(Package* package, Module* module, TypeInfo* type_info,
                         ImportCache* import_cache, bool emit_positions)
    : package_(package),
      module_(module),
      type_info_(type_info),
      import_cache_(import_cache),
      emit_positions_(emit_positions),
      // TODO(leary): 2019-07-19 Create a way to get the file path from the
      // module.
      fileno_(package->GetOrCreateFileno("fake_file.x")) {
  XLS_VLOG(5) << "Constructed IR converter: " << this;
}

absl::StatusOr<xls::Function*> IrConverter::VisitFunction(
    Function* f, const SymbolicBindings* symbolic_bindings) {
  IrConverterVisitor visitor(this);
  auto visit = [&](AstNode* node) -> absl::Status {
    return node->Accept(&visitor);
  };
  return HandleFunction(f, symbolic_bindings, visit);
}

void IrConverter::InstantiateFunctionBuilder(absl::string_view mangled_name) {
  XLS_CHECK(!function_builder_.has_value());
  function_builder_.emplace(mangled_name, package_);
}

void IrConverter::AddConstantDep(ConstantDef* constant_def) {
  XLS_VLOG(2) << "Adding constant dep: " << constant_def->ToString();
  constant_deps_.push_back(constant_def);
}

absl::StatusOr<BValue> IrConverter::DefAlias(AstNode* from, AstNode* to) {
  XLS_RET_CHECK_NE(from, to);
  auto it = node_to_ir_.find(from);
  if (it == node_to_ir_.end()) {
    return absl::InternalError(absl::StrFormat(
        "TypeAliasError: %s Internal error: could not find AST node for "
        "aliasing: %s (%s) to: %s (%s)",
        SpanToString(from->GetSpan()), from->ToString(),
        from->GetNodeTypeName(), to->ToString(), to->GetNodeTypeName()));
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

absl::StatusOr<BValue> IrConverter::DefWithStatus(
    AstNode* node,
    const std::function<absl::StatusOr<BValue>(absl::optional<SourceLocation>)>&
        ir_func) {
  absl::optional<SourceLocation> loc = ToSourceLocation(node->GetSpan());
  XLS_ASSIGN_OR_RETURN(BValue result, ir_func(loc));
  XLS_VLOG(4) << absl::StreamFormat(
      "Define node '%s' (%s) to be %s @ %s", node->ToString(),
      node->GetNodeTypeName(), ToString(result), SpanToString(node->GetSpan()));
  SetNodeToIr(node, result);
  return result;
}

BValue IrConverter::Def(
    AstNode* node,
    const std::function<BValue(absl::optional<SourceLocation>)>& ir_func) {
  return DefWithStatus(node,
                       [&ir_func](absl::optional<SourceLocation> loc)
                           -> absl::StatusOr<BValue> { return ir_func(loc); })
      .value();
}

IrConverter::CValue IrConverter::DefConst(AstNode* node, xls::Value ir_value) {
  auto ir_func = [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Literal(ir_value, loc);
  };
  BValue result = Def(node, ir_func);
  CValue c_value{ir_value, result};
  SetNodeToIr(node, c_value);
  return c_value;
}

absl::StatusOr<BValue> IrConverter::Use(AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Could not resolve IR value for %s node: %s",
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
        return function_builder_->AddUnOp(xls::Op::kNeg, operand, loc);
      });
      return absl::OkStatus();
    }
    case UnopKind::kInvert: {
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->AddUnOp(xls::Op::kNot, operand, loc);
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

SymbolicBindings IrConverter::GetSymbolicBindingsTuple() const {
  absl::flat_hash_set<std::string> module_level_constant_identifiers;
  for (const ConstantDef* constant : module_->GetConstantDefs()) {
    module_level_constant_identifiers.insert(constant->identifier());
  }
  absl::flat_hash_map<std::string, int64> sans_module_level_constants;
  for (const auto& item : symbolic_binding_map_) {
    if (module_level_constant_identifiers.contains(item.first)) {
      continue;
    }
    sans_module_level_constants[item.first] = item.second;
  }
  return SymbolicBindings(std::move(sans_module_level_constants));
}

absl::Status IrConverter::HandleNumber(Number* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim dim, type->GetTotalBitCount());
  int64 bit_count = absl::get<int64>(dim.value());
  XLS_ASSIGN_OR_RETURN(Bits bits, node->GetBits(bit_count));
  DefConst(node, Value(bits));
  return absl::OkStatus();
}

absl::Status IrConverter::HandleXlsTuple(XlsTuple* node) {
  std::vector<BValue> operands;
  for (Expr* o : node->members()) {
    XLS_ASSIGN_OR_RETURN(BValue v, Use(o));
    operands.push_back(v);
  }
  Def(node, [this, &operands](absl::optional<SourceLocation> loc) {
    return function_builder_->Tuple(std::move(operands), loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleParam(Param* node) {
  XLS_VLOG(5) << "HandleParam: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(xls::Type * type, ResolveTypeToIr(node->type()));
  Def(node->name_def(), [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Param(node->identifier(), type);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleConstRef(ConstRef* node) {
  return DefAlias(node->name_def(), /*to=*/node).status();
}

absl::Status IrConverter::HandleNameRef(NameRef* node) {
  return DefAlias(ToAstNode(node->name_def()), /*to=*/node).status();
}

absl::Status IrConverter::HandleConstantDef(ConstantDef* node,
                                            const VisitFunc& visit) {
  XLS_VLOG(5) << "Visiting ConstantDef expr: " << node->value()->ToString();
  XLS_RETURN_IF_ERROR(visit(node->value()));
  XLS_VLOG(5) << "Aliasing NameDef for constant: "
              << node->name_def()->ToString();
  return DefAlias(node->value(), /*to=*/node->name_def()).status();
}

absl::Status IrConverter::HandleLet(Let* node, const VisitFunc& visit) {
  XLS_RETURN_IF_ERROR(visit(node->rhs()));
  if (node->name_def_tree()->is_leaf()) {
    XLS_RETURN_IF_ERROR(
        DefAlias(node->rhs(), /*to=*/ToAstNode(node->name_def_tree()->leaf()))
            .status());
    XLS_RETURN_IF_ERROR(visit(node->body()));
    XLS_RETURN_IF_ERROR(DefAlias(node->body(), node).status());
  } else {
    // Walk the tree of names we're trying to bind, performing tuple_index
    // operations on the RHS to get to the values we want to bind to those
    // names.
    XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->rhs()));
    std::vector<BValue> levels = {rhs};
    // Invoked at each level of the NameDefTree: binds the name in the
    // NameDefTree to the correponding value (being pattern matched).
    //
    // Args:
    //  x: Current subtree of the NameDefTree.
    //  level: Level (depth) in the NameDefTree, root is 0.
    //  index: Index of node in the current tree level (e.g. leftmost is 0).
    auto walk = [&](NameDefTree* x, int64 level, int64 index) -> absl::Status {
      levels.resize(level);
      levels.push_back(Def(x, [this, &levels, x,
                               index](absl::optional<SourceLocation> loc) {
        if (loc.has_value()) {
          loc = ToSourceLocation(x->is_leaf() ? ToAstNode(x->leaf())->GetSpan()
                                              : x->GetSpan());
        }
        return function_builder_->TupleIndex(levels.back(), index, loc);
      }));
      if (x->is_leaf()) {
        XLS_RETURN_IF_ERROR(DefAlias(x, ToAstNode(x->leaf())).status());
      }
      return absl::OkStatus();
    };

    XLS_RETURN_IF_ERROR(node->name_def_tree()->DoPreorder(walk));
    XLS_RETURN_IF_ERROR(visit(node->body()));
    XLS_RETURN_IF_ERROR(DefAlias(node->body(), /*to=*/node).status());
  }

  if (last_expression_ == nullptr) {
    last_expression_ = node->body();
  }
  return absl::OkStatus();
}

absl::Status IrConverter::HandleCast(Cast* node, const VisitFunc& visit) {
  XLS_RETURN_IF_ERROR(visit(node->expr()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> output_type,
                       ResolveType(node));
  if (auto* array_type = dynamic_cast<ArrayType*>(output_type.get())) {
    return CastToArray(node, *array_type);
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> input_type,
                       ResolveType(node->expr()));
  if (dynamic_cast<ArrayType*>(input_type.get()) != nullptr) {
    return CastFromArray(node, *output_type);
  }
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim new_bit_count_ctd,
                       output_type->GetTotalBitCount());
  int64 new_bit_count = absl::get<int64>(new_bit_count_ctd.value());
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim input_bit_count_ctd,
                       input_type->GetTotalBitCount());
  int64 old_bit_count = absl::get<int64>(input_bit_count_ctd.value());
  if (new_bit_count < old_bit_count) {
    auto bvalue_status = DefWithStatus(
        node,
        [this, node, new_bit_count](
            absl::optional<SourceLocation> loc) -> absl::StatusOr<BValue> {
          XLS_ASSIGN_OR_RETURN(BValue input, Use(node->expr()));
          return function_builder_->BitSlice(input, 0, new_bit_count);
        });
    XLS_RETURN_IF_ERROR(bvalue_status.status());
  } else {
    XLS_ASSIGN_OR_RETURN(bool signed_input, IsSigned(*input_type));
    auto bvalue_status = DefWithStatus(
        node,
        [this, node, new_bit_count, signed_input](
            absl::optional<SourceLocation> loc) -> absl::StatusOr<BValue> {
          XLS_ASSIGN_OR_RETURN(BValue input, Use(node->expr()));
          if (signed_input) {
            return function_builder_->SignExtend(input, new_bit_count);
          }
          return function_builder_->ZeroExtend(input, new_bit_count);
        });
    XLS_RETURN_IF_ERROR(bvalue_status.status());
  }
  return absl::OkStatus();
}

absl::Status IrConverter::HandleMatch(Match* node, const VisitFunc& visit) {
  if (node->arms().empty() ||
      !node->arms().back()->patterns()[0]->IsIrrefutable()) {
    return absl::UnimplementedError(absl::StrFormat(
        "ConversionError: %s Only matches with trailing irrefutable patterns "
        "are currently supported for IR conversion.",
        node->span().ToString()));
  }

  XLS_RETURN_IF_ERROR(visit(node->matched()));
  XLS_ASSIGN_OR_RETURN(BValue matched, Use(node->matched()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> matched_type,
                       ResolveType(node->matched()));

  MatchArm* default_arm = node->arms().back();
  if (default_arm->patterns().size() != 1) {
    return absl::UnimplementedError(
        absl::StrFormat("ConversionError: %s Multiple patterns in default arm "
                        "is not currently supported for IR conversion.",
                        node->span().ToString()));
  }
  XLS_RETURN_IF_ERROR(
      HandleMatcher(default_arm->patterns()[0],
                    {static_cast<int64>(node->arms().size()) - 1}, matched,
                    *matched_type, visit)
          .status());
  XLS_RETURN_IF_ERROR(visit(default_arm->expr()));

  std::vector<BValue> arm_selectors;
  std::vector<BValue> arm_values;
  for (int64 i = 0; i < node->arms().size() - 1; ++i) {
    MatchArm* arm = node->arms()[i];

    // Visit all the MatchArm's patterns.
    std::vector<BValue> this_arm_selectors;
    for (NameDefTree* pattern : arm->patterns()) {
      XLS_ASSIGN_OR_RETURN(
          BValue selector,
          HandleMatcher(pattern, {i}, matched, *matched_type, visit));
      this_arm_selectors.push_back(selector);
    }

    // "Or" together the patterns in this arm, if necessary, to determine if the
    // arm is selected.
    if (this_arm_selectors.size() > 1) {
      arm_selectors.push_back(function_builder_->AddNaryOp(
          Op::kOr, this_arm_selectors, ToSourceLocation(arm->span())));
    } else {
      arm_selectors.push_back(this_arm_selectors[0]);
    }
    XLS_RETURN_IF_ERROR(visit(arm->expr()));
    XLS_ASSIGN_OR_RETURN(BValue arm_rhs_value, Use(arm->expr()));
    arm_values.push_back(arm_rhs_value);
  }

  // So now we have the following representation of the match arms:
  //   match x {
  //     42  => blah
  //     64  => snarf
  //     128 => yep
  //     _   => burp
  //   }
  //
  //   selectors:     [x==42, x==64, x==128]
  //   values:        [blah,  snarf,    yep]
  //   default_value: burp
  XLS_ASSIGN_OR_RETURN(BValue default_value, Use(default_arm->expr()));
  SetNodeToIr(node, function_builder_->MatchTrue(arm_selectors, arm_values,
                                                 default_value));
  last_expression_ = node;
  return absl::OkStatus();
}

absl::StatusOr<int64> IrConverter::QueryConstRangeCall(For* node,
                                                       const VisitFunc& visit) {
  auto error = [&] {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConversionError: %s For-loop is of an unsupported "
                        "form for IR conversion; only a range(0, CONSTANT) "
                        "call is supported; got iterable: %s",
                        node->span().ToString(), node->iterable()->ToString()));
  };
  auto* iterable_call = dynamic_cast<Invocation*>(node->iterable());
  if (iterable_call == nullptr) {
    return error();
  }
  auto* callee_name_ref = dynamic_cast<NameRef*>(iterable_call->callee());
  if (callee_name_ref == nullptr) {
    return error();
  }
  if (!absl::holds_alternative<BuiltinNameDef*>(callee_name_ref->name_def())) {
    return error();
  }
  auto* builtin_name_def =
      absl::get<BuiltinNameDef*>(callee_name_ref->name_def());
  if (builtin_name_def->identifier() != "range") {
    return error();
  }

  XLS_RET_CHECK_EQ(iterable_call->args().size(), 2);
  Expr* start = iterable_call->args()[0];
  Expr* limit = iterable_call->args()[1];

  XLS_RETURN_IF_ERROR(visit(start));
  XLS_RETURN_IF_ERROR(visit(limit));

  XLS_ASSIGN_OR_RETURN(Bits start_bits, GetConstBits(start));
  if (!start_bits.IsZero()) {
    return error();
  }
  XLS_ASSIGN_OR_RETURN(Bits limit_bits, GetConstBits(limit));
  return limit_bits.ToUint64();
}

absl::Status IrConverter::HandleFor(
    For* node, const VisitFunc& visit,
    const VisitIrConverterFunc& visit_converter) {
  XLS_RETURN_IF_ERROR(visit(node->init()));

  // TODO(leary): We currently only support counted loops with fixed upper
  // bounds that start at zero; i.e. those of the form like:
  //
  //  for (i, ...): (u32, ...) in range(u32:0, N) {
  //    ...
  //  }
  XLS_ASSIGN_OR_RETURN(int64 trip_count, QueryConstRangeCall(node, visit));

  XLS_VLOG(5) << "Converting for-loop @ " << node->span();
  IrConverter body_converter(package_, module_, type_info_, import_cache_,
                             emit_positions_);
  body_converter.set_symbolic_binding_map(symbolic_binding_map_);

  // Note: there should be no name collisions (i.e. this name is unique)
  // because:
  //
  // a) Double underscore symbols are reserved for the compiler.
  //    TODO(leary): document this in the DSL reference.
  // b) The function name being built must be unique in the module.
  // c) The loop number bumps for each loop in that function.
  std::string body_fn_name =
      absl::StrFormat("__%s_counted_for_%d_body", function_builder_->name(),
                      GetAndBumpCountedForCount());
  body_converter.InstantiateFunctionBuilder(body_fn_name);
  std::vector<absl::variant<NameDefTree::Leaf, NameDefTree*>> flat =
      node->names()->Flatten1();
  if (flat.size() != 2) {
    return absl::UnimplementedError(
        "Expect for loop to have counter (induction variable) and carry data "
        "for IR conversion.");
  }

  auto to_ast_node =
      [](absl::variant<NameDefTree::Leaf, NameDefTree*> x) -> AstNode* {
    if (absl::holds_alternative<NameDefTree*>(x)) {
      return absl::get<NameDefTree*>(x);
    }
    return ToAstNode(absl::get<NameDefTree::Leaf>(x));
  };

  // Add the induction value.
  AstNode* ivar = to_ast_node(flat[0]);
  auto* name_def = dynamic_cast<NameDef*>(ivar);
  XLS_RET_CHECK(name_def != nullptr);
  XLS_ASSIGN_OR_RETURN(xls::Type * ivar_type, ResolveTypeToIr(name_def));
  body_converter.SetNodeToIr(name_def, body_converter.function_builder_->Param(
                                           name_def->identifier(), ivar_type));

  // Add the loop carry value.
  AstNode* carry = to_ast_node(flat[1]);
  if (auto* name_def = dynamic_cast<NameDef*>(carry)) {
    XLS_ASSIGN_OR_RETURN(xls::Type * type, ResolveTypeToIr(name_def));
    BValue param =
        body_converter.function_builder_->Param(name_def->identifier(), type);
    body_converter.SetNodeToIr(name_def, param);
  } else {
    // For tuple loop carries we have to destructure names on entry.
    NameDefTree* accum = absl::get<NameDefTree*>(flat[1]);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> carry_type,
                         ResolveType(accum));
    XLS_ASSIGN_OR_RETURN(xls::Type * carry_ir_type, TypeToIr(*carry_type));
    BValue carry =
        body_converter.function_builder_->Param("__loop_carry", carry_ir_type);
    body_converter.SetNodeToIr(accum, carry);
    XLS_RETURN_IF_ERROR(
        body_converter.HandleMatcher(accum, {}, carry, *carry_type, visit)
            .status());
  }

  // We need to capture the lexical scope and pass it to his loop body function.
  //
  // So we suffix free variables for the function body onto the function
  // parameters.
  FreeVariables freevars = node->body()->GetFreeVariables(node->span().start());
  freevars = freevars.DropBuiltinDefs();
  std::vector<NameDef*> relevant_name_defs;
  for (const auto& any_name_def : freevars.GetNameDefs()) {
    auto* name_def = absl::get<NameDef*>(any_name_def);
    absl::optional<const ConcreteType*> type = type_info_->GetItem(name_def);
    if (!type.has_value()) {
      continue;
    }
    if (dynamic_cast<const FunctionType*>(type.value()) != nullptr) {
      continue;
    }
    AstNode* definer = name_def->definer();
    if (dynamic_cast<EnumDef*>(definer) != nullptr ||
        dynamic_cast<TypeDef*>(definer) != nullptr) {
      continue;
    }
    relevant_name_defs.push_back(name_def);
    XLS_VLOG(5) << "Converting freevar name: " << name_def->ToString();
    XLS_ASSIGN_OR_RETURN(xls::Type * name_def_type, TypeToIr(**type));
    body_converter.SetNodeToIr(name_def,
                               body_converter.function_builder_->Param(
                                   name_def->identifier(), name_def_type));
  }

  XLS_RETURN_IF_ERROR(visit_converter(&body_converter, node->body()));
  XLS_ASSIGN_OR_RETURN(xls::Function * body_function,
                       body_converter.function_builder_->Build());
  XLS_VLOG(5) << "Converted body function: " << body_function->name();

  std::vector<BValue> invariant_args;
  for (NameDef* name_def : relevant_name_defs) {
    XLS_ASSIGN_OR_RETURN(BValue value, Use(name_def));
    invariant_args.push_back(value);
  }

  XLS_ASSIGN_OR_RETURN(BValue init, Use(node->init()));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->CountedFor(init, trip_count, /*stride=*/1,
                                         body_function, invariant_args);
  });
  return absl::OkStatus();
}

absl::StatusOr<BValue> IrConverter::HandleMatcher(
    NameDefTree* matcher, absl::Span<const int64> index,
    const BValue& matched_value, const ConcreteType& matched_type,
    const VisitFunc& visit) {
  if (matcher->is_leaf()) {
    NameDefTree::Leaf leaf = matcher->leaf();
    XLS_VLOG(5) << absl::StreamFormat("Matcher is leaf: %s (%s)",
                                      ToAstNode(leaf)->ToString(),
                                      ToAstNode(leaf)->GetNodeTypeName());
    if (absl::holds_alternative<WildcardPattern*>(leaf)) {
      return Def(matcher, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Literal(UBits(1, 1), loc);
      });
    } else if (absl::holds_alternative<Number*>(leaf) ||
               absl::holds_alternative<ColonRef*>(leaf)) {
      XLS_RETURN_IF_ERROR(visit(ToAstNode(leaf)));
      XLS_ASSIGN_OR_RETURN(BValue to_match, Use(ToAstNode(leaf)));
      return Def(matcher, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Eq(to_match, matched_value);
      });
    } else if (absl::holds_alternative<NameRef*>(leaf)) {
      // Comparing for equivalence to a (referenced) name.
      auto* name_ref = absl::get<NameRef*>(leaf);
      auto* name_def = absl::get<NameDef*>(name_ref->name_def());
      XLS_ASSIGN_OR_RETURN(BValue to_match, Use(name_def));
      BValue result = Def(matcher, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Eq(to_match, matched_value);
      });
      XLS_RETURN_IF_ERROR(DefAlias(name_def, name_ref).status());
      return result;
    } else {
      XLS_RET_CHECK(absl::holds_alternative<NameDef*>(leaf));
      auto* name_def = absl::get<NameDef*>(leaf);
      BValue ok = Def(name_def, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Literal(UBits(1, 1));
      });
      SetNodeToIr(matcher, matched_value);
      SetNodeToIr(ToAstNode(leaf), matched_value);
      return ok;
    }
  }

  auto* matched_tuple_type = dynamic_cast<const TupleType*>(&matched_type);
  BValue ok = function_builder_->Literal(UBits(/*value=*/1, /*bit_count=*/1));
  for (int64 i = 0; i < matched_tuple_type->size(); ++i) {
    const ConcreteType& element_type = matched_tuple_type->GetMemberType(i);
    NameDefTree* element = matcher->nodes()[i];
    BValue member = function_builder_->TupleIndex(matched_value, i);
    std::vector<int64> sub_index(index.begin(), index.end());
    sub_index.push_back(i);
    XLS_ASSIGN_OR_RETURN(BValue cond, HandleMatcher(element, sub_index, member,
                                                    element_type, visit));
    ok = function_builder_->And(ok, cond);
  }
  return ok;
}

absl::StatusOr<BValue> IrConverter::DefMapWithBuiltin(
    Invocation* parent_node, NameRef* node, AstNode* arg,
    const SymbolicBindings& symbolic_bindings) {
  XLS_ASSIGN_OR_RETURN(
      const std::string mangled_name,
      MangleDslxName(node->identifier(), {}, module_, &symbolic_bindings));
  XLS_ASSIGN_OR_RETURN(BValue arg_value, Use(arg));
  XLS_VLOG(5) << "Mapping with builtin; arg: "
              << arg_value.GetType()->ToString();
  auto* array_type = arg_value.GetType()->AsArrayOrDie();
  if (!package_->HasFunctionWithName(mangled_name)) {
    FunctionBuilder fb(mangled_name, package_);
    BValue param = fb.Param("arg", array_type->element_type());
    const std::string& builtin_name = node->identifier();
    BValue result;
    if (builtin_name == "clz") {
      result = fb.Clz(param);
    } else if (builtin_name == "ctz") {
      result = fb.Ctz(param);
    } else {
      return absl::InternalError("Invalid builtin name for map: " +
                                 builtin_name);
    }
    XLS_RETURN_IF_ERROR(fb.Build().status());
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * f, package_->GetFunction(mangled_name));
  return Def(parent_node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Map(arg_value, f);
  });
}

absl::StatusOr<BValue> IrConverter::HandleMap(Invocation* node,
                                              const VisitFunc& visit) {
  for (Expr* arg : node->args().subspan(0, node->args().size() - 1)) {
    XLS_RETURN_IF_ERROR(visit(arg));
  }
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Expr* fn_node = node->args()[1];
  XLS_VLOG(5) << "Function being mapped AST: " << fn_node->ToString();
  absl::optional<const SymbolicBindings*> node_sym_bindings =
      GetInvocationBindings(node);

  std::string map_fn_name;
  Module* lookup_module = nullptr;
  if (auto* name_ref = dynamic_cast<NameRef*>(fn_node)) {
    map_fn_name = name_ref->identifier();
    if (GetParametricBuiltins().contains(map_fn_name)) {
      XLS_VLOG(5) << "Map of parametric builtin: " << map_fn_name;
      return DefMapWithBuiltin(node, name_ref, node->args()[0],
                               *node_sym_bindings.value());
    }
    lookup_module = module_;
  } else if (auto* colon_ref = dynamic_cast<ColonRef*>(fn_node)) {
    map_fn_name = colon_ref->attr();
    absl::optional<Import*> import_node = colon_ref->ResolveImportSubject();
    absl::optional<const ImportedInfo*> info =
        type_info_->GetImported(*import_node);
    lookup_module = (*info)->module;
  } else {
    return absl::UnimplementedError("Unhandled function mapping: " +
                                    fn_node->ToString());
  }

  absl::optional<Function*> mapped_fn = lookup_module->GetFunction(map_fn_name);
  std::vector<std::string> free = (*mapped_fn)->GetFreeParametricKeys();
  absl::btree_set<std::string> free_set(free.begin(), free.end());
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName((*mapped_fn)->identifier(), free_set, lookup_module,
                     node_sym_bindings.value()));
  XLS_VLOG(5) << "Getting function with mangled name: " << mangled_name
              << " from package: " << package_->name();
  XLS_ASSIGN_OR_RETURN(xls::Function * f, package_->GetFunction(mangled_name));
  return Def(node, [&](absl::optional<SourceLocation> loc) -> BValue {
    return function_builder_->Map(arg, f, loc);
  });
}

absl::Status IrConverter::HandleIndex(Index* node, const VisitFunc& visit) {
  XLS_RETURN_IF_ERROR(visit(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));

  absl::optional<const ConcreteType*> lhs_type =
      type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  if (dynamic_cast<const TupleType*>(lhs_type.value()) != nullptr) {
    // Tuple indexing requires a compile-time-constant RHS.
    XLS_RETURN_IF_ERROR(visit(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(Bits rhs, GetConstBits(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(uint64 index, rhs.ToUint64());
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->TupleIndex(lhs, index, loc);
    });
  } else if (dynamic_cast<const BitsType*>(lhs_type.value()) != nullptr) {
    IndexRhs rhs = node->rhs();
    if (absl::holds_alternative<WidthSlice*>(rhs)) {
      auto* width_slice = absl::get<WidthSlice*>(rhs);
      XLS_RETURN_IF_ERROR(visit(width_slice->start()));
      XLS_ASSIGN_OR_RETURN(BValue start, Use(width_slice->start()));
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> output_type,
                           ResolveType(node));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim output_type_dim,
                           output_type->GetTotalBitCount());
      int64 width = absl::get<int64>(output_type_dim.value());
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->DynamicBitSlice(lhs, start, width, loc);
      });
    } else {
      auto* slice = absl::get<Slice*>(rhs);
      absl::optional<StartAndWidth> saw =
          type_info_->GetSliceStartAndWidth(slice, GetSymbolicBindingsTuple());
      XLS_RET_CHECK(saw.has_value());
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->BitSlice(lhs, saw->start, saw->width, loc);
      });
    }
  } else {
    XLS_RETURN_IF_ERROR(visit(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(BValue index, Use(ToAstNode(node->rhs())));
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->ArrayIndex(lhs, {index}, loc);
    });
  }
  return absl::OkStatus();
}

absl::Status IrConverter::HandleArray(Array* node, const VisitFunc& visit) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  const ArrayType* array_type = dynamic_cast<ArrayType*>(type.get());
  XLS_RET_CHECK(array_type != nullptr);
  std::vector<BValue> members;
  for (Expr* member : node->members()) {
    XLS_RETURN_IF_ERROR(visit(member));
    XLS_ASSIGN_OR_RETURN(BValue member_value, Use(member));
    members.push_back(member_value);
  }

  if (node->has_ellipsis()) {
    ConcreteTypeDim array_size_ctd = array_type->size();
    int64 array_size = absl::get<int64>(array_size_ctd.value());
    while (members.size() < array_size) {
      members.push_back(members.back());
    }
  }
  Def(node, [&](absl::optional<SourceLocation> loc) {
    xls::Type* type = members[0].GetType();
    return function_builder_->Array(std::move(members), type, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleInvocation(Invocation* node,
                                           const VisitFunc& visit) {
  XLS_ASSIGN_OR_RETURN(std::string called_name, GetCalleeIdentifier(node));
  auto accept_args = [&]() -> absl::StatusOr<std::vector<BValue>> {
    std::vector<BValue> values;
    for (Expr* arg : node->args()) {
      XLS_RETURN_IF_ERROR(visit(arg));
      XLS_ASSIGN_OR_RETURN(BValue value, Use(arg));
      values.push_back(value);
    }
    return values;
  };

  if (package_->HasFunctionWithName(called_name)) {
    XLS_ASSIGN_OR_RETURN(xls::Function * f, package_->GetFunction(called_name));
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());

    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->Invoke(args, f, loc);
    });
    return absl::OkStatus();
  }

  // A few builtins are handled specially.
  if (called_name == "fail!" || called_name == "trace") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 1)
        << called_name << " builtin only accepts a single argument";
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->Identity(args[0]);
    });
    return absl::OkStatus();
  }
  if (called_name == "map") {
    return HandleMap(node, visit).status();
  }

  // The rest of the builtins have "handle" methods we can resolve.
  absl::flat_hash_map<std::string, decltype(&IrConverter::HandleBuiltinClz)>
      map = {
          {"clz", &IrConverter::HandleBuiltinClz},
          {"ctz", &IrConverter::HandleBuiltinCtz},
          {"sgt", &IrConverter::HandleBuiltinSGt},
          {"sge", &IrConverter::HandleBuiltinSGe},
          {"slt", &IrConverter::HandleBuiltinSLt},
          {"sle", &IrConverter::HandleBuiltinSLe},
          {"signex", &IrConverter::HandleBuiltinSignex},
          {"one_hot", &IrConverter::HandleBuiltinOneHot},
          {"one_hot_sel", &IrConverter::HandleBuiltinOneHotSel},
          {"bit_slice", &IrConverter::HandleBuiltinBitSlice},
          {"rev", &IrConverter::HandleBuiltinRev},
          {"and_reduce", &IrConverter::HandleBuiltinAndReduce},
          {"or_reduce", &IrConverter::HandleBuiltinOrReduce},
          {"xor_reduce", &IrConverter::HandleBuiltinXorReduce},
          {"update", &IrConverter::HandleBuiltinUpdate},
      };
  auto it = map.find(called_name);
  if (it == map.end()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConversionError: %s Could not find name for "
                        "invocation: %s; available: [%s]",
                        node->span().ToString(), called_name,
                        absl::StrJoin(module_->GetFunctionNames(), ", ")));
  }
  XLS_RETURN_IF_ERROR(accept_args().status());
  auto f = it->second;
  return (this->*f)(node);
}

absl::StatusOr<xls::Function*> IrConverter::HandleFunction(
    Function* node, const SymbolicBindings* symbolic_bindings,
    const VisitFunc& visit) {
  XLS_VLOG(5) << "HandleFunction: " << node->ToString();
  if (symbolic_bindings != nullptr) {
    SetSymbolicBindings(symbolic_bindings);
  }

  // We use a function builder for the duration of converting this AST Function.
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(node->identifier(), node->GetFreeParametricKeySet(),
                     module_, symbolic_bindings));
  InstantiateFunctionBuilder(mangled_name);

  for (Param* param : node->params()) {
    XLS_RETURN_IF_ERROR(visit(param));
  }

  for (ParametricBinding* parametric_binding : node->parametric_bindings()) {
    XLS_VLOG(4) << "Resolving parametric binding: "
                << parametric_binding->ToString();

    absl::optional<int64> sb_value =
        get_symbolic_binding(parametric_binding->identifier());
    XLS_RET_CHECK(sb_value.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_type,
                         ResolveType(parametric_binding->type()));
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim parametric_width_ctd,
                         parametric_type->GetTotalBitCount());
    int64 bit_count = absl::get<int64>(parametric_width_ctd.value());
    DefConst(parametric_binding,
             Value(UBits(*sb_value, /*bit_count=*/bit_count)));
    XLS_RETURN_IF_ERROR(
        DefAlias(parametric_binding, /*to=*/parametric_binding->name_def())
            .status());
  }

  XLS_VLOG(3) << "Function has " << constant_deps_.size() << " constant deps";
  for (ConstantDef* dep : constant_deps_) {
    XLS_VLOG(4) << "Visiting constant dep: " << dep->ToString();
    XLS_RETURN_IF_ERROR(visit(dep));
  }
  ClearConstantDeps();

  XLS_VLOG(5) << "body: " << node->body()->ToString();
  XLS_RETURN_IF_ERROR(visit(node->body()));
  auto* last_expression =
      last_expression_ == nullptr ? node->body() : last_expression_;

  // If the last expression is a name reference, it may refer to
  // not-the-last-thing-we-function-built, so we go retrieve it explicitly and
  // make it the last thing.
  if (auto* name_ref = dynamic_cast<NameRef*>(last_expression)) {
    XLS_ASSIGN_OR_RETURN(BValue last_value, Use(name_ref));
    Def(last_expression, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->Identity(last_value, loc);
    });
  }
  XLS_ASSIGN_OR_RETURN(xls::Function * f, function_builder_->Build());
  XLS_VLOG(5) << "Built function: " << f->name();
  XLS_RETURN_IF_ERROR(VerifyFunction(f));
  return f;
}

absl::Status IrConverter::HandleColonRef(ColonRef* node,
                                         const VisitFunc& visit) {
  // Implementation note: ColonRef "invocation" are handled in Invocation (by
  // resolving the mangled callee name, which should have been IR converted in
  // dependency order).

  if (absl::optional<Import*> import = node->ResolveImportSubject()) {
    absl::optional<const ImportedInfo*> imported =
        type_info_->GetImported(import.value());
    XLS_RET_CHECK(imported.has_value());
    Module* imported_mod = (*imported)->module;
    XLS_ASSIGN_OR_RETURN(ConstantDef * constant_def,
                         imported_mod->GetConstantDef(node->attr()));
    XLS_RETURN_IF_ERROR(HandleConstantDef(constant_def, visit));
    return DefAlias(constant_def->name_def(), /*to=*/node).status();
  }

  EnumDef* enum_def;
  if (absl::holds_alternative<NameRef*>(node->subject())) {
    XLS_ASSIGN_OR_RETURN(enum_def,
                         DerefEnum(absl::get<NameRef*>(node->subject())));
  } else {
    XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                         ToTypeDefinition(ToAstNode(node->subject())));
    XLS_ASSIGN_OR_RETURN(enum_def, DerefEnum(type_definition));
  }
  XLS_ASSIGN_OR_RETURN(auto value, enum_def->GetValue(node->attr()));
  Expr* value_expr = ToExprNode(value);
  XLS_RETURN_IF_ERROR(visit(value_expr));
  return DefAlias(/*from=*/value_expr, /*to=*/node).status();
}

absl::Status IrConverter::HandleSplatStructInstance(SplatStructInstance* node,
                                                    const VisitFunc& visit) {
  XLS_RETURN_IF_ERROR(visit(node->splatted()));
  XLS_ASSIGN_OR_RETURN(BValue original, Use(node->splatted()));

  absl::flat_hash_map<std::string, BValue> updates;
  for (const auto& item : node->members()) {
    XLS_RETURN_IF_ERROR(visit(item.second));
    XLS_ASSIGN_OR_RETURN(updates[item.first], Use(item.second));
  }

  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefStruct(ToTypeDefinition(node->struct_ref())));
  std::vector<BValue> members;
  for (int64 i = 0; i < struct_def->members().size(); ++i) {
    const std::string& k = struct_def->GetMemberName(i);
    if (auto it = updates.find(k); it != updates.end()) {
      members.push_back(it->second);
    } else {
      members.push_back(function_builder_->TupleIndex(original, i));
    }
  }

  Def(node, [this, &members](absl::optional<SourceLocation> loc) {
    return function_builder_->Tuple(std::move(members), loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::HandleStructInstance(StructInstance* node,
                                               const VisitFunc& visit) {
  std::vector<BValue> operands;
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefStruct(ToTypeDefinition(node->struct_def())));
  std::vector<Value> const_operands;
  for (auto [_, member_expr] : node->GetOrderedMembers(struct_def)) {
    XLS_RETURN_IF_ERROR(visit(member_expr));
    XLS_ASSIGN_OR_RETURN(BValue operand, Use(member_expr));
    operands.push_back(operand);
  }

  Def(node, [this, &operands](absl::optional<SourceLocation> loc) {
    return function_builder_->Tuple(std::move(operands), loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<std::string> IrConverter::GetCalleeIdentifier(Invocation* node) {
  XLS_VLOG(5) << "Getting callee identifier for invocation: "
              << node->ToString();
  Expr* callee = node->callee();
  std::string callee_name;
  Module* m;
  if (auto* name_ref = dynamic_cast<NameRef*>(callee)) {
    callee_name = name_ref->identifier();
    m = module_;
  } else if (auto* colon_ref = dynamic_cast<ColonRef*>(callee)) {
    callee_name = colon_ref->attr();
    absl::optional<Import*> import = colon_ref->ResolveImportSubject();
    XLS_RET_CHECK(import.has_value());
    absl::optional<const ImportedInfo*> info = type_info_->GetImported(*import);
    m = (*info)->module;
  } else {
    return absl::InternalError("Invalid callee: " + callee->ToString());
  }

  absl::optional<Function*> f = m->GetFunction(callee_name);
  if (!f.has_value()) {
    // For e.g. builtins that are not in the module we just provide the name
    // directly.
    return callee_name;
  }

  absl::btree_set<std::string> free_keys = (*f)->GetFreeParametricKeySet();
  if (!(*f)->IsParametric()) {
    return MangleDslxName((*f)->identifier(), free_keys, m);
  }

  absl::optional<const SymbolicBindings*> resolved_symbolic_bindings =
      GetInvocationBindings(node);
  XLS_RET_CHECK(resolved_symbolic_bindings.has_value());
  XLS_VLOG(2) << absl::StreamFormat("Node `%s` (%s) @ %s symbolic bindings %s",
                                    node->ToString(), node->GetNodeTypeName(),
                                    node->span().ToString(),
                                    (*resolved_symbolic_bindings)->ToString());
  XLS_RET_CHECK(!(*resolved_symbolic_bindings)->empty());
  return MangleDslxName((*f)->identifier(), free_keys, m,
                        resolved_symbolic_bindings.value());
}

absl::Status IrConverter::HandleBinop(Binop* node) {
  XLS_VLOG(5) << "HandleBinop: " << node->ToString();
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

absl::Status IrConverter::HandleAttr(Attr* node, const VisitFunc& visit) {
  XLS_RETURN_IF_ERROR(visit(node->lhs()));
  absl::optional<const ConcreteType*> lhs_type =
      type_info_->GetItem(node->lhs());
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

absl::Status IrConverter::HandleBuiltinScmp(SignedCmp cmp, Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->args()[1]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    switch (cmp) {
      case SignedCmp::kLt:
        return function_builder_->SLt(lhs, rhs, loc);
      case SignedCmp::kGt:
        return function_builder_->SGt(lhs, rhs, loc);
      case SignedCmp::kLe:
        return function_builder_->SLe(lhs, rhs, loc);
      case SignedCmp::kGe:
        return function_builder_->SGe(lhs, rhs, loc);
    }
    XLS_LOG(FATAL) << "Invalid signed comparison: " << cmp;
  });
  return absl::OkStatus();
}

absl::Status IrConverter::CastToArray(Cast* node,
                                      const ArrayType& output_type) {
  XLS_ASSIGN_OR_RETURN(BValue bits, Use(node->expr()));
  std::vector<BValue> slices;
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim element_bit_count_dim,
                       output_type.element_type().GetTotalBitCount());
  int64 element_bit_count = absl::get<int64>(element_bit_count_dim.value());
  int64 array_size = absl::get<int64>(output_type.size().value());
  // MSb becomes lowest-indexed array element.
  for (int64 i = 0; i < array_size; ++i) {
    slices.push_back(function_builder_->BitSlice(bits, i * element_bit_count,
                                                 element_bit_count));
  }
  std::reverse(slices.begin(), slices.end());
  xls::Type* element_type = package_->GetBitsType(element_bit_count);
  Def(node, [this, &slices, element_type](absl::optional<SourceLocation> loc) {
    return function_builder_->Array(std::move(slices), element_type, loc);
  });
  return absl::OkStatus();
}

absl::Status IrConverter::CastFromArray(Cast* node,
                                        const ConcreteType& output_type) {
  XLS_ASSIGN_OR_RETURN(BValue array, Use(node->expr()));
  XLS_ASSIGN_OR_RETURN(xls::Type * input_type, ResolveTypeToIr(node->expr()));
  xls::ArrayType* array_type = input_type->AsArrayOrDie();
  const int64 array_size = array_type->size();
  std::vector<BValue> pieces;
  for (int64 i = 0; i < array_size; ++i) {
    BValue index = function_builder_->Literal(UBits(i, 32));
    pieces.push_back(function_builder_->ArrayIndex(array, {index}));
  }
  Def(node, [this, &pieces](absl::optional<SourceLocation> loc) {
    return function_builder_->Concat(std::move(pieces), loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<IrConverter::DerefVariant> IrConverter::DerefStructOrEnum(
    TypeDefinition node) {
  while (absl::holds_alternative<TypeDef*>(node)) {
    auto* type_def = absl::get<TypeDef*>(node);
    TypeAnnotation* annotation = type_def->type();
    if (auto* type_ref_annotation =
            dynamic_cast<TypeRefTypeAnnotation*>(annotation)) {
      node = type_ref_annotation->type_ref()->type_definition();
    } else {
      return absl::UnimplementedError(
          "Unhandled typedef for resolving to struct-or-enum: " +
          annotation->ToString());
    }
  }

  if (absl::holds_alternative<StructDef*>(node)) {
    return absl::get<StructDef*>(node);
  }
  if (absl::holds_alternative<EnumDef*>(node)) {
    return absl::get<EnumDef*>(node);
  }

  XLS_RET_CHECK(absl::holds_alternative<ColonRef*>(node));
  auto* colon_ref = absl::get<ColonRef*>(node);
  absl::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value());
  absl::optional<const ImportedInfo*> info = type_info_->GetImported(*import);
  Module* imported_mod = (*info)->module;
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       imported_mod->GetTypeDefinition(colon_ref->attr()));
  // Recurse to resolve the typedef within the imported module.
  return DerefStructOrEnum(td);
}

absl::StatusOr<StructDef*> IrConverter::DerefStruct(TypeDefinition node) {
  XLS_ASSIGN_OR_RETURN(DerefVariant v, DerefStructOrEnum(node));
  XLS_RET_CHECK(absl::holds_alternative<StructDef*>(v));
  return absl::get<StructDef*>(v);
}

absl::StatusOr<EnumDef*> IrConverter::DerefEnum(TypeDefinition node) {
  XLS_ASSIGN_OR_RETURN(DerefVariant v, DerefStructOrEnum(node));
  XLS_RET_CHECK(absl::holds_alternative<EnumDef*>(v));
  return absl::get<EnumDef*>(v);
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

absl::StatusOr<Value> IrConverter::GetConstValue(AstNode* node) const {
  absl::optional<IrValue> ir_value = GetNodeToIr(node);
  if (!ir_value.has_value()) {
    return absl::InternalError(
        absl::StrFormat("AST node had no associated IR value: %s @ %s",
                        node->ToString(), SpanToString(node->GetSpan())));
  }
  if (!absl::holds_alternative<CValue>(*ir_value)) {
    return absl::InternalError(absl::StrFormat(
        "AST node had a non-const IR value: %s", node->ToString()));
  }
  return absl::get<CValue>(*ir_value).ir_value;
}

absl::StatusOr<Bits> IrConverter::GetConstBits(AstNode* node) const {
  XLS_ASSIGN_OR_RETURN(Value value, GetConstValue(node));
  return value.GetBitsWithStatus();
}

absl::Status IrConverter::HandleConstantArray(ConstantArray* node,
                                              const VisitFunc& visit) {
  // Note: previously we would force constant evaluation here, but because all
  // constexprs should be evaluated during typechecking, we shouldn't need to
  // forcibly do constant evaluation at IR conversion time; therefore, we just
  // build BValues and let XLS opt constant fold them.
  return HandleArray(node, visit);
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
    XLS_ASSIGN_OR_RETURN(int64 element_count,
                         ResolveDimToInt(array_type->size()));
    xls::Type* result = package_->GetArrayType(element_count, element_type);
    XLS_VLOG(4) << "Converted type to IR; concrete type: " << concrete_type
                << " ir: " << result->ToString()
                << " element_count: " << element_count;
    return result;
  }
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int64 bit_count, ResolveDimToInt(bits_type->size()));
    return package_->GetBitsType(bit_count);
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    XLS_RET_CHECK(absl::holds_alternative<int64>(enum_type->size().value()));
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

// Notes that "constant_def" is a dependency on "converter", but first traverses
// to all free variables referred to by "constant_def"'s expression and adds
// them as dependencies first. This lets us constant expressions that refer to
// other constant expressions, since we can make sure that transitive dependency
// chain is all converted to IR.
static absl::Status AddConstantAndTransitives(ConstantDef* constant_def,
                                              IrConverter* converter) {
  FreeVariables free_variables =
      constant_def->value()->GetFreeVariables(constant_def->span().start());
  XLS_VLOG(4) << "Free variables of `" << constant_def->ToString() << "`: {"
              << absl::StrJoin(free_variables.Keys(), ", ") << "}";
  std::vector<std::pair<std::string, AnyNameDef>> freevars =
      free_variables.GetNameDefTuples();
  for (const auto& [identifier, any_name_def] : freevars) {
    if (absl::holds_alternative<BuiltinNameDef*>(any_name_def)) {
      continue;
    }
    auto* name_def = absl::get<NameDef*>(any_name_def);
    AstNode* definer = name_def->definer();
    XLS_VLOG(5) << "Definer of " << name_def->ToString() << ": "
                << (definer == nullptr ? std::string{"(nil)"}
                                       : definer->ToString());
    if (auto* constant_def = dynamic_cast<ConstantDef*>(definer)) {
      XLS_RETURN_IF_ERROR(AddConstantAndTransitives(constant_def, converter));
    }
  }
  converter->AddConstantDep(constant_def);
  return absl::OkStatus();
}

}  // namespace internal

// Converts a single function into its emitted text form.
//
// Args:
//   package: IR package we're converting the function into.
//   module: Module we're converting a function within.
//   function: Function we're converting.
//   type_info: Type information about module from the typechecking phase.
//   import_cache: Cache of modules potentially referenced by "module" above.
//   symbolic_bindings: Parametric bindings to use during conversion, if this
//     function is parametric.
//   emit_positions: Whether to emit position information into the IR based on
//     the AST's source positions.
//
// Returns an error status that indicates whether the conversion was successful.
// On success there will be a corresponding (built) function inside of
// "package".
static absl::Status ConvertOneFunctionInternal(
    Package* package, Module* module, Function* function, TypeInfo* type_info,
    ImportCache* import_cache, const SymbolicBindings* symbolic_bindings,
    bool emit_positions) {
  absl::flat_hash_map<std::string, Function*> function_by_name =
      module->GetFunctionByName();
  absl::flat_hash_map<std::string, ConstantDef*> constant_by_name =
      module->GetConstantByName();
  absl::flat_hash_map<std::string, TypeDefinition> type_definition_by_name =
      module->GetTypeDefinitionByName();
  absl::flat_hash_map<std::string, Import*> import_by_name =
      module->GetImportByName();

  internal::IrConverter converter(package, module, type_info, import_cache,
                                  emit_positions);

  FreeVariables free_variables =
      function->body()->GetFreeVariables(function->span().start());
  std::vector<std::pair<std::string, AnyNameDef>> freevars =
      free_variables.GetNameDefTuples();
  for (const auto& [identifier, any_name_def] : freevars) {
    if (function_by_name.contains(identifier) ||
        type_definition_by_name.contains(identifier) ||
        import_by_name.contains(identifier) ||
        absl::holds_alternative<BuiltinNameDef*>(any_name_def)) {
      continue;
    } else if (auto it = constant_by_name.find(identifier);
               it != constant_by_name.end()) {
      XLS_RETURN_IF_ERROR(AddConstantAndTransitives(it->second, &converter));
    } else {
      return absl::UnimplementedError(absl::StrFormat(
          "Cannot convert free variable: %s; not a function nor constant",
          identifier));
    }
  }

  auto set_to_string = [](const absl::btree_set<std::string>& s) {
    return absl::StrCat("{", absl::StrJoin(s, ", "), "}");
  };
  // TODO(leary): 2020-11-19 We use btrees in particular so this could use dual
  // iterators via the sorted property for O(n) superset comparison, but this
  // was easier to write and know it was correct on a first cut (couldn't find a
  // superset helper in absl's container algorithms at a first pass).
  auto is_superset = [](absl::btree_set<std::string> lhs,
                        const absl::btree_set<std::string>& rhs) {
    for (const auto& item : rhs) {
      lhs.erase(item);
    }
    return !lhs.empty();
  };

  absl::btree_set<std::string> symbolic_binding_keys;
  if (symbolic_bindings != nullptr) {
    symbolic_binding_keys = symbolic_bindings->GetKeySet();
  }
  absl::btree_set<std::string> f_parametric_keys =
      function->GetFreeParametricKeySet();
  if (is_superset(f_parametric_keys, symbolic_binding_keys)) {
    return absl::InternalError(absl::StrFormat(
        "Not enough symbolic bindings to convert function: %s; need %s got %s",
        function->identifier(), set_to_string(f_parametric_keys),
        set_to_string(symbolic_binding_keys)));
  }

  XLS_VLOG(3) << absl::StreamFormat("Converting function: %s",
                                    function->ToString());
  XLS_RETURN_IF_ERROR(
      converter.VisitFunction(function, symbolic_bindings).status());
  return absl::OkStatus();
}

absl::StatusOr<std::string> MangleDslxName(
    absl::string_view function_name,
    const absl::btree_set<std::string>& free_keys, Module* module,
    const SymbolicBindings* symbolic_bindings) {
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
  if (symbolic_bindings_values.empty()) {
    return absl::StrFormat("__%s__%s", module_name, function_name);
  }
  std::string suffix = absl::StrJoin(symbolic_bindings_values, "_");
  return absl::StrFormat("__%s__%s__%s", module_name, function_name, suffix);
}

absl::StatusOr<std::unique_ptr<Package>> ConvertModuleToPackage(
    Module* module, TypeInfo* type_info, ImportCache* import_cache,
    bool emit_positions, bool traverse_tests) {
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrder(module, type_info, traverse_tests));
  XLS_VLOG(3) << "Conversion order: ["
              << absl::StrJoin(
                     order, ", ",
                     [](std::string* out, const ConversionRecord& record) {
                       absl::StrAppend(out, record.ToString());
                     })
              << "]";
  auto package = absl::make_unique<Package>(module->name());
  for (const ConversionRecord& record : order) {
    XLS_VLOG(3) << "Converting to IR: " << record.ToString();
    XLS_RETURN_IF_ERROR(ConvertOneFunctionInternal(
        package.get(), record.m, record.f, record.type_info, import_cache,
        &record.bindings, emit_positions));
  }

  XLS_VLOG(3) << "Verifying converted package";
  XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
  return std::move(package);
}

absl::StatusOr<std::string> ConvertModule(Module* module, TypeInfo* type_info,
                                          ImportCache* import_cache,
                                          bool emit_positions) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Package> package,
      ConvertModuleToPackage(module, type_info, import_cache, emit_positions));
  return package->DumpIr();
}

absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, absl::string_view entry_function_name, TypeInfo* type_info,
    ImportCache* import_cache, const SymbolicBindings* symbolic_bindings,
    bool emit_positions) {
  Package package(module->name());
  absl::optional<Function*> f = module->GetFunction(entry_function_name);
  XLS_RET_CHECK(f.has_value());
  XLS_RETURN_IF_ERROR(ConvertOneFunctionInternal(
      &package, module, *f, type_info, import_cache,
      /*symbolic_bindings=*/nullptr, emit_positions));
  return package.DumpIr();
}

absl::StatusOr<Value> InterpValueToValue(const InterpValue& iv) {
  switch (iv.tag()) {
    case InterpValueTag::kSBits:
    case InterpValueTag::kUBits:
    case InterpValueTag::kEnum:
      return Value(iv.GetBitsOrDie());
    case InterpValueTag::kTuple:
    case InterpValueTag::kArray: {
      std::vector<Value> ir_values;
      for (const InterpValue& e : iv.GetValuesOrDie()) {
        XLS_ASSIGN_OR_RETURN(Value ir_value, InterpValueToValue(e));
        ir_values.push_back(std::move(ir_value));
      }
      if (iv.tag() == InterpValueTag::kTuple) {
        return Value::Tuple(std::move(ir_values));
      }
      return Value::Array(std::move(ir_values));
    }
    default:
      return absl::InvalidArgumentError(
          "Cannot convert interpreter value with tag: " +
          TagToString(iv.tag()));
  }
}

absl::StatusOr<InterpValue> ValueToInterpValue(const Value& v,
                                               const ConcreteType* type) {
  switch (v.kind()) {
    case ValueKind::kBits: {
      InterpValueTag tag = InterpValueTag::kUBits;
      if (type != nullptr) {
        XLS_RET_CHECK(type != nullptr);
        auto* bits_type = dynamic_cast<const BitsType*>(type);
        tag = bits_type->is_signed() ? InterpValueTag::kSBits
                                     : InterpValueTag::kUBits;
      }
      return InterpValue::MakeBits(tag, v.bits());
    }
    case ValueKind::kArray:
    case ValueKind::kTuple: {
      auto get_type = [&](int64 i) -> const ConcreteType* {
        if (type == nullptr) {
          return nullptr;
        }
        if (v.kind() == ValueKind::kArray) {
          auto* array_type = dynamic_cast<const ArrayType*>(type);
          XLS_CHECK(array_type != nullptr);
          return &array_type->element_type();
        }
        auto* tuple_type = dynamic_cast<const TupleType*>(type);
        XLS_CHECK(tuple_type != nullptr);
        return &tuple_type->GetMemberType(i);
      };
      std::vector<InterpValue> members;
      for (int64 i = 0; i < v.elements().size(); ++i) {
        const Value& e = v.elements()[i];
        XLS_ASSIGN_OR_RETURN(InterpValue iv,
                             ValueToInterpValue(e, get_type(i)));
        members.push_back(iv);
      }
      return InterpValue::MakeTuple(std::move(members));
    }
    default:
      return absl::InvalidArgumentError(
          "Cannot convert IR value to interpreter value: " + v.ToString());
  }
}

}  // namespace xls::dslx
