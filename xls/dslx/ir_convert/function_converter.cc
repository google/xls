// Copyright 2023 The XLS Authors
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

#include "xls/dslx/ir_convert/function_converter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/convert_format_macro.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/ir_convert/ir_conversion_utils.h"
#include "xls/dslx/ir_convert/proc_config_ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/type_system/deduce_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"
#include "xls/ir/bits.h"
#include "xls/ir/fileno.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {
namespace {

constexpr WarningCollector* kNoWarningCollector = nullptr;

// For values that are generic sizes (e.g. indexing into an array) we prefer to
// use this bitwidth, which should suffice for all programs we can compile in
// effect.
//
// This corresponds to what we think of as our "usize" type, though we don't
// actually have that type yet, tracked in
// https://github.com/google/xls/issues/450
constexpr size_t kUsizeBits = 32;

// Returns a status that indicates an error in the IR conversion process.
absl::Status IrConversionErrorStatus(const std::optional<Span>& span,
                                     std::string_view message) {
  return absl::InternalError(
      absl::StrFormat("IrConversionError: %s %s",
                      span ? span->ToString() : "<no span>", message));
}

// Convert a NameDefTree node variant to an AstNode pointer (either the leaf
// node or the interior NameDefTree node).
AstNode* ToAstNode(const std::variant<NameDefTree::Leaf, NameDefTree*>& x) {
  if (std::holds_alternative<NameDefTree*>(x)) {
    return std::get<NameDefTree*>(x);
  }
  return ToAstNode(std::get<NameDefTree::Leaf>(x));
}

absl::StatusOr<TypeDefinition> ToTypeDefinition(
    const TypeAnnotation* type_annotation) {
  auto* type_ref_type_annotation =
      dynamic_cast<const TypeRefTypeAnnotation*>(type_annotation);
  XLS_RET_CHECK(type_annotation != nullptr);
  return type_ref_type_annotation->type_ref()->type_definition();
}

}  // namespace

absl::StatusOr<xls::Function*> EmitImplicitTokenEntryWrapper(
    xls::Function* implicit_token_f, dslx::Function* dslx_function, bool is_top,
    PackageInterfaceProto* interface_proto,
    const PackageInterfaceProto::Function& implicit_token_proto) {
  XLS_RET_CHECK_GE(implicit_token_f->params().size(), 2);
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(dslx_function->owner()->name(),
                     dslx_function->identifier(), CallingConvention::kTypical,
                     /*free_keys=*/{}, /*parametric_env=*/nullptr));
  PackageInterfaceProto::Function* wrapper_proto =
      interface_proto->add_functions();
  wrapper_proto->mutable_base()->set_name(mangled_name);
  FunctionBuilder fb(mangled_name, implicit_token_f->package(), true);
  fb.SetForeignFunctionData(dslx_function->extern_verilog_module());
  // Entry is a top entity.
  if (is_top) {
    XLS_RETURN_IF_ERROR(fb.SetAsTop());
  }
  // Clone all the params except for the leading `(token, bool)`.
  std::vector<BValue> params;
  for (const xls::Param* p : implicit_token_f->params().subspan(2)) {
    params.push_back(fb.Param(p->name(), p->GetType()));
    auto* param = wrapper_proto->add_parameters();
    param->set_name(p->name());
    *param->mutable_type() = p->GetType()->ToProto();
  }
  // Copy sv data
  for (int64_t i = 0; i < implicit_token_f->params().size() - 2; ++i) {
    if (implicit_token_proto.parameters(i + 2).has_sv_type()) {
      wrapper_proto->mutable_parameters(i)->set_sv_type(
          implicit_token_proto.parameters(i + 2).sv_type());
    }
  }

  // Invoke the function with the primordial "implicit token" values.
  BValue token = fb.AfterAll({});
  BValue activated = fb.Literal(Value::Bool(true));
  std::vector<BValue> args = {token, activated};
  args.insert(args.end(), params.begin(), params.end());

  // The built wrapper simply "exists" inside of the package as a side effect of
  // IR conversion, no need to return it back out to caller.
  BValue wrapped_result = fb.Invoke(args, implicit_token_f);
  XLS_RET_CHECK(wrapped_result.GetType()->IsTuple());
  BValue result = fb.TupleIndex(wrapped_result, 1);
  return fb.BuildWithReturnValue(result);
}

bool GetRequiresImplicitToken(dslx::Function& f, ImportData* import_data,
                              const ConvertOptions& options) {
  std::optional<bool> requires_opt =
      import_data->GetRootTypeInfo(f.owner()).value()->GetRequiresImplicitToken(
          f);
  CHECK(requires_opt.has_value());
  return requires_opt.value();
}

struct ScopedTypeInfoSwap {
  ScopedTypeInfoSwap(FunctionConverter* converter, TypeInfo* new_type_info)
      : converter_(converter),
        original_type_info_(converter_->current_type_info_) {
    CHECK(new_type_info != nullptr);
    converter_->current_type_info_ = new_type_info;
  }

  ~ScopedTypeInfoSwap() {
    converter_->current_type_info_ = original_type_info_;
  }

  FunctionConverter* converter_;
  TypeInfo* original_type_info_;
};

// RAII helper that establishes a control predicate for a lexical scope that
// chains onto the original control predicate.
//
// Automatically determines whether this is necessary at all by looking at
// whether the "parent" `FunctionConverter` has an implicit token calling
// convention, becomes a nop otherwise.
class ScopedControlPredicate {
 public:
  using ChainedPredicateFun = std::function<BValue(const PredicateFun&)>;

  ScopedControlPredicate(FunctionConverter* parent,
                         ChainedPredicateFun make_predicate)
      : parent_(parent), active_(parent->implicit_token_data_.has_value()) {
    if (!active_) {
      return;
    }
    orig_control_predicate_ =
        parent_->implicit_token_data_->create_control_predicate;
    // Curry "make_predicate" with the control-predicate-creating function we
    // had before -- this lets us build our "chain" of control predication.
    parent_->implicit_token_data_->create_control_predicate =
        [this, make_predicate = std::move(make_predicate)] {
          return make_predicate(orig_control_predicate_);
        };
  }

  ~ScopedControlPredicate() {
    if (!active_) {
      return;
    }
    parent_->implicit_token_data_->create_control_predicate =
        orig_control_predicate_;
  }

 private:
  FunctionConverter* parent_;

  // Whether implicit token calling convention is active for this conversion.
  const bool active_;

  // The original control predicate before entering this lexical scope, and that
  // is restored afterwards.
  PredicateFun orig_control_predicate_;
};

// Helper that dispatches to the appropriate FunctionConverter handler for the
// AST node being visited.
class FunctionConverterVisitor : public AstNodeVisitor {
 public:
  explicit FunctionConverterVisitor(FunctionConverter* converter)
      : converter_(converter) {}

  // Causes node "n" to accept this visitor (basic double-dispatch).
  absl::Status Visit(const AstNode* n) {
    VLOG(6) << this << " visiting: `" << n->ToString() << "` ("
            << n->GetNodeTypeName() << ")" << " @ "
            << SpanToString(n->GetSpan());
    return n->Accept(this);
  }

  // Causes all children of "node" to accept this visitor.
  absl::Status VisitChildren(const AstNode* node) {
    for (AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(Visit(child));
    }
    return absl::OkStatus();
  }

  absl::Status HandleConstAssert(const ConstAssert* node) override {
    // No need to IR-convert these, because they're checked by the typechecker.
    return absl::OkStatus();
  }

  // A macro used for AST types where we want to visit all children, then call
  // the FunctionConverter handler (i.e. postorder traversal).
#define TRAVERSE_DISPATCH(__type)                            \
  absl::Status Handle##__type(const __type* node) override { \
    XLS_RETURN_IF_ERROR(VisitChildren(node));                \
    return converter_->Handle##__type(node);                 \
  }

  TRAVERSE_DISPATCH(Unop)
  TRAVERSE_DISPATCH(Binop)
  TRAVERSE_DISPATCH(XlsTuple)
  TRAVERSE_DISPATCH(ZeroMacro)
  TRAVERSE_DISPATCH(AllOnesMacro)

  // A macro used for AST types where we don't want to visit any children, just
  // call the FunctionConverter handler.
#define NO_TRAVERSE_DISPATCH(__type)                         \
  absl::Status Handle##__type(const __type* node) override { \
    return converter_->Handle##__type(node);                 \
  }

  NO_TRAVERSE_DISPATCH(Param)
  NO_TRAVERSE_DISPATCH(NameRef)
  NO_TRAVERSE_DISPATCH(ConstRef)
  NO_TRAVERSE_DISPATCH(Number)
  NO_TRAVERSE_DISPATCH(String)

  // A macro used for AST types where we don't want to visit any children, just
  // call the FunctionConverter handler.
#define NO_TRAVERSE_DISPATCH_VISIT(__type)                   \
  absl::Status Handle##__type(const __type* node) override { \
    return converter_->Handle##__type(node);                 \
  }

  NO_TRAVERSE_DISPATCH_VISIT(Attr)
  NO_TRAVERSE_DISPATCH_VISIT(Array)
  NO_TRAVERSE_DISPATCH_VISIT(Block)
  NO_TRAVERSE_DISPATCH_VISIT(ConstantArray)
  NO_TRAVERSE_DISPATCH_VISIT(Cast)
  NO_TRAVERSE_DISPATCH_VISIT(ColonRef)
  NO_TRAVERSE_DISPATCH_VISIT(ConstantDef)
  NO_TRAVERSE_DISPATCH_VISIT(For)
  NO_TRAVERSE_DISPATCH_VISIT(FormatMacro)
  NO_TRAVERSE_DISPATCH_VISIT(Index)
  NO_TRAVERSE_DISPATCH_VISIT(Invocation)
  NO_TRAVERSE_DISPATCH_VISIT(Let)
  NO_TRAVERSE_DISPATCH_VISIT(Match)
  NO_TRAVERSE_DISPATCH_VISIT(Range)
  NO_TRAVERSE_DISPATCH_VISIT(SplatStructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(Statement)
  NO_TRAVERSE_DISPATCH_VISIT(StructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(Conditional)
  NO_TRAVERSE_DISPATCH_VISIT(TupleIndex)

  // A macro used for AST types that we never expect to visit (if we do we
  // provide an error message noting it was unexpected).
#define INVALID(__type)                                      \
  absl::Status Handle##__type(const __type* node) override { \
    return Invalid(node);                                    \
  }

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
  INVALID(ChannelTypeAnnotation)
  INVALID(TupleTypeAnnotation)
  INVALID(TypeRefTypeAnnotation)
  INVALID(TestFunction)
  INVALID(TestProc)

  // The visitor operates within a function, so none of these should be visible.
  INVALID(BuiltinNameDef)
  INVALID(ChannelDecl)
  INVALID(EnumDef)
  INVALID(Import)
  INVALID(Function)
  INVALID(TypeAlias)
  INVALID(Proc)
  INVALID(Module)
  INVALID(QuickCheck)
  INVALID(Spawn)
  INVALID(StructDef)
  INVALID(ProcMember)

  // This should have been unrolled into a sequence of statements and is
  // unconvertible.
  INVALID(UnrollFor)

 private:
  // Called when we visit a node we don't expect to observe in the traversal.
  absl::Status Invalid(const AstNode* node) {
    return absl::UnimplementedError(absl::StrFormat(
        "AST node unsupported for IR conversion: %s @ %s",
        node->GetNodeTypeName(), SpanToString(node->GetSpan())));
  }

  // The converter object we call back to for node handling.
  FunctionConverter* converter_;
};

absl::Status FunctionConverter::Visit(const AstNode* node) {
  FunctionConverterVisitor visitor(this);
  return visitor.Visit(node);
}

/* static */ std::string FunctionConverter::IrValueToString(
    const IrValue& value) {
  if (std::holds_alternative<BValue>(value)) {
    return absl::StrFormat("%p", std::get<BValue>(value).node());
  }

  if (std::holds_alternative<CValue>(value)) {
    return absl::StrFormat("%p", std::get<CValue>(value).value.node());
  }

  return absl::StrFormat("%p", std::get<Channel*>(value));
}

FunctionConverter::FunctionConverter(PackageData& package_data, Module* module,
                                     ImportData* import_data,
                                     ConvertOptions options,
                                     ProcConversionData* proc_data, bool is_top)
    : package_data_(package_data),
      module_(module),
      import_data_(import_data),
      options_(options),
      fileno_(module->fs_path().has_value()
                  ? package_data.conversion_info->package->GetOrCreateFileno(
                        std::string{module->fs_path().value()})
                  : Fileno(0)),
      proc_data_(proc_data),
      is_top_(is_top) {
  VLOG(5) << "Constructed IR converter: " << this;
}

bool FunctionConverter::GetRequiresImplicitToken(dslx::Function* f) const {
  return xls::dslx::GetRequiresImplicitToken(*f, import_data_, options_);
}

void FunctionConverter::SetFunctionBuilder(
    std::unique_ptr<BuilderBase> builder) {
  CHECK(function_builder_ == nullptr);
  function_proto_.reset();
  proc_proto_.reset();
  PackageInterfaceProto::FunctionBase* base;
  if (builder->function()->IsFunction()) {
    function_proto_ =
        package_data_.conversion_info->interface.add_functions();
    base = function_proto_.value()->mutable_base();
  } else {
    CHECK(builder->function()->IsProc())
        << "Building of block interfaces unsupported.";
    proc_proto_ = package_data_.conversion_info->interface.add_procs();
    base = proc_proto_.value()->mutable_base();
  }
  *base->mutable_name() = builder->function()->name();
  if (is_top_) {
    base->set_top(true);
  }
  function_builder_ = std::move(builder);
}

void FunctionConverter::AddConstantDep(ConstantDef* constant_def) {
  VLOG(2) << "Adding constant dep: " << constant_def->ToString();
  constant_deps_.push_back(constant_def);
}

absl::Status FunctionConverter::DefAlias(const AstNode* from,
                                         const AstNode* to) {
  XLS_RET_CHECK_NE(from, to);
  auto it = node_to_ir_.find(from);
  if (it == node_to_ir_.end()) {
    return absl::InternalError(absl::StrFormat(
        "TypeAliasError: %s internal error during IR conversion: could not "
        "find AST node for aliasing: %s (%s) to: %s (%s)",
        SpanToString(from->GetSpan()), from->ToString(),
        from->GetNodeTypeName(), to->ToString(), to->GetNodeTypeName()));
  }
  IrValue value = it->second;
  VLOG(6) << absl::StreamFormat(
      "Aliased node '%s' (%s) to be same as '%s' (%s): %s", to->ToString(),
      to->GetNodeTypeName(), from->ToString(), from->GetNodeTypeName(),
      IrValueToString(value));
  node_to_ir_[to] = std::move(value);
  if (const auto* name_def = dynamic_cast<const NameDef*>(to)) {
    // Name the aliased node based on the identifier in the NameDef.
    if (std::holds_alternative<BValue>(node_to_ir_.at(from))) {
      BValue ir_node = std::get<BValue>(node_to_ir_.at(from));
      ir_node.SetName(name_def->identifier());
    } else if (std::holds_alternative<CValue>(node_to_ir_.at(from))) {
      BValue ir_node = std::get<CValue>(node_to_ir_.at(from)).value;
      ir_node.SetName(name_def->identifier());
    }
    // Do nothing for channels; they have no BValue-type representation (they
    // exist as _global_ entities, not nodes in a Proc's IR), so they can't
    // be [re]named.
  }
  return absl::OkStatus();
}

absl::StatusOr<BValue> FunctionConverter::DefWithStatus(
    const AstNode* node,
    const std::function<absl::StatusOr<BValue>(const SourceInfo&)>& ir_func) {
  SourceInfo loc = ToSourceInfo(node->GetSpan());
  XLS_ASSIGN_OR_RETURN(BValue result, ir_func(loc));
  VLOG(6) << absl::StreamFormat("Define node '%s' (%s) to be %s @ %s",
                                node->ToString(), node->GetNodeTypeName(),
                                IrValueToString(result),
                                SpanToString(node->GetSpan()));

  // If there was an error in the function builder while defining this node
  // (which is an internal error -- the system-level expectation is that any
  // issues that would occur in IR building would have been caught previously in
  // type inference) then we flag them here to avoid returning husks of BValues
  // that callers can trip over and segfault (e.g. because they have a nullptr
  // type).
  XLS_RETURN_IF_ERROR(function_builder_->GetError());

  SetNodeToIr(node, result);
  return result;
}

BValue FunctionConverter::Def(
    const AstNode* node,
    const std::function<BValue(const SourceInfo&)>& ir_func) {
  return DefWithStatus(
             node,
             [&ir_func](const SourceInfo& loc) -> absl::StatusOr<BValue> {
               return ir_func(loc);
             })
      .value();
}

FunctionConverter::CValue FunctionConverter::DefConst(const AstNode* node,
                                                      xls::Value ir_value) {
  auto ir_func = [&](const SourceInfo& loc) {
    return function_builder_->Literal(ir_value, loc);
  };
  BValue result = Def(node, ir_func);
  CValue c_value{ir_value, result};
  SetNodeToIr(node, c_value);
  return c_value;
}

absl::StatusOr<BValue> FunctionConverter::Use(const AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Could not resolve IR value for %s node: %s",
                        node->GetNodeTypeName(), node->ToString()));
  }
  const IrValue& ir_value = it->second;
  VLOG(6) << absl::StreamFormat("Using node '%s' (%p) as IR value %s.",
                                node->ToString(), node,
                                IrValueToString(ir_value));
  if (std::holds_alternative<BValue>(ir_value)) {
    return std::get<BValue>(ir_value);
  }
  XLS_RET_CHECK(std::holds_alternative<CValue>(ir_value));
  return std::get<CValue>(ir_value).value;
}

void FunctionConverter::SetNodeToIr(const AstNode* node, IrValue value) {
  VLOG(6) << absl::StreamFormat("Setting node '%s' (%p) to IR value %s.",
                                node->ToString(), node, IrValueToString(value));
  node_to_ir_[node] = std::move(value);
}

std::optional<FunctionConverter::IrValue> FunctionConverter::GetNodeToIr(
    const AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status FunctionConverter::HandleUnop(const Unop* node) {
  XLS_ASSIGN_OR_RETURN(BValue operand, Use(node->operand()));
  switch (node->unop_kind()) {
    case UnopKind::kNegate: {
      Def(node, [&](const SourceInfo& loc) {
        return function_builder_->AddUnOp(xls::Op::kNeg, operand, loc);
      });
      return absl::OkStatus();
    }
    case UnopKind::kInvert: {
      Def(node, [&](const SourceInfo& loc) {
        return function_builder_->AddUnOp(xls::Op::kNot, operand, loc);
      });
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid UnopKind: ", static_cast<int64_t>(node->kind())));
}

absl::Status FunctionConverter::HandleConcat(const Binop* node, BValue lhs,
                                             BValue rhs) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> output_type, ResolveType(node));
  std::vector<BValue> pieces = {lhs, rhs};
  if (dynamic_cast<BitsType*>(output_type.get()) != nullptr) {
    Def(node, [&](const SourceInfo& loc) {
      return function_builder_->Concat(pieces, loc);
    });
    return absl::OkStatus();
  }

  // Fallthrough case should be an ArrayType.
  auto* array_output_type = dynamic_cast<ArrayType*>(output_type.get());
  XLS_RET_CHECK(array_output_type != nullptr);
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->ArrayConcat(pieces, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleEq(const Binop* node, BValue lhs,
                                         BValue rhs) {
  return DefWithStatus(node,
                       [&](const SourceInfo& loc) -> absl::StatusOr<BValue> {
                         return function_builder_->Eq(lhs, rhs, loc);
                       })
      .status();
}

absl::Status FunctionConverter::HandleNe(const Binop* node, BValue lhs,
                                         BValue rhs) {
  return DefWithStatus(node,
                       [&](const SourceInfo& loc) -> absl::StatusOr<BValue> {
                         return function_builder_->Ne(lhs, rhs, loc);
                       })
      .status();
}

absl::Status FunctionConverter::HandleNumber(const Number* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, ResolveType(node));
  XLS_RET_CHECK(!type->IsMeta());
  XLS_ASSIGN_OR_RETURN(TypeDim dim, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                       std::get<InterpValue>(dim.value()).GetBitValueViaSign());
  XLS_ASSIGN_OR_RETURN(Bits bits, node->GetBits(bit_count));
  DefConst(node, Value(bits));
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleString(const String* node) {
  std::vector<Value> elements;
  for (const uint8_t letter : node->text()) {
    elements.push_back(Value(UBits(letter, /*bit_count=*/8)));
  }
  XLS_ASSIGN_OR_RETURN(Value array, Value::Array(elements));
  DefConst(node, array);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleTupleIndex(const TupleIndex* node) {
  XLS_RETURN_IF_ERROR(Visit(ToAstNode(node->lhs())));
  XLS_ASSIGN_OR_RETURN(BValue v, Use(node->lhs()));
  XLS_RETURN_IF_ERROR(Visit(ToAstNode(node->index())));
  XLS_ASSIGN_OR_RETURN(Bits rhs, GetConstBits(ToAstNode(node->index())));
  XLS_ASSIGN_OR_RETURN(uint64_t index, rhs.ToUint64());

  Def(node, [this, v, index](const SourceInfo& loc) {
    return function_builder_->TupleIndex(v, index, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleXlsTuple(const XlsTuple* node) {
  std::vector<BValue> operands;
  for (Expr* o : node->members()) {
    XLS_ASSIGN_OR_RETURN(BValue v, Use(o));
    operands.push_back(v);
  }
  Def(node, [this, &operands](const SourceInfo& loc) {
    return function_builder_->Tuple(operands, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleZeroMacro(const ZeroMacro* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue iv, current_type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(iv));
  Def(node, [this, &value](const SourceInfo& loc) {
    return function_builder_->Literal(value, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleAllOnesMacro(const AllOnesMacro* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue iv, current_type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(iv));
  Def(node, [this, &value](const SourceInfo& loc) {
    return function_builder_->Literal(value, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleParam(const Param* node) {
  VLOG(5) << "FunctionConverter::HandleParam: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(xls::Type * type,
                       ResolveTypeToIr(node->type_annotation()));
  Def(node->name_def(), [&](const SourceInfo& loc) {
    return function_builder_->Param(node->identifier(), type);
  });
  XLS_RET_CHECK(function_proto_);
  PackageInterfaceProto::NamedValue* param =
      function_proto_.value()->add_parameters();
  param->set_name(node->identifier());
  *param->mutable_type() = type->ToProto();
  XLS_ASSIGN_OR_RETURN(std::optional<std::string> sv_value,
                       current_type_info_->FindSvType(node->type_annotation()));
  if (sv_value) {
    param->set_sv_type(*sv_value);
  }
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleConstRef(const ConstRef* node) {
  return DefAlias(node->name_def(), /*to=*/node);
}

absl::Status FunctionConverter::HandleNameRef(const NameRef* node) {
  AstNode* from = ToAstNode(node->name_def());

  if (!node_to_ir_.contains(from)) {
    XLS_RET_CHECK(proc_id_.has_value());
    XLS_RET_CHECK(proc_data_->id_to_members.contains(proc_id_.value()));
    for (const auto& [k, v] : proc_data_->id_to_members.at(proc_id_.value())) {
      if (k == node->identifier()) {
        if (std::holds_alternative<Value>(v)) {
          VLOG(4) << "Reference to Proc member: " << k
                  << " : Value : " << std::get<Value>(v).ToString();
          CValue cvalue;
          cvalue.ir_value = std::get<Value>(v);
          cvalue.value = function_builder_->Literal(cvalue.ir_value);
          SetNodeToIr(from, cvalue);
        } else {
          VLOG(4) << "Reference to Proc member: " << k
                  << " : Chan  : " << std::get<Channel*>(v)->ToString();
          SetNodeToIr(from, std::get<Channel*>(v));
        }
      }
    }
  }

  return DefAlias(from, /*to=*/node);
}

absl::Status FunctionConverter::HandleConstantDef(const ConstantDef* node) {
  // We've already evaluated constants to their values; we don't need to dive
  // into them for [useless] IR conversion.
  VLOG(5) << "Visiting ConstantDef expr: " << node->value()->ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue iv,
                       current_type_info_->GetConstExpr(node->value()));
  XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(iv));
  Def(node->value(), [this, &value](const SourceInfo& loc) {
    return function_builder_->Literal(value, loc);
  });
  VLOG(5) << "Aliasing NameDef for constant: " << node->name_def()->ToString();
  return DefAlias(node->value(), /*to=*/node->name_def());
}

absl::Status FunctionConverter::HandleLet(const Let* node) {
  VLOG(5) << "FunctionConverter::HandleLet: `" << node->ToString()
          << "`; rhs: `" << node->rhs()->ToString() << "`";
  XLS_RETURN_IF_ERROR(Visit(node->rhs()));

  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->rhs()));
  XLS_RET_CHECK(rhs.valid());

  // Verify that the RHS conforms to the annotation (if present).
  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(xls::Type * annotated_type,
                         ResolveTypeToIr(node->type_annotation()));
    xls::Type* value_type = rhs.GetType();
    XLS_RET_CHECK_EQ(annotated_type, value_type);
  }

  XLS_RETURN_IF_ERROR(DefAlias(node->rhs(), /*to=*/node));

  if (node->name_def_tree()->is_leaf()) {
    // Alias so that the RHS expression is now known as the name definition it
    // is bound to.
    XLS_RETURN_IF_ERROR(
        DefAlias(node->rhs(), /*to=*/ToAstNode(node->name_def_tree()->leaf())));
  } else {
    // Walk the tree of names we're trying to bind, performing tuple_index
    // operations on the RHS to get to the values we want to bind to those
    // names.
    std::vector<BValue> levels = {rhs};
    // Invoked at each level of the NameDefTree: binds the name in the
    // NameDefTree to the corresponding value (being pattern matched).
    //
    // Args:
    //  x: Current subtree of the NameDefTree.
    //  level: Level (depth) in the NameDefTree, root is 0.
    //  index: Index of node in the current tree level (e.g. leftmost is 0).
    auto walk = [&](NameDefTree* x, int64_t level,
                    int64_t index) -> absl::Status {
      VLOG(6) << absl::StreamFormat("Walking level %d index %d: `%s`", level,
                                    index, x->ToString());
      XLS_RET_CHECK(x != nullptr);
      levels.resize(level);
      levels.push_back(Def(x, [this, &levels, x, index](SourceInfo loc) {
        if (!loc.Empty()) {
          loc = ToSourceInfo(x->is_leaf() ? ToAstNode(x->leaf())->GetSpan()
                                          : x->GetSpan());
        }

        BValue tuple = levels.back();
        CHECK(tuple.valid());

        xls::TupleType* tuple_type = tuple.GetType()->AsTupleOrDie();
        CHECK_LT(index, tuple_type->size())
            << "index: " << index << " type: " << tuple_type->ToString();

        BValue tuple_index = function_builder_->TupleIndex(tuple, index, loc);
        CHECK_OK(function_builder_->GetError());

        return tuple_index;
      }));
      if (x->is_leaf()) {
        XLS_RETURN_IF_ERROR(DefAlias(x, ToAstNode(x->leaf())));
      }
      return absl::OkStatus();
    };

    XLS_RETURN_IF_ERROR(node->name_def_tree()->DoPreorder(walk));
  }

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleCast(const Cast* node) {
  XLS_RETURN_IF_ERROR(Visit(node->expr()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> output_type, ResolveType(node));
  if (auto* array_type = dynamic_cast<ArrayType*>(output_type.get())) {
    return CastToArray(node, *array_type);
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> input_type,
                       ResolveType(node->expr()));
  if (dynamic_cast<ArrayType*>(input_type.get()) != nullptr &&
      !IsArrayOfBitsConstructor(*input_type)) {
    return CastFromArray(node, *output_type);
  }
  XLS_ASSIGN_OR_RETURN(TypeDim new_bit_count_ctd,
                       output_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t new_bit_count,
      std::get<InterpValue>(new_bit_count_ctd.value()).GetBitValueViaSign());
  XLS_ASSIGN_OR_RETURN(TypeDim input_bit_count_ctd,
                       input_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t old_bit_count,
      std::get<InterpValue>(input_bit_count_ctd.value()).GetBitValueViaSign());
  if (new_bit_count < old_bit_count) {
    auto bvalue_status = DefWithStatus(
        node,
        [this, node,
         new_bit_count](const SourceInfo& loc) -> absl::StatusOr<BValue> {
          XLS_ASSIGN_OR_RETURN(BValue input, Use(node->expr()));
          return function_builder_->BitSlice(input, 0, new_bit_count);
        });
    XLS_RETURN_IF_ERROR(bvalue_status.status());
  } else {
    XLS_ASSIGN_OR_RETURN(bool signed_input, IsSigned(*input_type));
    auto bvalue_status = DefWithStatus(
        node,
        [this, node, new_bit_count,
         signed_input](const SourceInfo& loc) -> absl::StatusOr<BValue> {
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

absl::Status FunctionConverter::HandleBuiltinCheckedCast(
    const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);

  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> output_type, ResolveType(node));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> input_type,
                       ResolveType(node->args()[0]));

  XLS_ASSIGN_OR_RETURN(bool signed_input, IsSigned(*input_type));

  XLS_ASSIGN_OR_RETURN(TypeDim new_bit_count_ctd,
                       output_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t new_bit_count,
      std::get<InterpValue>(new_bit_count_ctd.value()).GetBitValueViaSign());

  XLS_ASSIGN_OR_RETURN(TypeDim input_bit_count_ctd,
                       input_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t old_bit_count,
      std::get<InterpValue>(input_bit_count_ctd.value()).GetBitValueViaSign());

  if (dynamic_cast<ArrayType*>(output_type.get()) != nullptr ||
      dynamic_cast<ArrayType*>(input_type.get()) != nullptr) {
    return IrConversionErrorStatus(
        node->span(),
        absl::StrFormat("CheckedCast to and from array "
                        "is not currently supported for IR conversion; "
                        "attempted checked cast from: %s to: %s",
                        input_type->ToString(), output_type->ToString()));
  }

  // TODO(tedhong): 2023-05-22 Add verilog assertion that cast has not
  // lost any data.
  if (new_bit_count < old_bit_count) {
    Def(node, [this, arg, new_bit_count](const SourceInfo& loc) {
      return function_builder_->BitSlice(arg, 0, new_bit_count);
    });
  } else {
    Def(node, [this, arg, signed_input, new_bit_count](const SourceInfo& loc) {
      if (signed_input) {
        return function_builder_->SignExtend(arg, new_bit_count);
      }
      return function_builder_->ZeroExtend(arg, new_bit_count);
    });
  }
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinWideningCast(
    const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);

  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> output_type, ResolveType(node));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> input_type,
                       ResolveType(node->args()[0]));

  CHECK_NE(dynamic_cast<BitsType*>(input_type.get()), nullptr);
  CHECK_NE(dynamic_cast<BitsType*>(output_type.get()), nullptr);

  XLS_ASSIGN_OR_RETURN(bool signed_input, IsSigned(*input_type));

  XLS_ASSIGN_OR_RETURN(TypeDim new_bit_count_ctd,
                       output_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t new_bit_count,
      std::get<InterpValue>(new_bit_count_ctd.value()).GetBitValueViaSign());

  // Perform actual cast. Validity is checked during type_check.
  Def(node, [this, arg, signed_input, new_bit_count](const SourceInfo& loc) {
    if (signed_input) {
      return function_builder_->SignExtend(arg, new_bit_count);
    }
    return function_builder_->ZeroExtend(arg, new_bit_count);
  });

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleMatch(const Match* node) {
  if (node->arms().empty() ||
      !node->arms().back()->patterns()[0]->IsIrrefutable()) {
    return IrConversionErrorStatus(
        node->span(),
        "Only matches with trailing irrefutable patterns (i.e. `_ => ...`) "
        "are currently supported for IR conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->matched()));
  XLS_ASSIGN_OR_RETURN(BValue matched, Use(node->matched()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> matched_type,
                       ResolveType(node->matched()));

  std::vector<BValue> arm_selectors;
  std::vector<BValue> arm_values;
  for (int64_t i = 0; i < node->arms().size() - 1; ++i) {
    MatchArm* arm = node->arms()[i];

    // Visit all the MatchArm's patterns.
    std::vector<BValue> this_arm_selectors;
    for (NameDefTree* pattern : arm->patterns()) {
      XLS_ASSIGN_OR_RETURN(BValue selector,
                           HandleMatcher(pattern, {i}, matched, *matched_type));
      XLS_RET_CHECK(selector.valid());
      this_arm_selectors.push_back(selector);
    }

    // "Or" together the patterns in this arm, if necessary, to determine if the
    // arm is selected.
    if (this_arm_selectors.size() > 1) {
      arm_selectors.push_back(function_builder_->AddNaryOp(
          Op::kOr, this_arm_selectors, ToSourceInfo(arm->span())));
    } else {
      arm_selectors.push_back(this_arm_selectors[0]);
    }

    {
      ScopedControlPredicate scp(
          this, [&](const PredicateFun& orig_control_predicate) {
            // This arm is "activated" when:
            // * none of the previous arms have been selected
            // * this arm is selected
            // * the previously-established control predicate is true
            auto prior_selectors = absl::MakeSpan(arm_selectors)
                                       .subspan(0, arm_selectors.size() - 1);
            BValue not_any_prev_selected;
            if (prior_selectors.empty()) {
              not_any_prev_selected = function_builder_->Literal(UBits(1, 1));
            } else {
              not_any_prev_selected = function_builder_->Not(
                  function_builder_->Or(prior_selectors));
            }
            BValue this_arm_selected = arm_selectors.back();
            return function_builder_->And({orig_control_predicate(),
                                           not_any_prev_selected,
                                           this_arm_selected});
          });
      XLS_RETURN_IF_ERROR(Visit(arm->expr()));
    }

    XLS_ASSIGN_OR_RETURN(BValue arm_rhs_value, Use(arm->expr()));
    arm_values.push_back(arm_rhs_value);
  }

  // For compute of the default arm the control predicate is "none of the other
  // arms matched".
  ScopedControlPredicate scp(
      this, [&](const PredicateFun& orig_control_predicate) {
        // The default arm is "activated" when:
        // * none of the previous arms have been selected
        // * the previously-established control predicate is true
        auto prior_selectors = absl::MakeSpan(arm_selectors);
        BValue not_any_prev_selected;
        if (prior_selectors.empty()) {
          not_any_prev_selected = function_builder_->Literal(UBits(1, 1));
        } else {
          not_any_prev_selected =
              function_builder_->Not(function_builder_->Or(prior_selectors));
        }
        return function_builder_->And(
            {orig_control_predicate(), not_any_prev_selected});
      });
  MatchArm* default_arm = node->arms().back();
  if (default_arm->patterns().size() != 1) {
    return IrConversionErrorStatus(
        node->span(),
        "Multiple patterns in default arm "
        "is not currently supported for IR conversion.");
  }
  XLS_RETURN_IF_ERROR(
      HandleMatcher(default_arm->patterns()[0],
                    {static_cast<int64_t>(node->arms().size()) - 1}, matched,
                    *matched_type)
          .status());
  XLS_RETURN_IF_ERROR(Visit(default_arm->expr()));

  // So now we have the following representation of the match arms:
  //   match x {
  //     42  => a
  //     64  => b
  //     128 => d
  //     _   => d
  //   }
  //
  //   selectors:     [x==42, x==64, x==128]
  //   values:        [a,         b,      c]
  //   default_value: d
  XLS_ASSIGN_OR_RETURN(BValue default_value, Use(default_arm->expr()));

  if (arm_selectors.empty()) {
    // If there are no case arms, there is only a default value, avoid emitting
    // lots of junk operations like zero-bit concatenations via MatchTrue
    // builder by just setting the node value to be the default value.
    XLS_RET_CHECK(arm_values.empty());
    SetNodeToIr(node, default_value);
  } else {
    BValue result =
        function_builder_->MatchTrue(arm_selectors, arm_values, default_value);
    XLS_RETURN_IF_ERROR(function_builder_->GetError());
    SetNodeToIr(node, result);
  }
  return absl::OkStatus();
}

absl::StatusOr<FunctionConverter::RangeData> FunctionConverter::GetRangeData(
    const Expr* iterable) {
  auto error = [&] {
    return IrConversionErrorStatus(
        iterable->span(),
        absl::StrFormat("iterable `%s` "
                        "must be bits-typed, constexpr, and its start must be "
                        "less than or equal to its limit.",
                        iterable->ToString()));
  };

  // Easy case first: using the `..` range operator.
  InterpValue start_value(InterpValue::MakeToken());
  InterpValue limit_value(InterpValue::MakeToken());
  ParametricEnv bindings(parametric_env_map_);

  const auto* range_op = dynamic_cast<const Range*>(iterable);
  if (range_op != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        start_value, ConstexprEvaluator::EvaluateToValue(
                         import_data_, current_type_info_, kNoWarningCollector,
                         bindings, range_op->start(), nullptr));
    XLS_ASSIGN_OR_RETURN(
        limit_value, ConstexprEvaluator::EvaluateToValue(
                         import_data_, current_type_info_, kNoWarningCollector,
                         bindings, range_op->end(), nullptr));
  } else {
    const auto* iterable_call = dynamic_cast<const Invocation*>(iterable);
    if (iterable_call == nullptr) {
      return error();
    }
    auto* callee_name_ref = dynamic_cast<NameRef*>(iterable_call->callee());
    if (callee_name_ref == nullptr) {
      return error();
    }
    if (!std::holds_alternative<BuiltinNameDef*>(callee_name_ref->name_def())) {
      return error();
    }
    auto* builtin_name_def =
        std::get<BuiltinNameDef*>(callee_name_ref->name_def());
    if (builtin_name_def->identifier() != "range") {
      return error();
    }

    XLS_RET_CHECK_EQ(iterable_call->args().size(), 2);
    Expr* start = iterable_call->args()[0];
    Expr* limit = iterable_call->args()[1];

    XLS_ASSIGN_OR_RETURN(
        start_value, ConstexprEvaluator::EvaluateToValue(
                         import_data_, current_type_info_, kNoWarningCollector,
                         bindings, start, nullptr));
    XLS_ASSIGN_OR_RETURN(
        limit_value, ConstexprEvaluator::EvaluateToValue(
                         import_data_, current_type_info_, kNoWarningCollector,
                         bindings, limit, nullptr));
  }

  if (!start_value.IsBits() || !limit_value.IsBits()) {
    return error();
  }

  XLS_ASSIGN_OR_RETURN(int64_t start_int, start_value.GetBitValueViaSign());
  XLS_ASSIGN_OR_RETURN(int64_t bit_width, start_value.GetBitCount());

  XLS_ASSIGN_OR_RETURN(InterpValue start_ge_limit, start_value.Ge(limit_value));
  if (start_ge_limit.IsTrue()) {
    return RangeData{start_int, 0, bit_width};
  }

  XLS_ASSIGN_OR_RETURN(InterpValue trip_count, limit_value.Sub(start_value));
  XLS_RET_CHECK(trip_count.IsBits());
  int64_t trip_count_int;
  if (trip_count.IsSigned()) {
    XLS_ASSIGN_OR_RETURN(trip_count_int, trip_count.GetBitValueViaSign());
  } else {
    XLS_ASSIGN_OR_RETURN(trip_count_int, trip_count.GetBitValueViaSign());
  }

  return RangeData{start_int, trip_count_int, bit_width};
}

absl::Status FunctionConverter::HandleFor(const For* node) {
  XLS_RETURN_IF_ERROR(Visit(node->init()));

  XLS_ASSIGN_OR_RETURN(RangeData range_data, GetRangeData(node->iterable()));

  VLOG(5) << "Converting for-loop @ " << node->span();
  FunctionConverter body_converter(package_data_, module_, import_data_,
                                   options_, proc_data_, /*is_top=*/false);
  body_converter.set_parametric_env_map(parametric_env_map_);

  // The body conversion uses the same types that we use in the caller.
  body_converter.current_type_info_ = current_type_info_;

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
  auto body_builder =
      std::make_unique<FunctionBuilder>(body_fn_name, package());
  auto* body_builder_ptr = body_builder.get();
  body_converter.SetFunctionBuilder(std::move(body_builder));

  // Grab the two tuple of `(ivar, accum)`.
  std::vector<std::variant<NameDefTree::Leaf, NameDefTree*>> flat =
      node->names()->Flatten1();
  if (flat.size() != 2) {
    return absl::UnimplementedError(
        "Expect for loop to have counter (induction variable) and carry data "
        "for IR conversion.");
  }

  // Add the induction value (the "ranged" counter).
  auto ivar = std::get<NameDefTree::Leaf>(flat[0]);
  absl::StatusOr<BValue> loop_index_or = absl::visit(
      Visitor{
          [&](NameDef* name_def) -> absl::StatusOr<BValue> {
            XLS_RET_CHECK(name_def != nullptr);
            XLS_ASSIGN_OR_RETURN(xls::Type * ivar_type,
                                 ResolveTypeToIr(name_def));
            return body_converter.AddParam(name_def->identifier(), ivar_type);
          },
          [&](WildcardPattern* ivar) -> absl::StatusOr<BValue> {
            XLS_ASSIGN_OR_RETURN(xls::Type * ivar_type, ResolveTypeToIr(ivar));
            return body_converter.AddParam("__", ivar_type);
          },
          [&](Range*) -> absl::StatusOr<BValue> {
            return absl::InternalError("Induction variable cannot be a range");
          },
          [&](Number*) -> absl::StatusOr<BValue> {
            return absl::InternalError("Induction variable cannot be a number");
          },
          [&](ColonRef*) -> absl::StatusOr<BValue> {
            return absl::InternalError(
                "Induction variable cannot be a colon-reference");
          },
          [&](NameRef*) -> absl::StatusOr<BValue> {
            return absl::InternalError(
                "Induction variable cannot be a name-reference");
          },
      },
      ivar);
  XLS_ASSIGN_OR_RETURN(auto loop_index, loop_index_or);

  // IR `counted_for` ops only support a trip count, not a set of iterables, so
  // we need to add an offset to that trip count/index to support nonzero loop
  // start indices.
  // Pulling values out of the iterable invocation is safe, as it was all
  // checked in QueryConstRangeCall.
  Value index_offset =
      Value(UBits(range_data.start_value, range_data.bit_width));
  BValue offset_literal =
      body_converter.function_builder_->Literal(index_offset);
  BValue offset_sum =
      body_converter.function_builder_->Add(loop_index, offset_literal);
  body_converter.SetNodeToIr(ToAstNode(ivar), offset_sum);

  // Add the loop carry value.
  AstNode* carry_node = ToAstNode(flat[1]);
  if (auto* carry_name_def = dynamic_cast<NameDef*>(carry_node)) {
    // When the loop carry value is just a name; e.g. the `x` in `for (i, x)` we
    // can simply bind it.
    XLS_ASSIGN_OR_RETURN(xls::Type * type, ResolveTypeToIr(carry_name_def));
    if (implicit_token_data_.has_value()) {
      // Note: this is somewhat conservative, even if the for-body does not
      // require an implicit token, for bodies are not marked independently from
      // their enclosing functions.
      BValue unwrapped = body_converter.AddTokenWrappedParam(type);
      body_converter.SetNodeToIr(carry_name_def, unwrapped);
    } else {
      BValue param =
          body_converter.AddParam(carry_name_def->identifier(), type);
      body_converter.SetNodeToIr(carry_name_def, param);
    }
  } else {
    // For tuple loop carries we have to destructure names on entry.
    // Note this could be something like a NameDef or something like a
    // WildcardPattern -- even if it's a wildcard pattern we throw away, we
    // still want to make the loop with the same pattern.
    AstNode* accum = carry_node;
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> carry_type, ResolveType(accum));
    XLS_ASSIGN_OR_RETURN(
        xls::Type * carry_ir_type,
        TypeToIr(package(), *carry_type, ParametricEnv(parametric_env_map_)));
    BValue carry;
    if (implicit_token_data_.has_value()) {
      carry = body_converter.AddTokenWrappedParam(carry_ir_type);
    } else {
      carry = body_converter.AddParam("__loop_carry", carry_ir_type);
    }
    body_converter.SetNodeToIr(accum, carry);
    // This will destructure the names for us in the body of the anonymous
    // function.
    if (auto* ndt = dynamic_cast<NameDefTree*>(accum)) {
      XLS_RETURN_IF_ERROR(body_converter
                              .HandleMatcher(/*matcher=*/ndt, /*index=*/{},
                                             /*matched_value=*/carry,
                                             /*matched_type=*/*carry_type)
                              .status());
    } else {
      XLS_RET_CHECK(dynamic_cast<WildcardPattern*>(accum) != nullptr)
          << "Expect post-typechecking loop binding to be NameDefTree or "
             "WildcardPattern";
    }
  }

  // We need to capture the lexical scope and pass it to his loop body function.
  //
  // So we suffix free variables for the function body onto the function
  // parameters.
  FreeVariables freevars =
      GetFreeVariables(node->body(), &node->span().start());
  freevars = freevars.DropBuiltinDefs();
  std::vector<const NameDef*> relevant_name_defs;
  for (const auto& any_name_def : freevars.GetNameDefs()) {
    const auto* freevar_name_def = std::get<const NameDef*>(any_name_def);
    std::optional<const Type*> type =
        current_type_info_->GetItem(freevar_name_def);
    if (!type.has_value()) {
      continue;
    }
    if (dynamic_cast<const FunctionType*>(type.value()) != nullptr) {
      continue;
    }
    AstNode* definer = freevar_name_def->definer();
    if (dynamic_cast<EnumDef*>(definer) != nullptr ||
        dynamic_cast<TypeAlias*>(definer) != nullptr) {
      continue;
    }
    VLOG(5) << "Converting freevar name: " << freevar_name_def->ToString();

    std::optional<IrValue> ir_value = GetNodeToIr(freevar_name_def);
    if (!ir_value.has_value()) {
      return absl::InternalError(
          absl::StrFormat("AST node had no associated IR value: %s @ %s",
                          node->ToString(), SpanToString(node->GetSpan())));
    }

    // If free variable is a constant, create constant node inside body.
    // This preserves const-ness of loop body uses (e.g. loop bounds for
    // a nested loop).
    if (std::holds_alternative<CValue>(*ir_value)) {
      Value constant_value = std::get<CValue>(*ir_value).ir_value;
      body_converter.DefConst(freevar_name_def, constant_value);
    } else {
      // Otherwise, pass in the variable to the loop body function as
      // a parameter.
      relevant_name_defs.push_back(freevar_name_def);
      XLS_ASSIGN_OR_RETURN(
          xls::Type * name_def_type,
          TypeToIr(package(), **type, ParametricEnv(parametric_env_map_)));
      body_converter.SetNodeToIr(
          freevar_name_def, body_converter.AddParam(
                                freevar_name_def->identifier(), name_def_type));
    }
  }

  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(body_converter.implicit_token_data_.has_value());
  }

  FunctionConverterVisitor visitor(&body_converter);
  XLS_RETURN_IF_ERROR(visitor.Visit(node->body()));

  // We also need to thread the token out of the body function: convert our
  // signature:
  //
  //   f(ivar: I, accum: T, invariant...) -> T
  //
  // To:
  //   f(ivar: I, (token, activated, accum): (token, bool, T), invariant...)
  //       -> (token, bool, T)
  //
  // Note we could make the "activated" boolean an invariant argument, but this
  // is simpler because it keeps all the params in the same spot regardless of
  // whether it's a function body or any other kind of function.
  if (implicit_token_data_.has_value()) {
    XLS_ASSIGN_OR_RETURN(BValue retval, body_converter.Use(node->body()));
    std::vector<BValue> after_all_tokens(
        body_converter.implicit_token_data_->control_tokens.begin(),
        body_converter.implicit_token_data_->control_tokens.end());
    after_all_tokens.push_back(
        body_converter.implicit_token_data_->entry_token);
    BValue implicit_token;
    if (after_all_tokens.size() > 1) {
      implicit_token =
          body_converter.function_builder_->AfterAll(after_all_tokens);
    } else {
      implicit_token = after_all_tokens[0];
    }

    body_converter.function_builder_->Tuple(
        {implicit_token, body_converter.implicit_token_data_->activated,
         retval});
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * body_function,
                       body_builder_ptr->Build());
  VLOG(5) << "Converted body function: " << body_function->name();

  std::vector<BValue> invariant_args;
  for (const NameDef* nd : relevant_name_defs) {
    XLS_ASSIGN_OR_RETURN(BValue value, Use(nd));
    invariant_args.push_back(value);
  }

  XLS_ASSIGN_OR_RETURN(BValue init, Use(node->init()));
  if (implicit_token_data_.has_value()) {
    BValue activated = range_data.trip_count == 0
                           ? function_builder_->Literal(UBits(0, 1))
                           : implicit_token_data_->activated;
    init = function_builder_->Tuple(
        {implicit_token_data_->entry_token, activated, init});
  }

  Def(node, [&](const SourceInfo& loc) {
    BValue result =
        function_builder_->CountedFor(init, range_data.trip_count, /*stride=*/1,
                                      body_function, invariant_args);
    // If a token was threaded through, we grab it and note it's an assertion
    // token.
    if (implicit_token_data_.has_value()) {
      BValue token = function_builder_->TupleIndex(result, 0);
      implicit_token_data_->control_tokens.push_back(token);
      return function_builder_->TupleIndex(result, 2);
    }
    return result;
  });
  return absl::OkStatus();
}

absl::StatusOr<BValue> FunctionConverter::HandleMatcher(
    NameDefTree* matcher, absl::Span<const int64_t> index,
    const BValue& matched_value, const Type& matched_type) {
  if (matcher->is_leaf()) {
    NameDefTree::Leaf leaf = matcher->leaf();
    VLOG(5) << absl::StreamFormat("Matcher is leaf: %s (%s)",
                                  ToAstNode(leaf)->ToString(),
                                  ToAstNode(leaf)->GetNodeTypeName());
    auto equality = [&]() -> absl::StatusOr<BValue> {
      XLS_RETURN_IF_ERROR(Visit(ToAstNode(leaf)));
      XLS_ASSIGN_OR_RETURN(BValue to_match, Use(ToAstNode(leaf)));
      return Def(matcher, [&](const SourceInfo& loc) {
        return function_builder_->Eq(to_match, matched_value);
      });
    };
    return absl::visit(
        Visitor{
            [&](WildcardPattern*) -> absl::StatusOr<BValue> {
              return Def(matcher, [&](const SourceInfo& loc) {
                return function_builder_->Literal(UBits(1, 1), loc);
              });
            },
            [&](Number* n) -> absl::StatusOr<BValue> { return equality(); },
            [&](ColonRef* n) -> absl::StatusOr<BValue> { return equality(); },
            [&](Range* n) -> absl::StatusOr<BValue> {
              XLS_RETURN_IF_ERROR(Visit(ToAstNode(n->start())));
              XLS_RETURN_IF_ERROR(Visit(ToAstNode(n->end())));
              bool signed_input =
                  down_cast<const BitsType*>(&matched_type)->is_signed();
              XLS_ASSIGN_OR_RETURN(BValue start, Use(n->start()));
              XLS_ASSIGN_OR_RETURN(BValue limit, Use(n->end()));
              SourceInfo loc = ToSourceInfo(n->span());
              auto ge = [&](const BValue& lhs, const BValue& rhs) {
                if (signed_input) {
                  return function_builder_->SGe(lhs, rhs, loc);
                }
                return function_builder_->UGe(lhs, rhs, loc);
              };
              auto lt = [&](const BValue& lhs, const BValue& rhs) {
                if (signed_input) {
                  return function_builder_->SLt(lhs, rhs, loc);
                }
                return function_builder_->ULt(lhs, rhs, loc);
              };
              return function_builder_->And(ge(matched_value, start),
                                            lt(matched_value, limit));
            },
            [&](NameRef* n) -> absl::StatusOr<BValue> {
              // Comparing for equivalence to a (referenced) name.
              auto* name_ref = std::get<NameRef*>(leaf);
              const auto* name_def =
                  std::get<const NameDef*>(name_ref->name_def());
              XLS_ASSIGN_OR_RETURN(BValue to_match, Use(name_def));
              BValue result = Def(matcher, [&](const SourceInfo& loc) {
                return function_builder_->Eq(to_match, matched_value);
              });
              XLS_RETURN_IF_ERROR(DefAlias(name_def, name_ref));
              return result;
            },
            [&](NameDef* name_def) -> absl::StatusOr<BValue> {
              BValue ok = Def(name_def, [&](const SourceInfo& loc) {
                return function_builder_->Literal(UBits(1, 1));
              });
              SetNodeToIr(matcher, matched_value);
              SetNodeToIr(ToAstNode(leaf), matched_value);
              return ok;
            },
        },
        leaf);
  }

  auto* matched_tuple_type = dynamic_cast<const TupleType*>(&matched_type);
  BValue ok = function_builder_->Literal(UBits(/*value=*/1, /*bit_count=*/1));
  for (int64_t i = 0; i < matched_tuple_type->size(); ++i) {
    const Type& element_type = matched_tuple_type->GetMemberType(i);
    NameDefTree* element = matcher->nodes()[i];
    BValue member = function_builder_->TupleIndex(matched_value, i);
    std::vector<int64_t> sub_index(index.begin(), index.end());
    sub_index.push_back(i);
    XLS_ASSIGN_OR_RETURN(
        BValue cond, HandleMatcher(element, sub_index, member, element_type));
    ok = function_builder_->And(ok, cond);
  }
  return ok;
}

absl::StatusOr<BValue> FunctionConverter::DefMapWithBuiltin(
    const Invocation* parent_node, NameRef* node, AstNode* arg,
    const ParametricEnv& parametric_env) {
  // Builtins always use the "typical" calling convention (are never "implicit
  // token").
  XLS_ASSIGN_OR_RETURN(const std::string mangled_name,
                       MangleDslxName(module_->name(), node->identifier(),
                                      CallingConvention::kTypical,
                                      /*free_keys=*/{}, &parametric_env));
  XLS_ASSIGN_OR_RETURN(BValue arg_value, Use(arg));
  VLOG(5) << "Mapping with builtin; arg: " << arg_value.GetType()->ToString();
  auto* array_type = arg_value.GetType()->AsArrayOrDie();
  std::optional<xls::Function*> f = package()->TryGetFunction(mangled_name);
  if (!f.has_value()) {
    FunctionBuilder fb(mangled_name, package());

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
    // Add an interface entry.
    PackageInterfaceProto::Function* fp =
        package_data_.conversion_info->interface.add_functions();
    *fp->mutable_base()->mutable_name() = mangled_name;
    auto* param_proto = fp->add_parameters();
    param_proto->set_name("arg");
    *param_proto->mutable_type() = array_type->element_type()->ToProto();
    *fp->mutable_result_type() = result.GetType()->ToProto();
    XLS_ASSIGN_OR_RETURN(f, fb.Build());
  }

  return Def(parent_node, [&](const SourceInfo& loc) {
    return function_builder_->Map(arg_value, *f);
  });
}

absl::StatusOr<BValue> FunctionConverter::HandleMap(const Invocation* node) {
  for (Expr* arg : node->args().subspan(0, node->args().size() - 1)) {
    XLS_RETURN_IF_ERROR(Visit(arg));
  }
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Expr* fn_node = node->args()[1];
  VLOG(5) << "Function being mapped AST: " << fn_node->ToString();
  std::optional<const ParametricEnv*> node_parametric_env =
      GetInvocationCalleeBindings(node);

  std::string map_fn_name;
  Module* lookup_module = nullptr;
  if (auto* name_ref = dynamic_cast<NameRef*>(fn_node)) {
    map_fn_name = name_ref->identifier();
    if (IsNameParametricBuiltin(map_fn_name)) {
      VLOG(5) << "Map of parametric builtin: " << map_fn_name;
      return DefMapWithBuiltin(node, name_ref, node->args()[0],
                               *node_parametric_env.value());
    }
    lookup_module = module_;
  } else if (auto* colon_ref = dynamic_cast<ColonRef*>(fn_node)) {
    map_fn_name = colon_ref->attr();
    Import* import_node = colon_ref->ResolveImportSubject().value();
    std::optional<const ImportedInfo*> info =
        current_type_info_->GetImported(import_node);
    lookup_module = (*info)->module;
  } else {
    return absl::UnimplementedError("Unhandled function mapping: " +
                                    fn_node->ToString());
  }

  XLS_ASSIGN_OR_RETURN(Function * mapped_fn,
                       lookup_module->GetMemberOrError<Function>(map_fn_name));
  std::vector<std::string> free = mapped_fn->GetFreeParametricKeys();
  absl::btree_set<std::string> free_set(free.begin(), free.end());
  CallingConvention convention = GetCallingConvention(mapped_fn);
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(lookup_module->name(), mapped_fn->identifier(), convention,
                     free_set, node_parametric_env.value()));
  VLOG(5) << "Getting function with mangled name: " << mangled_name
          << " from package: " << package()->name();
  XLS_ASSIGN_OR_RETURN(xls::Function * f, package()->GetFunction(mangled_name));
  return Def(node, [&](const SourceInfo& loc) -> BValue {
    return function_builder_->Map(arg, f, loc);
  });
}

absl::Status FunctionConverter::HandleIndex(const Index* node) {
  XLS_RETURN_IF_ERROR(Visit(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));

  std::optional<const Type*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  if (dynamic_cast<const TupleType*>(lhs_type.value()) != nullptr) {
    // Tuple indexing requires a compile-time-constant RHS.
    XLS_RETURN_IF_ERROR(Visit(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(Bits rhs, GetConstBits(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(uint64_t index, rhs.ToUint64());
    Def(node, [&](const SourceInfo& loc) {
      return function_builder_->TupleIndex(lhs, index, loc);
    });
  } else if (dynamic_cast<const BitsType*>(lhs_type.value()) != nullptr) {
    IndexRhs rhs = node->rhs();
    if (std::holds_alternative<WidthSlice*>(rhs)) {
      auto* width_slice = std::get<WidthSlice*>(rhs);
      XLS_RETURN_IF_ERROR(Visit(width_slice->start()));
      XLS_ASSIGN_OR_RETURN(BValue start, Use(width_slice->start()));
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> output_type,
                           ResolveType(node));
      XLS_ASSIGN_OR_RETURN(TypeDim output_type_dim,
                           output_type->GetTotalBitCount());
      XLS_ASSIGN_OR_RETURN(int64_t width, output_type_dim.GetAsInt64());
      Def(node, [&](const SourceInfo& loc) {
        return function_builder_->DynamicBitSlice(lhs, start, width, loc);
      });
    } else {
      auto* slice = std::get<Slice*>(rhs);
      std::optional<StartAndWidth> saw =
          current_type_info_->GetSliceStartAndWidth(slice, GetParametricEnv());
      XLS_RET_CHECK(saw.has_value());
      Def(node, [&](const SourceInfo& loc) {
        return function_builder_->BitSlice(lhs, saw->start, saw->width, loc);
      });
    }
  } else {
    XLS_RETURN_IF_ERROR(Visit(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(BValue index, Use(ToAstNode(node->rhs())));
    Def(node, [&](const SourceInfo& loc) {
      return function_builder_->ArrayIndex(lhs, {index}, loc);
    });
  }
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleArray(const Array* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, ResolveType(node));
  const ArrayType* array_type = dynamic_cast<ArrayType*>(type.get());
  XLS_RET_CHECK(array_type != nullptr);
  std::vector<BValue> members;
  for (Expr* member : node->members()) {
    XLS_RETURN_IF_ERROR(Visit(member));
    XLS_ASSIGN_OR_RETURN(BValue member_value, Use(member));
    members.push_back(member_value);
  }

  if (node->has_ellipsis()) {
    TypeDim array_size_ctd = array_type->size();
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_size_ctd.GetAsInt64());
    while (members.size() < array_size) {
      members.push_back(members.back());
    }
  }
  // TODO(google/xls#917): Remove this check when empty arrays are supported.
  if (members.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Array %s was empty.", node->ToString()));
  }
  Def(node, [&](const SourceInfo& loc) {
    xls::Type* type = members[0].GetType();
    return function_builder_->Array(members, type, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleUdfInvocation(const Invocation* node,
                                                    xls::Function* f,
                                                    std::vector<BValue> args) {
  VLOG(5) << "HandleUdfInvocation: " << f->name() << " via "
          << node->ToString();
  XLS_RET_CHECK(package_data_.ir_to_dslx.contains(f)) << f->name();
  dslx::Function* dslx_callee =
      dynamic_cast<dslx::Function*>(package_data_.ir_to_dslx.at(f));

  const bool callee_requires_implicit_token =
      GetRequiresImplicitToken(dslx_callee);
  VLOG(6) << "HandleUdfInvocation: callee: " << dslx_callee->ToString()
          << " callee_requires_implicit_token: "
          << callee_requires_implicit_token;
  if (callee_requires_implicit_token) {
    XLS_RET_CHECK(implicit_token_data_.has_value()) << absl::StreamFormat(
        "If callee (%s @ %s) requires an implicit token, caller must require "
        "a token as well (property is transitive across call graph).",
        f->name(), node->span().ToString());
    // Prepend the token and the control predicate boolean on the args.
    std::vector<BValue> new_args = {implicit_token_data_->entry_token};
    new_args.push_back(implicit_token_data_->create_control_predicate());
    for (const BValue& arg : args) {
      new_args.push_back(arg);
    }
    args = new_args;
  }

  Def(node, [&](const SourceInfo& loc) {
    BValue result = function_builder_->Invoke(args, f, loc);
    if (!callee_requires_implicit_token) {
      return result;
    }
    // If the callee needs an implicit token it will also produce an implicit
    // result token. We have to grab that and make it one of our
    // "control_tokens_". It is guaranteed to be the first member of the
    // tuple result.
    BValue result_token = function_builder_->TupleIndex(result, 0);
    implicit_token_data_->control_tokens.push_back(result_token);
    return function_builder_->TupleIndex(result, 1);
  });
  return absl::OkStatus();
}

absl::StatusOr<FunctionConverter::AssertionLabelData>
FunctionConverter::GetAssertionLabel(std::string_view caller_name,
                                     const Expr* label_expr, const Span& span) {
  ParametricEnv bindings(parametric_env_map_);
  XLS_RETURN_IF_ERROR(
      ConstexprEvaluator::Evaluate(import_data_, current_type_info_,
                                   kNoWarningCollector, bindings, label_expr));

  std::optional<InterpValue> start_value =
      current_type_info_->GetConstExprOption(label_expr);
  XLS_RET_CHECK(start_value.has_value());
  XLS_ASSIGN_OR_RETURN(std::optional<std::string> label,
                       InterpValueAsString(start_value.value()));
  XLS_RET_CHECK(label.has_value());

  // TODO(cdleary): 2024-03-12 We should put the label into the assertion
  // failure error message.
  std::string message = absl::StrFormat("Assertion failure via %s @ %s",
                                        caller_name, span.ToString());
  return AssertionLabelData{.label = label.value(), .message = message};
}

absl::Status FunctionConverter::HandleFailBuiltin(const Invocation* node,
                                                  Expr* label_expr,
                                                  BValue arg) {
  if (options_.emit_fail_as_assert) {
    // For a fail node we both create a predicate that corresponds to the
    // "control" leading to this DSL program point.
    XLS_RET_CHECK(implicit_token_data_.has_value())
        << "Invoking fail!(), but no implicit token is present for caller @ "
        << node->span();
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    BValue control_predicate = implicit_token_data_->create_control_predicate();

    XLS_ASSIGN_OR_RETURN(AssertionLabelData label_data,
                         GetAssertionLabel("fail!", label_expr, node->span()));
    BValue assert_result_token =
        function_builder_->Assert(implicit_token_data_->entry_token,
                                  function_builder_->Not(control_predicate),
                                  label_data.message, label_data.label);
    XLS_RETURN_IF_ERROR(function_builder_->GetError());
    implicit_token_data_->control_tokens.push_back(assert_result_token);
    tokens_.push_back(assert_result_token);
  }
  // The result of the failure call is the argument given; e.g. if we were to
  // remove assertions this is the value that would flow in the case that the
  // assertion was hit.
  Def(node,
      [&](const SourceInfo& loc) { return function_builder_->Identity(arg); });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleAssertBuiltin(const Invocation* node,
                                                    BValue assert_predicate,
                                                    Expr* label_expr) {
  if (options_.emit_fail_as_assert) {
    // For a fail node we both create a predicate that corresponds to the
    // "control" leading to this DSL program point.
    XLS_RET_CHECK(implicit_token_data_.has_value())
        << "Invoking assert!(), but no implicit token is present for caller @ "
        << node->span();
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    BValue control_predicate = implicit_token_data_->create_control_predicate();

    // Variables:
    // * we got to the control point (CP)
    // * the assert predicate (AP)
    //
    // OK = !CP | AP
    BValue ok = function_builder_->Or(function_builder_->Not(control_predicate),
                                      assert_predicate);

    XLS_ASSIGN_OR_RETURN(
        AssertionLabelData label_data,
        GetAssertionLabel("assert!", label_expr, node->span()));
    BValue assert_result_token =
        function_builder_->Assert(implicit_token_data_->entry_token, ok,
                                  label_data.message, label_data.label);
    implicit_token_data_->control_tokens.push_back(assert_result_token);
    tokens_.push_back(assert_result_token);
  }

  // The result of the assert call is unit, the empty tuple.
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Tuple(std::vector<BValue>());
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleFormatMacro(const FormatMacro* node) {
  XLS_RET_CHECK(implicit_token_data_.has_value())
      << "Invoking trace_fmt!(), but no implicit token is present for caller @ "
      << node->span();
  XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
  BValue control_predicate = implicit_token_data_->create_control_predicate();

  // We have to rewrite the format string if a struct is present.
  //
  // Traverse through the original format steps, and if we encounter a struct,
  // blow it up getting its component elements.
  //
  // Implementation note: in order to do this, we walk through the original
  // format steps and keep track of what corresponding argument number an
  // interpolation would be referencing. When a struct is encountered, we "deep
  // traverse it's elements" and push corresponding string format data for the
  // fields. Ultimately this produces fmt_steps and corresponding ir_args -- we
  // may have done a bunch of additional GetTupleElement() operations to feed
  // ir_args leaf struct fields component-wise.

  std::vector<BValue> args;
  for (const Expr* arg : node->args()) {
    XLS_RETURN_IF_ERROR(Visit(arg));
    XLS_ASSIGN_OR_RETURN(BValue argval, Use(arg));
    args.push_back(argval);
  }

  XLS_ASSIGN_OR_RETURN(
      BValue trace_result_token,
      ConvertFormatMacro(*node, implicit_token_data_->entry_token,
                         control_predicate, args, *current_type_info_,
                         *function_builder_));

  implicit_token_data_->control_tokens.push_back(trace_result_token);

  // The result of the trace is the output token, so pass it along.
  Def(node, [&](const SourceInfo& loc) { return trace_result_token; });
  tokens_.push_back(trace_result_token);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleCoverBuiltin(const Invocation* node,
                                                   BValue condition) {
  // TODO(https://github.com/google/xls/issues/232): 2021-05-21: Control cover!
  // emission with the same flag as fail!, since they share a good amount of
  // infra and conceptually are related in how they lower to Verilog.
  if (options_.emit_fail_as_assert) {
    // For a cover node we both create a predicate that corresponds to the
    // "control" leading to this DSL program point.
    XLS_RET_CHECK(implicit_token_data_.has_value())
        << "Invoking cover!(), but no implicit token is present for caller @ "
        << node->span();
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    BValue control_predicate = implicit_token_data_->create_control_predicate();

    // Variables:
    // * we got to the control point (CP)
    // * the cover condition (CC)
    //
    // COVERED = CP & CC
    BValue covered = function_builder_->And(control_predicate, condition);
    XLS_RET_CHECK_EQ(node->args().size(), 2);
    String* label = dynamic_cast<String*>(node->args()[0]);
    XLS_RET_CHECK(label != nullptr)
        << "cover!() argument 0 must be a literal string "
        << "(should have been typechecked?).";
    function_builder_->Cover(covered, label->text());
  }

  // The result of the cover call is unit, the empty tuple.
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Tuple(std::vector<BValue>());
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleInvocation(const Invocation* node) {
  VLOG(5) << "FunctionConverter::HandleInvocation: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(std::string called_name, GetCalleeIdentifier(node));
  auto accept_args = [&]() -> absl::StatusOr<std::vector<BValue>> {
    std::vector<BValue> values;
    for (Expr* arg : node->args()) {
      XLS_RETURN_IF_ERROR(Visit(arg));
      XLS_ASSIGN_OR_RETURN(BValue value, Use(arg));
      values.push_back(value);
    }
    return values;
  };

  if (std::optional<xls::Function*> f = package()->TryGetFunction(called_name);
      f.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    return HandleUdfInvocation(node, *f, std::move(args));
  }

  // A few builtins are handled specially.

  if (called_name == "fail!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 2)
        << called_name << " builtin requires two arguments";
    return HandleFailBuiltin(node, /*label_expr=*/node->args()[0],
                             /*arg=*/args[1]);
  }
  if (called_name == "assert!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 2)
        << called_name << " builtin requires two arguments";
    return HandleAssertBuiltin(node, /*predicate=*/args[0],
                               /*label_expr=*/node->args()[1]);
  }
  if (called_name == "cover!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 2)
        << called_name << " builtin requires two arguments";
    return HandleCoverBuiltin(node, args[1]);
  }
  if (called_name == "trace!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 1)
        << called_name << " builtin only accepts a single argument";
    Def(node, [&](const SourceInfo& loc) {
      return function_builder_->Identity(args[0], loc);
    });
    return absl::OkStatus();
  }
  if (called_name == "map") {
    return HandleMap(node).status();
  }
  if (called_name == "send") {
    return HandleBuiltinSend(node);
  }
  if (called_name == "send_if") {
    return HandleBuiltinSendIf(node);
  }
  if (called_name == "recv") {
    return HandleBuiltinRecv(node);
  }
  if (called_name == "recv_if") {
    return HandleBuiltinRecvIf(node);
  }
  if (called_name == "recv_non_blocking") {
    return HandleBuiltinRecvNonBlocking(node);
  }
  if (called_name == "recv_if_non_blocking") {
    return HandleBuiltinRecvIfNonBlocking(node);
  }
  if (called_name == "join") {
    return HandleBuiltinJoin(node);
  }
  if (called_name == "token") {
    return HandleBuiltinToken(node);
  }

  // The rest of the builtins have "handle" methods we can resolve.
  absl::flat_hash_map<std::string,
                      decltype(&FunctionConverter::HandleBuiltinClz)>
      map = {
          {"array_rev", &FunctionConverter::HandleBuiltinArrayRev},
          {"array_size", &FunctionConverter::HandleBuiltinArraySize},
          {"clz", &FunctionConverter::HandleBuiltinClz},
          {"ctz", &FunctionConverter::HandleBuiltinCtz},
          {"gate!", &FunctionConverter::HandleBuiltinGate},
          {"signex", &FunctionConverter::HandleBuiltinSignex},
          {"decode", &FunctionConverter::HandleBuiltinDecode},
          {"encode", &FunctionConverter::HandleBuiltinEncode},
          {"one_hot", &FunctionConverter::HandleBuiltinOneHot},
          {"one_hot_sel", &FunctionConverter::HandleBuiltinOneHotSel},
          {"priority_sel", &FunctionConverter::HandleBuiltinPrioritySel},
          {"slice", &FunctionConverter::HandleBuiltinArraySlice},
          {"bit_slice_update", &FunctionConverter::HandleBuiltinBitSliceUpdate},
          {"rev", &FunctionConverter::HandleBuiltinRev},
          {"zip", &FunctionConverter::HandleBuiltinZip},
          {"and_reduce", &FunctionConverter::HandleBuiltinAndReduce},
          {"or_reduce", &FunctionConverter::HandleBuiltinOrReduce},
          {"xor_reduce", &FunctionConverter::HandleBuiltinXorReduce},
          {"widening_cast", &FunctionConverter::HandleBuiltinWideningCast},
          {"checked_cast", &FunctionConverter::HandleBuiltinCheckedCast},
          {"update", &FunctionConverter::HandleBuiltinUpdate},
          {"umulp", &FunctionConverter::HandleBuiltinUMulp},
          {"smulp", &FunctionConverter::HandleBuiltinSMulp},
      };
  auto it = map.find(called_name);
  if (it == map.end()) {
    return IrConversionErrorStatus(
        node->span(),
        absl::StrFormat("Could not find name for "
                        "invocation: `%s`; available: [%s]",
                        called_name,
                        absl::StrJoin(module_->GetFunctionNames(), ", ")));
  }
  XLS_RETURN_IF_ERROR(accept_args().status());
  auto f = it->second;
  return (this->*f)(node);
}

absl::Status FunctionConverter::HandleBuiltinSend(const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Send nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* payload = node->args()[2];

  XLS_RETURN_IF_ERROR(Visit(token));
  XLS_RETURN_IF_ERROR(Visit(channel));
  XLS_RETURN_IF_ERROR(Visit(payload));
  IrValue ir_value = node_to_ir_[channel];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }
  XLS_ASSIGN_OR_RETURN(BValue token_value, Use(token));
  XLS_ASSIGN_OR_RETURN(BValue data_value, Use(payload));

  BValue result;
  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    result = builder_ptr->SendIf(
        std::get<Channel*>(ir_value), token_value,
        implicit_token_data_->create_control_predicate(), data_value);
  } else {
    result = builder_ptr->Send(std::get<Channel*>(ir_value), token_value,
                               data_value);
  }
  node_to_ir_[node] = result;
  tokens_.push_back(result);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinSendIf(const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Send nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  Expr* token = node->args()[0];
  Expr* channel = node->args()[1];
  Expr* predicate = node->args()[2];
  Expr* payload = node->args()[3];

  XLS_RETURN_IF_ERROR(Visit(token));
  XLS_RETURN_IF_ERROR(Visit(channel));
  IrValue ir_value = node_to_ir_[channel];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_ASSIGN_OR_RETURN(BValue token_value, Use(token));
  XLS_RETURN_IF_ERROR(Visit(predicate));
  XLS_ASSIGN_OR_RETURN(BValue predicate_value, Use(predicate));

  XLS_RETURN_IF_ERROR(Visit(payload));
  XLS_ASSIGN_OR_RETURN(BValue data_value, Use(payload));
  BValue result;
  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    result = builder_ptr->SendIf(
        std::get<Channel*>(ir_value), token_value,
        builder_ptr->And({implicit_token_data_->create_control_predicate(),
                          predicate_value}),
        data_value);
  } else {
    result = builder_ptr->SendIf(std::get<Channel*>(ir_value), token_value,
                                 predicate_value, data_value);
  }
  node_to_ir_[node] = result;
  tokens_.push_back(result);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleRange(const Range* node) {
  // Range must be constexpr, since it implicitly defines a structural type
  // (array of N elements).
  auto maybe_type = current_type_info_->GetItem(node);
  XLS_RET_CHECK(maybe_type.has_value());
  auto* array_type = dynamic_cast<ArrayType*>(maybe_type.value());
  if (array_type == nullptr) {
    return absl::InvalidArgumentError(
        "Range expressions must resolve to array-of-bits type.");
  }
  auto* element_type =
      dynamic_cast<const BitsType*>(&array_type->element_type());
  if (element_type == nullptr) {
    return absl::InvalidArgumentError(
        "Range expressions must resolve to array-of-bits type.");
  }

  XLS_ASSIGN_OR_RETURN(RangeData range_data, GetRangeData(node));
  std::vector<Value> elements;
  for (int i = 0; i < range_data.trip_count; i++) {
    Value value =
        element_type->is_signed()
            ? Value(SBits(i + range_data.start_value, range_data.bit_width))
            : Value(UBits(i + range_data.start_value, range_data.bit_width));
    elements.push_back(value);
  }

  XLS_ASSIGN_OR_RETURN(Value array, Value::Array(elements));
  DefConst(node, array);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinRecv(const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->args()[0]));
  XLS_RETURN_IF_ERROR(Visit(node->args()[1]));
  IrValue ir_value = node_to_ir_[node->args()[1]];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->args()[0]));
  BValue value;
  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    value = builder_ptr->ReceiveIf(
        std::get<Channel*>(ir_value), token,
        implicit_token_data_->create_control_predicate());
  } else {
    value = builder_ptr->Receive(std::get<Channel*>(ir_value), token);
  }
  BValue token_value = builder_ptr->TupleIndex(value, 0);
  tokens_.push_back(token_value);
  node_to_ir_[node] = value;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinRecvNonBlocking(
    const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->args()[0]));
  XLS_RETURN_IF_ERROR(Visit(node->args()[1]));
  IrValue ir_value = node_to_ir_[node->args()[1]];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }
  XLS_RETURN_IF_ERROR(Visit(node->args()[2]));

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue default_value, Use(node->args()[2]));

  BValue recv;
  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    recv = builder_ptr->ReceiveIfNonBlocking(
        std::get<Channel*>(ir_value), token,
        implicit_token_data_->create_control_predicate());
  } else {
    recv = builder_ptr->ReceiveNonBlocking(std::get<Channel*>(ir_value), token);
  }
  BValue token_value = builder_ptr->TupleIndex(recv, 0);
  BValue received_value = builder_ptr->TupleIndex(recv, 1);
  BValue receive_activated = builder_ptr->TupleIndex(recv, 2);

  // IR non-blocking receive has a default value of zero. Mux in the
  // default_value specified in DSLX.
  BValue value =
      builder_ptr->Select(receive_activated, {default_value, received_value});
  BValue repackaged_result =
      builder_ptr->Tuple({token_value, value, receive_activated});

  tokens_.push_back(token_value);
  node_to_ir_[node] = repackaged_result;

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinRecvIf(const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->args()[0]));
  XLS_RETURN_IF_ERROR(Visit(node->args()[1]));
  IrValue ir_value = node_to_ir_[node->args()[1]];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->args()[2]));
  XLS_RETURN_IF_ERROR(Visit(node->args()[3]));

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue predicate, Use(node->args()[2]));
  XLS_ASSIGN_OR_RETURN(BValue default_value, Use(node->args()[3]));

  BValue recv;
  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    recv = builder_ptr->ReceiveIf(
        std::get<Channel*>(ir_value), token,
        builder_ptr->And(
            {implicit_token_data_->create_control_predicate(), predicate}));
  } else {
    recv =
        builder_ptr->ReceiveIf(std::get<Channel*>(ir_value), token, predicate);
  }
  BValue token_value = builder_ptr->TupleIndex(recv, 0);
  BValue received_value = builder_ptr->TupleIndex(recv, 1);

  // IR receive-if has a default value of zero. Mux in the
  // default_value specified in DSLX.
  BValue value =
      builder_ptr->Select(predicate, {default_value, received_value});
  BValue repackaged_result = builder_ptr->Tuple({token_value, value});

  tokens_.push_back(token_value);
  node_to_ir_[node] = repackaged_result;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinRecvIfNonBlocking(
    const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->args()[0]));
  XLS_RETURN_IF_ERROR(Visit(node->args()[1]));
  IrValue ir_value = node_to_ir_[node->args()[1]];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->args()[2]));
  XLS_RETURN_IF_ERROR(Visit(node->args()[3]));

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue predicate, Use(node->args()[2]));
  XLS_ASSIGN_OR_RETURN(BValue default_value, Use(node->args()[3]));

  BValue recv;
  if (implicit_token_data_.has_value()) {
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    recv = builder_ptr->ReceiveIfNonBlocking(
        std::get<Channel*>(ir_value), token,
        builder_ptr->And(
            {implicit_token_data_->create_control_predicate(), predicate}));
  } else {
    recv = builder_ptr->ReceiveIfNonBlocking(std::get<Channel*>(ir_value),
                                             token, predicate);
  }
  BValue token_value = builder_ptr->TupleIndex(recv, 0);
  BValue received_value = builder_ptr->TupleIndex(recv, 1);
  BValue receive_activated = builder_ptr->TupleIndex(recv, 2);

  // IR non-blocking receive-if has a default value of zero. Mux in the
  // default_value specified in DSLX.
  BValue value =
      builder_ptr->Select(receive_activated, {default_value, received_value});
  BValue repackaged_result =
      builder_ptr->Tuple({token_value, value, receive_activated});

  tokens_.push_back(token_value);
  node_to_ir_[node] = repackaged_result;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinJoin(const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Join nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  std::vector<BValue> ir_tokens;
  ir_tokens.reserve(node->args().size());
  for (Expr* token : node->args()) {
    XLS_RETURN_IF_ERROR(Visit(token));
    XLS_ASSIGN_OR_RETURN(BValue ir_token, Use(token));
    ir_tokens.push_back(ir_token);
  }
  BValue value = builder_ptr->AfterAll(ir_tokens);
  node_to_ir_[node] = value;
  tokens_.push_back(value);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinToken(const Invocation* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Token nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  BValue value = function_builder_->Literal(Value::Token());
  node_to_ir_[node] = value;
  tokens_.push_back(value);
  return absl::OkStatus();
}

absl::Status FunctionConverter::AddImplicitTokenParams() {
  XLS_RET_CHECK(!implicit_token_data_.has_value());
  implicit_token_data_.emplace();
  implicit_token_data_->entry_token =
      function_builder_->Param("__token", package()->GetTokenType());
  implicit_token_data_->activated =
      function_builder_->Param("__activated", package()->GetBitsType(1));
  implicit_token_data_->create_control_predicate = [&]() -> BValue {
    // At the start we're unconditionally executing, so the control predicate
    // is whether this function has been activated at all.
    return implicit_token_data_->activated;
  };
  XLS_RET_CHECK(function_proto_);

  auto* tok_param = function_proto_.value()->add_parameters();
  tok_param->set_name("__token");
  *tok_param->mutable_type() = package()->GetTokenType()->ToProto();
  auto* act_param = function_proto_.value()->add_parameters();
  act_param->set_name("__activated");
  *act_param->mutable_type() = package()->GetBitsType(1)->ToProto();

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleFunction(
    Function* node, TypeInfo* type_info, const ParametricEnv* parametric_env) {
  XLS_RET_CHECK(type_info != nullptr);
  XLS_RET_CHECK(node != nullptr);
  Function& f = *node;

  VLOG(5) << "HandleFunction: " << f.ToString();

  if (parametric_env != nullptr) {
    SetParametricEnv(parametric_env);
  }

  ScopedTypeInfoSwap stis(this, type_info);

  // We use a function builder for the duration of converting this AST Function.
  const bool requires_implicit_token = GetRequiresImplicitToken(node);
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(module_->name(), f.identifier(),
                     requires_implicit_token ? CallingConvention::kImplicitToken
                                             : CallingConvention::kTypical,
                     f.GetFreeParametricKeySet(), parametric_env));
  auto builder =
      std::make_unique<FunctionBuilder>(mangled_name, package(), true);

  auto* builder_ptr = builder.get();
  SetFunctionBuilder(std::move(builder));
  // Function is a top entity.
  if (is_top_ && !requires_implicit_token) {
    XLS_RETURN_IF_ERROR(builder_ptr->SetAsTop());
  }

  VLOG(6) << "Function " << f.identifier() << " requires_implicit_token? "
          << (requires_implicit_token ? "true" : "false");
  if (requires_implicit_token) {
    XLS_RETURN_IF_ERROR(AddImplicitTokenParams());
    XLS_RET_CHECK(implicit_token_data_.has_value());
  }

  for (Param* param : f.params()) {
    XLS_RETURN_IF_ERROR(Visit(param));
  }

  // Replace const values known here with their values in FFI template to
  // pass down templates with less variables.
  FfiPartialValueSubstituteHelper const_prefill(f.extern_verilog_module());

  for (ParametricBinding* parametric_binding : f.parametric_bindings()) {
    VLOG(5) << "Resolving parametric binding: "
            << parametric_binding->ToString();

    std::optional<InterpValue> parametric_value =
        GetParametricBinding(parametric_binding->identifier());
    XLS_RET_CHECK(parametric_value.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_type,
                         ResolveType(parametric_binding->name_def()));
    XLS_RET_CHECK(!parametric_type->IsMeta());
    XLS_ASSIGN_OR_RETURN(TypeDim parametric_width_ctd,
                         parametric_type->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(Value param_value,
                         InterpValueToValue(*parametric_value));
    const CValue evaluated = DefConst(parametric_binding, param_value);
    const_prefill.SetNamedValue(parametric_binding->name_def()->identifier(),
                                evaluated.ir_value);
    XLS_RETURN_IF_ERROR(
        DefAlias(parametric_binding, /*to=*/parametric_binding->name_def()));
  }

  // If there is foreign function data, all constant values are replaced now.
  builder_ptr->SetForeignFunctionData(const_prefill.GetUpdatedFfiData());

  VLOG(3) << "Function has " << constant_deps_.size() << " constant deps";
  for (ConstantDef* dep : constant_deps_) {
    VLOG(5) << "Visiting constant dep: " << dep->ToString();
    XLS_RETURN_IF_ERROR(Visit(dep));
  }

  VLOG(5) << "body: " << f.body()->ToString();
  XLS_RETURN_IF_ERROR(Visit(f.body()));

  XLS_ASSIGN_OR_RETURN(BValue return_value, Use(f.body()));

  if (requires_implicit_token) {
    // Now join all the assertion tokens together to make the output token.
    // This set may be empty if "emit_fail_as_assert" is false.
    BValue join_token =
        function_builder_->AfterAll(implicit_token_data_->control_tokens);
    std::vector<BValue> elements = {join_token, return_value};
    return_value = function_builder_->Tuple(elements);
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * ir_fn,
                       builder_ptr->BuildWithReturnValue(return_value));
  XLS_RET_CHECK(function_proto_);
  *function_proto_.value()->mutable_result_type() =
      return_value.GetType()->ToProto();
  XLS_ASSIGN_OR_RETURN(std::optional<std::string> sv_type,
                       current_type_info_->FindSvType(f.return_type()));
  if (sv_type) {
    *function_proto_.value()->mutable_sv_result_type() = *sv_type;
  }
  VLOG(5) << "Built function: " << ir_fn->name();
  XLS_RETURN_IF_ERROR(VerifyFunction(ir_fn));

  // If it's a public fallible function, or it's the entry function for the
  // package, we make a wrapper so that the external world (e.g. JIT, verilog
  // module) doesn't need to take implicit token arguments.
  //
  // Implementation note regarding parametric functions: *if* we wrapped those
  // to be exposed, we'd be wrapping up all the (implicitly instantiated based
  // on usage) concrete IR conversions (with the instantiation args mangled into
  // the name). Those don't seem like very public symbols with respect to the
  // outside world, since they're driven and named by DSL instantiation, so we
  // forgo exposing them here.
  if (requires_implicit_token && (node->is_public() || is_top_) &&
      !node->IsParametric()) {
    XLS_ASSIGN_OR_RETURN(
        xls::Function * wrapper,
        EmitImplicitTokenEntryWrapper(
            ir_fn, node, is_top_, &package_data_.conversion_info->interface,
            **function_proto_));
    package_data_.wrappers.insert(wrapper);
  }
  function_proto_.reset();

  package_data_.ir_to_dslx[ir_fn] = node;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleProcNextFunction(
    Function* f, const Invocation* invocation, TypeInfo* type_info,
    ImportData* import_data, const ParametricEnv* parametric_env,
    const ProcId& proc_id, ProcConversionData* proc_data) {
  XLS_RET_CHECK(type_info != nullptr);
  VLOG(5) << "HandleProcNextFunction: " << f->ToString();

  if (parametric_env != nullptr) {
    SetParametricEnv(parametric_env);
  }

  ScopedTypeInfoSwap stis(this, type_info);

  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(module_->name(), proc_id.ToString(),
                     CallingConvention::kProcNext, f->GetFreeParametricKeySet(),
                     parametric_env));
  std::string token_name = "__token";
  std::string state_name = "__state";

  Value initial_element = Value::Tuple({});
  if (proc_data_->id_to_initial_value.contains(proc_id)) {
    initial_element = proc_data_->id_to_initial_value.at(proc_id);
  }

  auto builder = std::make_unique<ProcBuilder>(mangled_name, package());
  auto implicit_token =
      builder->Literal(Value::Token(), SourceInfo(), token_name);
  BValue state = builder->StateElement(state_name, initial_element);
  tokens_.push_back(implicit_token);
  auto* builder_ptr = builder.get();
  SetFunctionBuilder(std::move(builder));
  // Proc is a top entity.
  if (is_top_) {
    XLS_RETURN_IF_ERROR(builder_ptr->SetAsTop());
  }

  // Set the one state element.
  XLS_RET_CHECK(proc_proto_);
  PackageInterfaceProto::NamedValue* state_proto =
      proc_proto_.value()->add_state();
  *state_proto->mutable_name() = state_name;
  *state_proto->mutable_type() = state.GetType()->ToProto();
  // State elements aren't emitted in an observable way so no need to track sv
  // types.

  // We make an implicit token in case any downstream functions need it; if it's
  // unused, it'll be optimized out later.
  implicit_token_data_ = ImplicitTokenData{
      .entry_token = implicit_token,
      .activated = builder_ptr->Literal(Value::Bool(true)),
      .create_control_predicate =
          [this]() { return implicit_token_data_->activated; },
  };

  // Bind the recurrent state element.
  XLS_RET_CHECK_EQ(f->params().size(), 1);
  SetNodeToIr(f->params()[0]->name_def(), state);

  proc_id_ = proc_id;

  for (ParametricBinding* parametric_binding : f->parametric_bindings()) {
    VLOG(5) << "Resolving parametric binding: "
            << parametric_binding->ToString();

    std::optional<InterpValue> parametric_value =
        GetParametricBinding(parametric_binding->identifier());
    XLS_RET_CHECK(parametric_value.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> parametric_type,
                         ResolveType(parametric_binding->name_def()));
    XLS_RET_CHECK(!parametric_type->IsMeta());

    XLS_ASSIGN_OR_RETURN(TypeDim parametric_width_ctd,
                         parametric_type->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, parametric_width_ctd.GetAsInt64());
    Value param_value;
    if (parametric_value->IsSigned()) {
      XLS_ASSIGN_OR_RETURN(int64_t bit_value,
                           parametric_value->GetBitValueViaSign());
      param_value = Value(SBits(bit_value, bit_count));
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t bit_value,
                           parametric_value->GetBitValueViaSign());
      param_value = Value(UBits(bit_value, bit_count));
    }
    DefConst(parametric_binding, param_value);
    XLS_RETURN_IF_ERROR(
        DefAlias(parametric_binding, /*to=*/parametric_binding->name_def()));
  }

  VLOG(3) << "Proc has " << constant_deps_.size() << " constant deps";
  for (ConstantDef* dep : constant_deps_) {
    VLOG(5) << "Visiting constant dep: " << dep->ToString();
    XLS_RETURN_IF_ERROR(Visit(dep));
  }

  XLS_RETURN_IF_ERROR(Visit(f->body()));

  BValue result = std::get<BValue>(node_to_ir_[f->body()]);

  XLS_ASSIGN_OR_RETURN(xls::Proc * p, builder_ptr->Build({result}));
  package_data_.ir_to_dslx[p] = f;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleColonRef(const ColonRef* node) {
  // Implementation note: ColonRef "invocations" are handled in Invocation (by
  // resolving the mangled callee name, which should have been IR converted in
  // dependency order).
  if (std::optional<Import*> import = node->ResolveImportSubject()) {
    VLOG(6) << "ColonRef @ " << node->span()
            << " was import subject; import: " << import.value()->ToString();
    std::optional<const ImportedInfo*> imported =
        current_type_info_->GetImported(*import);
    XLS_RET_CHECK(imported.has_value());
    Module* imported_mod = (*imported)->module;
    ScopedTypeInfoSwap stis(this, (*imported)->type_info);
    XLS_ASSIGN_OR_RETURN(ConstantDef * constant_def,
                         imported_mod->GetConstantDef(node->attr()));
    // A constant may be defined in terms of other constants
    // (pub const MY_CONST = std::foo(ANOTHER_CONST);), so we need to collect
    // constants transitively so we can visit all dependees.
    XLS_ASSIGN_OR_RETURN(auto constant_deps,
                         GetConstantDepFreevars(constant_def));
    for (const auto& dep : constant_deps) {
      XLS_RETURN_IF_ERROR(Visit(dep));
    }
    XLS_RETURN_IF_ERROR(HandleConstantDef(constant_def));
    return DefAlias(constant_def->name_def(), /*to=*/node);
  }

  XLS_ASSIGN_OR_RETURN(auto subject,
                       ResolveColonRefSubjectAfterTypeChecking(
                           import_data_, current_type_info_, node));
  return absl::visit(
      Visitor{
          [&](Module* module) -> absl::Status {
            return absl::InternalError("ColonRefs with imports unhandled.");
          },
          [&](EnumDef* enum_def) -> absl::Status {
            XLS_ASSIGN_OR_RETURN(
                TypeInfo * type_info,
                import_data_->GetRootTypeInfo(enum_def->owner()));
            ScopedTypeInfoSwap stis(this, type_info);
            XLS_ASSIGN_OR_RETURN(Expr * attr_value,
                                 enum_def->GetValue(node->attr()));

            // We've already computed enum member values during constexpr
            // evaluation.
            XLS_ASSIGN_OR_RETURN(InterpValue iv,
                                 current_type_info_->GetConstExpr(attr_value));
            XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(iv));
            Def(node, [this, &value](const SourceInfo& loc) {
              return function_builder_->Literal(value, loc);
            });
            return absl::OkStatus();
          },
          [&](BuiltinNameDef* builtin_name_def) -> absl::Status {
            XLS_ASSIGN_OR_RETURN(
                InterpValue interp_value,
                GetBuiltinNameDefColonAttr(builtin_name_def, node->attr()));
            XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(interp_value));
            DefConst(node, value);
            return absl::OkStatus();
          },
          [&](ArrayTypeAnnotation* array_type) -> absl::Status {
            // Type checking currently ensures that we're not taking a '::' on
            // anything other than a bits type.
            XLS_ASSIGN_OR_RETURN(xls::Type * input_type,
                                 ResolveTypeToIr(array_type));
            xls::BitsType* bits_type = input_type->AsBitsOrDie();
            const int64_t bit_count = bits_type->bit_count();
            XLS_ASSIGN_OR_RETURN(
                InterpValue interp_value,
                GetArrayTypeColonAttr(array_type, bit_count, node->attr()));
            XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(interp_value));
            DefConst(node, value);
            return absl::OkStatus();
          },
      },
      subject);
}

absl::Status FunctionConverter::HandleSplatStructInstance(
    const SplatStructInstance* node) {
  XLS_RETURN_IF_ERROR(Visit(node->splatted()));
  XLS_ASSIGN_OR_RETURN(BValue original, Use(node->splatted()));

  absl::flat_hash_map<std::string, BValue> updates;
  for (const auto& item : node->members()) {
    XLS_RETURN_IF_ERROR(Visit(item.second));
    XLS_ASSIGN_OR_RETURN(updates[item.first], Use(item.second));
  }

  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       ToTypeDefinition(node->struct_ref()));
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def, DerefStruct(type_definition));
  std::vector<BValue> members;
  for (int64_t i = 0; i < struct_def->members().size(); ++i) {
    const std::string& k = struct_def->GetMemberName(i);
    if (auto it = updates.find(k); it != updates.end()) {
      members.push_back(it->second);
    } else {
      members.push_back(function_builder_->TupleIndex(original, i));
    }
  }

  Def(node, [this, &members](const SourceInfo& loc) {
    return function_builder_->Tuple(members, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleStructInstance(
    const StructInstance* node) {
  std::vector<BValue> operands;
  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       ToTypeDefinition(node->struct_ref()));
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def, DerefStruct(type_definition));
  std::vector<Value> const_operands;
  for (const auto& [_, member_expr] : node->GetOrderedMembers(struct_def)) {
    XLS_RETURN_IF_ERROR(Visit(member_expr));
    XLS_ASSIGN_OR_RETURN(BValue operand, Use(member_expr));
    operands.push_back(operand);
  }

  Def(node, [this, &operands](const SourceInfo& loc) {
    return function_builder_->Tuple(operands, loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<std::string> FunctionConverter::GetCalleeIdentifier(
    const Invocation* node) {
  VLOG(5) << "Getting callee identifier for invocation: " << node->ToString();
  Expr* callee = node->callee();
  std::string callee_name;
  Module* m;
  if (auto* name_ref = dynamic_cast<NameRef*>(callee)) {
    callee_name = name_ref->identifier();
    m = module_;
  } else if (auto* colon_ref = dynamic_cast<ColonRef*>(callee)) {
    callee_name = colon_ref->attr();
    std::optional<Import*> import = colon_ref->ResolveImportSubject();
    XLS_RET_CHECK(import.has_value());
    std::optional<const ImportedInfo*> info =
        current_type_info_->GetImported(*import);
    m = (*info)->module;
  } else {
    return absl::InternalError("Invalid callee: " + callee->ToString());
  }

  std::optional<Function*> maybe_f = m->GetFunction(callee_name);
  if (!maybe_f.has_value()) {
    // For e.g. builtins that are not in the module we just provide the name
    // directly.
    return callee_name;
  }
  Function* f = maybe_f.value();

  // We have to mangle the parametric bindings into the name to get the fully
  // resolved symbol.
  absl::btree_set<std::string> free_keys = f->GetFreeParametricKeySet();
  const CallingConvention convention = GetCallingConvention(f);
  if (!f->IsParametric()) {
    return MangleDslxName(m->name(), f->identifier(), convention, free_keys);
  }

  std::optional<const ParametricEnv*> resolved_parametric_env =
      GetInvocationCalleeBindings(node);
  XLS_RET_CHECK(resolved_parametric_env.has_value());
  VLOG(5) << absl::StreamFormat("Node `%s` (%s) @ %s parametric bindings %s",
                                node->ToString(), node->GetNodeTypeName(),
                                node->span().ToString(),
                                (*resolved_parametric_env)->ToString());
  XLS_RET_CHECK(!(*resolved_parametric_env)->empty());
  return MangleDslxName(m->name(), f->identifier(), convention, free_keys,
                        resolved_parametric_env.value());
}

absl::Status FunctionConverter::HandleBinop(const Binop* node) {
  VLOG(5) << "HandleBinop: " << node->ToString();
  std::optional<const Type*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* bits_type = dynamic_cast<const BitsType*>(lhs_type.value());
  bool signed_input = bits_type != nullptr && bits_type->is_signed();
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->rhs()));
  std::function<BValue(const SourceInfo&)> ir_func;

  switch (node->binop_kind()) {
    // Eq and Ne are handled out of line so that they can expand array and tuple
    // comparisons.
    case BinopKind::kEq:
      return HandleEq(node, lhs, rhs);
    case BinopKind::kNe:
      return HandleNe(node, lhs, rhs);
    case BinopKind::kConcat:
      // Concat is handled out of line since it makes different IR ops for bits
      // and array kinds.
      return HandleConcat(node, lhs, rhs);
    // Arithmetic.
    case BinopKind::kAdd:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->Add(lhs, rhs, loc);
      };
      break;
    case BinopKind::kSub:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->Subtract(lhs, rhs, loc);
      };
      break;
    case BinopKind::kMul:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SMul(lhs, rhs, loc);
        }
        return function_builder_->UMul(lhs, rhs, loc);
      };
      break;
    case BinopKind::kDiv:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SDiv(lhs, rhs, loc);
        }
        return function_builder_->UDiv(lhs, rhs, loc);
      };
      break;
    case BinopKind::kMod:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SMod(lhs, rhs, loc);
        }
        return function_builder_->UMod(lhs, rhs, loc);
      };
      break;
    // Non-equality comparisons.
    case BinopKind::kGe:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SGe(lhs, rhs, loc);
        }
        return function_builder_->UGe(lhs, rhs, loc);
      };
      break;
    case BinopKind::kGt:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SGt(lhs, rhs, loc);
        }
        return function_builder_->UGt(lhs, rhs, loc);
      };
      break;
    case BinopKind::kLe:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SLe(lhs, rhs, loc);
        }
        return function_builder_->ULe(lhs, rhs, loc);
      };
      break;
    case BinopKind::kLt:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->SLt(lhs, rhs, loc);
        }
        return function_builder_->ULt(lhs, rhs, loc);
      };
      break;
    // Shifts.
    case BinopKind::kShr:
      ir_func = [&](const SourceInfo& loc) {
        if (signed_input) {
          return function_builder_->Shra(lhs, rhs, loc);
        }
        return function_builder_->Shrl(lhs, rhs, loc);
      };
      break;
    case BinopKind::kShl:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->Shll(lhs, rhs, loc);
      };
      break;
    // Bitwise.
    case BinopKind::kXor:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->Xor(lhs, rhs, loc);
      };
      break;
    case BinopKind::kAnd:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->And(lhs, rhs, loc);
      };
      break;
    case BinopKind::kOr:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->Or(lhs, rhs, loc);
      };
      break;
    // Logical.
    case BinopKind::kLogicalAnd:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->And(lhs, rhs, loc);
      };
      break;
    case BinopKind::kLogicalOr:
      ir_func = [&](const SourceInfo& loc) {
        return function_builder_->Or(lhs, rhs, loc);
      };
      break;
  }
  Def(node, ir_func);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleAttr(const Attr* node) {
  VLOG(5) << "FunctionConverter::HandleAttr: " << node->ToString() << " @ "
          << node->span().ToString();
  XLS_RETURN_IF_ERROR(Visit(node->lhs()));
  std::optional<const Type*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* struct_type = dynamic_cast<const StructType*>(lhs_type.value());
  std::string_view identifier = node->attr();
  XLS_ASSIGN_OR_RETURN(int64_t index, struct_type->GetMemberIndex(identifier));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));
  BValue ir = Def(node, [&](const SourceInfo& loc) {
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

absl::Status FunctionConverter::HandleBlock(const Block* node) {
  VLOG(5) << "FunctionConverter::HandleBlock; node: " << node->ToString();
  Expr* last_expr = nullptr;

  for (const Statement* s : node->statements()) {
    // We just want to see if it's an expr for "last expr in the block"
    // purposes. Generally we'll do Visit on the statement node to handle its
    // contents.
    if (std::holds_alternative<Expr*>(s->wrapped())) {
      last_expr = std::get<Expr*>(s->wrapped());
    }
    XLS_RETURN_IF_ERROR(Visit(s));
  }

  if (node->trailing_semi() || last_expr == nullptr) {
    // Define the block result as nil.
    Def(node, [&](const SourceInfo& loc) {
      return function_builder_->Tuple({}, loc);
    });
  } else {
    XLS_RET_CHECK(last_expr != nullptr);
    XLS_ASSIGN_OR_RETURN(BValue bvalue, Use(last_expr));
    SetNodeToIr(node, bvalue);
  }
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleStatement(const Statement* node) {
  VLOG(5) << "FunctionConverter::HandleStatement; node: " << node->ToString();
  return absl::visit(Visitor{
                         [&](Expr* e) -> absl::Status {
                           XLS_RETURN_IF_ERROR(Visit(ToAstNode(e)));
                           XLS_ASSIGN_OR_RETURN(BValue bvalue, Use(e));
                           SetNodeToIr(node, bvalue);
                           return absl::OkStatus();
                         },
                         [&](TypeAlias* n) -> absl::Status {
                           // Nothing to do, all was resolved at type inference
                           // time.
                           return absl::OkStatus();
                         },
                         [&](ConstAssert* n) -> absl::Status {
                           // Nothing to do, all was resolved at type inference
                           // time.
                           return absl::OkStatus();
                         },
                         [&](Let* let) -> absl::Status {
                           XLS_RETURN_IF_ERROR(Visit(ToAstNode(let)));
                           XLS_ASSIGN_OR_RETURN(BValue bvalue, Use(let));
                           SetNodeToIr(node, bvalue);
                           return absl::OkStatus();
                         },
                     },
                     node->wrapped());
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleConditional(const Conditional* node) {
  XLS_RETURN_IF_ERROR(Visit(node->test()));
  XLS_ASSIGN_OR_RETURN(BValue arg0, Use(node->test()));

  {
    ScopedControlPredicate scp(
        this, [&](const PredicateFun& orig_control_predicate) {
          BValue activated = orig_control_predicate();
          CHECK_EQ(activated.GetType()->AsBitsOrDie()->bit_count(), 1);
          return function_builder_->And(activated, arg0);
        });
    XLS_RETURN_IF_ERROR(Visit(node->consequent()));
  }

  XLS_ASSIGN_OR_RETURN(BValue arg1, Use(node->consequent()));

  {
    ScopedControlPredicate scp(
        this, [&](const PredicateFun& orig_control_predicate) {
          BValue activated = orig_control_predicate();
          CHECK_EQ(activated.GetType()->AsBitsOrDie()->bit_count(), 1);
          return function_builder_->And(orig_control_predicate(),
                                        function_builder_->Not(arg0));
        });
    XLS_RETURN_IF_ERROR(Visit(ToExprNode(node->alternate())));
  }

  XLS_ASSIGN_OR_RETURN(BValue arg2, Use(ToExprNode(node->alternate())));

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Select(arg0, arg1, arg2, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinAndReduce(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->AndReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinArraySize(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  // All array sizes are constexpr since they're based on known types.
  XLS_ASSIGN_OR_RETURN(InterpValue iv, current_type_info_->GetConstExpr(node));
  XLS_ASSIGN_OR_RETURN(Value v, InterpValueToValue(iv));
  DefConst(node, v);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinArrayRev(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  auto* array_type = arg.GetType()->AsArrayOrDie();

  Def(node, [&](const SourceInfo& loc) {
    std::vector<BValue> elems;
    for (int64_t i = 0; i < array_type->size(); ++i) {
      BValue index = function_builder_->Literal(
          UBits(static_cast<uint64_t>(array_type->size() - i - 1), 64));
      elems.push_back(function_builder_->ArrayIndex(arg, {index}, loc));
    }
    return function_builder_->Array(elems, array_type->element_type(), loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinZip(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->args()[1]));

  XLS_ASSIGN_OR_RETURN(xls::Type * result_type_base, ResolveTypeToIr(node));
  // Should never fail, because type inference should ensure this is a array
  // type.
  xls::ArrayType* result_type = down_cast<xls::ArrayType*>(result_type_base);

  Def(node, [&](const SourceInfo& loc) {
    std::vector<BValue> elems;
    int64_t result_size = result_type->size();
    for (int64_t i = 0; i < result_size; ++i) {
      BValue index = function_builder_->Literal(UBits(i, kUsizeBits));

      BValue lhs_element = function_builder_->ArrayIndex(lhs, {index}, loc);
      BValue rhs_element = function_builder_->ArrayIndex(rhs, {index}, loc);
      elems.push_back(
          function_builder_->Tuple({lhs_element, rhs_element}, loc));
    }
    return function_builder_->Array(elems, result_type->element_type(), loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinArraySlice(
    const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue start, Use(node->args()[1]));

  const Expr* arg2 = node->args()[2];
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> output_type, ResolveType(arg2));
  const auto* array_type = down_cast<const ArrayType*>(output_type.get());
  XLS_ASSIGN_OR_RETURN(int64_t width, array_type->size().GetAsInt64());

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->ArraySlice(arg, start, width, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinBitSliceUpdate(
    const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue start, Use(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(BValue update_value, Use(node->args()[2]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->BitSliceUpdate(arg, start, update_value, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinClz(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node,
      [&](const SourceInfo& loc) { return function_builder_->Clz(arg, loc); });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinCtz(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node,
      [&](const SourceInfo& loc) { return function_builder_->Ctz(arg, loc); });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinGate(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue predicate, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue value, Use(node->args()[1]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Gate(predicate, value, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinDecode(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_RET_CHECK_EQ(node->explicit_parametrics().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue input, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(
      xls::Type * return_type,
      ResolveTypeToIr(ToAstNode(node->explicit_parametrics()[0])));
  XLS_RET_CHECK(return_type->IsBits());
  const int64_t width = return_type->AsBitsOrDie()->bit_count();

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Decode(input, width, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinEncode(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue input, Use(node->args()[0]));

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Encode(input, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinOneHot(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue input, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(Bits lsb_prio, GetConstBits(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(uint64_t lsb_prio_value, lsb_prio.ToUint64());

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->OneHot(
        input, lsb_prio_value != 0 ? LsbOrMsb::kLsb : LsbOrMsb::kMsb, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinOneHotSel(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue selector, Use(node->args()[0]));

  // Implementation note:  During IR conversion, we will scalarize the
  // array element into BValues using multiple ArrayIndex ops
  // to create cases for the select operation.  This will bloat the
  // unoptimized IR  -- especially in the case where we are given
  // a literal array.
  //
  // For example, given a `one_hot_sel(sel, cases=[a, b, c])` the un-opt IR
  // will redundantly create individual array_index ops for each element
  // to then pass to the select op:
  //   array_val: bits[32][4] = array(a, b, c)
  //   literal_0: bits[32] = literal(value=0)
  //   literal_1: bits[32] = literal(value=1)
  //   literal_2: bits[32] = literal(value=2)
  //   array_index_0: bits[32] = array_index(array_val, indices=[literal_0])
  //   array_index_1: bits[32] = array_index(array_val, indices=[literal_1])
  //   array_index_2: bits[32] = array_index(array_val, indices=[literal_2])
  //   one_hot_sel_val: bits[32]
  //     = one_hot_sel(s, cases=[array_index_0, array_index_1, array_index_2])
  //
  // This is ok as the optimizer will look through this and simplify to
  //   one_hot_sel_val: bits[32] = one_hot_sel(s, cases=[a, b, c])
  //
  const Expr* cases_arg = node->args()[1];
  std::vector<BValue> cases;

  XLS_ASSIGN_OR_RETURN(BValue bvalue_cases_arg, Use(cases_arg));
  XLS_ASSIGN_OR_RETURN(xls::ArrayType * cases_arg_type,
                       bvalue_cases_arg.GetType()->AsArray());

  Def(node, [&](const SourceInfo& loc) {
    for (int64_t i = 0; i < cases_arg_type->size(); ++i) {
      BValue index = function_builder_->Literal(UBits(i, kUsizeBits), loc);
      BValue bvalue_case =
          function_builder_->ArrayIndex(bvalue_cases_arg, {index}, loc);
      cases.push_back(bvalue_case);
    }

    return function_builder_->OneHotSelect(selector, cases, loc);
  });

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinPrioritySel(
    const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue selector, Use(node->args()[0]));

  // See implementation note for HandleBuildinOneHotSel().
  const Expr* cases_arg = node->args()[1];
  std::vector<BValue> cases;

  XLS_ASSIGN_OR_RETURN(BValue bvalue_cases_arg, Use(cases_arg));
  XLS_ASSIGN_OR_RETURN(xls::ArrayType * cases_arg_type,
                       bvalue_cases_arg.GetType()->AsArray());

  Def(node, [&](const SourceInfo& loc) {
    for (int64_t i = 0; i < cases_arg_type->size(); ++i) {
      BValue index = function_builder_->Literal(UBits(i, kUsizeBits), loc);
      BValue bvalue_case =
          function_builder_->ArrayIndex(bvalue_cases_arg, {index}, loc);
      cases.push_back(bvalue_case);
    }

    return function_builder_->PrioritySelect(selector, cases, loc);
  });

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinOrReduce(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->OrReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinRev(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Reverse(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinSignex(const Invocation* node) {
  VLOG(5) << "FunctionConverter::HandleBuiltinSignex: " << node->ToString();
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));

  std::optional<const Type*> maybe_lhs_type =
      current_type_info_->GetItem(node->args()[0]);
  XLS_RET_CHECK(maybe_lhs_type.has_value());
  XLS_ASSIGN_OR_RETURN(TypeDim lhs_total_bit_count,
                       maybe_lhs_type.value()->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t old_bit_count, lhs_total_bit_count.GetAsInt64());

  std::optional<const Type*> maybe_rhs_type =
      current_type_info_->GetItem(node->args()[1]);
  XLS_RET_CHECK(maybe_rhs_type.has_value());
  XLS_ASSIGN_OR_RETURN(TypeDim rhs_total_bit_count,
                       maybe_rhs_type.value()->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t new_bit_count, rhs_total_bit_count.GetAsInt64());

  Def(node, [&](const SourceInfo& loc) {
    if (new_bit_count < old_bit_count) {
      return function_builder_->BitSlice(arg, 0, new_bit_count, loc);
    }
    return function_builder_->SignExtend(arg, new_bit_count, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinSMulp(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->args()[1]));

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->SMulp(lhs, rhs);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinUpdate(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  std::vector<BValue> indices;
  if (node->args()[1]->kind() == AstNodeKind::kXlsTuple) {
    for (auto c : node->args()[1]->GetChildren(false)) {
      XLS_ASSIGN_OR_RETURN(BValue index, Use(c));
      indices.push_back(index);
    }
  } else {
    XLS_ASSIGN_OR_RETURN(BValue index, Use(node->args()[1]));
    indices.push_back(index);
  }
  XLS_ASSIGN_OR_RETURN(BValue new_value, Use(node->args()[2]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->ArrayUpdate(arg, new_value, indices, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinUMulp(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->args()[1]));

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->UMulp(lhs, rhs);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinXorReduce(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->XorReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::CastToArray(const Cast* node,
                                            const ArrayType& output_type) {
  XLS_ASSIGN_OR_RETURN(BValue bits, Use(node->expr()));
  std::vector<BValue> slices;
  XLS_ASSIGN_OR_RETURN(TypeDim element_bit_count_dim,
                       output_type.element_type().GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t element_bit_count,
                       TypeDim::GetAs64Bits(element_bit_count_dim.value()));
  XLS_ASSIGN_OR_RETURN(int64_t array_size,
                       TypeDim::GetAs64Bits(output_type.size().value()));
  // MSb becomes lowest-indexed array element.
  for (int64_t i = 0; i < array_size; ++i) {
    slices.push_back(function_builder_->BitSlice(bits, i * element_bit_count,
                                                 element_bit_count));
  }
  std::reverse(slices.begin(), slices.end());
  xls::Type* element_type = package()->GetBitsType(element_bit_count);
  Def(node, [this, &slices, element_type](const SourceInfo& loc) {
    return function_builder_->Array(slices, element_type, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::CastFromArray(const Cast* node,
                                              const Type& output_type) {
  XLS_ASSIGN_OR_RETURN(BValue array, Use(node->expr()));
  XLS_ASSIGN_OR_RETURN(xls::Type * input_type, ResolveTypeToIr(node->expr()));
  xls::ArrayType* array_type = input_type->AsArrayOrDie();
  const int64_t array_size = array_type->size();
  std::vector<BValue> pieces;
  for (int64_t i = 0; i < array_size; ++i) {
    BValue index = function_builder_->Literal(UBits(i, kUsizeBits));
    pieces.push_back(function_builder_->ArrayIndex(array, {index}));
  }
  Def(node, [this, &pieces](const SourceInfo& loc) {
    return function_builder_->Concat(pieces, loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<FunctionConverter::DerefVariant>
FunctionConverter::DerefStructOrEnum(TypeDefinition node) {
  while (std::holds_alternative<TypeAlias*>(node)) {
    auto* type_alias = std::get<TypeAlias*>(node);
    TypeAnnotation& annotation = type_alias->type_annotation();
    if (auto* type_ref_annotation =
            dynamic_cast<TypeRefTypeAnnotation*>(&annotation)) {
      node = type_ref_annotation->type_ref()->type_definition();
    } else {
      return absl::UnimplementedError(
          "Unhandled typedef for resolving to struct-or-enum: " +
          annotation.ToString());
    }
  }

  if (std::holds_alternative<StructDef*>(node)) {
    return std::get<StructDef*>(node);
  }
  if (std::holds_alternative<EnumDef*>(node)) {
    return std::get<EnumDef*>(node);
  }

  XLS_RET_CHECK(std::holds_alternative<ColonRef*>(node));
  auto* colon_ref = std::get<ColonRef*>(node);
  std::optional<Import*> import = colon_ref->ResolveImportSubject();
  XLS_RET_CHECK(import.has_value());
  std::optional<const ImportedInfo*> info =
      current_type_info_->GetImported(*import);
  Module* imported_mod = (*info)->module;
  ScopedTypeInfoSwap stis(this, (*info)->type_info);
  XLS_ASSIGN_OR_RETURN(TypeDefinition td,
                       imported_mod->GetTypeDefinition(colon_ref->attr()));
  // Recurse to resolve the typedef within the imported module.
  return DerefStructOrEnum(td);
}

absl::StatusOr<StructDef*> FunctionConverter::DerefStruct(TypeDefinition node) {
  XLS_ASSIGN_OR_RETURN(DerefVariant v, DerefStructOrEnum(node));
  XLS_RET_CHECK(std::holds_alternative<StructDef*>(v));
  return std::get<StructDef*>(v);
}

absl::StatusOr<EnumDef*> FunctionConverter::DerefEnum(TypeDefinition node) {
  XLS_ASSIGN_OR_RETURN(DerefVariant v, DerefStructOrEnum(node));
  XLS_RET_CHECK(std::holds_alternative<EnumDef*>(v));
  return std::get<EnumDef*>(v);
}

absl::StatusOr<std::unique_ptr<Type>> FunctionConverter::ResolveType(
    const AstNode* node) {
  XLS_RET_CHECK(current_type_info_ != nullptr);
  std::optional<const Type*> t = current_type_info_->GetItem(node);
  if (!t.has_value()) {
    return IrConversionErrorStatus(
        node->GetSpan(),
        absl::StrFormat("Failed to convert IR because type was missing for AST "
                        "node: %s (kind: %s)",
                        node->ToString(), AstNodeKindToString(node->kind())));
  }

  return t.value()->MapSize([this](const TypeDim& dim) {
    return ResolveDim(dim, ParametricEnv(parametric_env_map_));
  });
}

absl::StatusOr<Value> FunctionConverter::GetConstValue(
    const AstNode* node) const {
  std::optional<IrValue> ir_value = GetNodeToIr(node);
  if (!ir_value.has_value()) {
    return absl::InternalError(
        absl::StrFormat("AST node had no associated IR value: %s @ %s",
                        node->ToString(), SpanToString(node->GetSpan())));
  }
  if (!std::holds_alternative<CValue>(*ir_value)) {
    return absl::InternalError(absl::StrFormat(
        "AST node had a non-const IR value: %s", node->ToString()));
  }
  return std::get<CValue>(*ir_value).ir_value;
}

absl::StatusOr<Bits> FunctionConverter::GetConstBits(
    const AstNode* node) const {
  XLS_ASSIGN_OR_RETURN(Value value, GetConstValue(node));
  return value.GetBitsWithStatus();
}

absl::Status FunctionConverter::HandleConstantArray(const ConstantArray* node) {
  // Note: previously we would force constant evaluation here, but because all
  // constexprs should be evaluated during typechecking, we shouldn't need to
  // forcibly do constant evaluation at IR conversion time; therefore, we just
  // build BValues and let XLS opt constant fold them.
  return HandleArray(node);
}

absl::StatusOr<xls::Type*> FunctionConverter::ResolveTypeToIr(
    const AstNode* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> type, ResolveType(node));
  return TypeToIr(package_data_.conversion_info->package.get(), *type,
                  ParametricEnv(parametric_env_map_));
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
        return Value::Tuple(ir_values);
      }
      return Value::Array(ir_values);
    }
    case InterpValueTag::kToken:
      return Value::Token();
    default:
      return absl::InvalidArgumentError(
          "Cannot convert interpreter value with tag: " +
          TagToString(iv.tag()));
  }
}

absl::StatusOr<std::vector<ConstantDef*>> GetConstantDepFreevars(
    AstNode* node) {
  Span span = node->GetSpan().value();
  FreeVariables free_variables = GetFreeVariables(node, &span.start());
  std::vector<std::pair<std::string, AnyNameDef>> freevars =
      free_variables.GetNameDefTuples();
  std::vector<ConstantDef*> constant_deps;
  for (const auto& [identifier, any_name_def] : freevars) {
    if (std::holds_alternative<BuiltinNameDef*>(any_name_def)) {
      continue;
    }
    const auto* name_def = std::get<const NameDef*>(any_name_def);
    AstNode* definer = name_def->definer();
    if (auto* constant_def = dynamic_cast<ConstantDef*>(definer)) {
      XLS_ASSIGN_OR_RETURN(auto sub_deps, GetConstantDepFreevars(constant_def));
      constant_deps.insert(constant_deps.end(), sub_deps.begin(),
                           sub_deps.end());
      constant_deps.push_back(constant_def);
    } else if (auto* enum_def = dynamic_cast<EnumDef*>(definer)) {
      XLS_ASSIGN_OR_RETURN(auto sub_deps, GetConstantDepFreevars(enum_def));
      constant_deps.insert(constant_deps.end(), sub_deps.begin(),
                           sub_deps.end());
    } else {
      // Not something we recognize as needing free variable analysis.
    }
  }
  return constant_deps;
}

}  // namespace xls::dslx
