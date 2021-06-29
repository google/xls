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

// Implementation note on rough usage of verbose logging levels in this file:
//
//  2: Function-scope conversion activity (things that happen a few times on a
//     per-function basis).
//  3: Conversion order.
//  5: Interesting events that may occur several times within a function
//     conversion.
//  6: Interesting events that may occur many times (and will generally be more
//     noisy) within a function conversion.

#include "xls/dslx/ir_converter.h"

#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/types/variant.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/deduce_ctx.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/extract_conversion_order.h"
#include "xls/dslx/interpreter.h"
#include "xls/dslx/mangle.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/lsb_or_msb.h"

namespace xls::dslx {
namespace {

// Bundles together a package pointer with a supplementary map we keep that
// shows the DSLX function that led to IR functions in the package.
struct PackageData {
  Package* package;
  absl::flat_hash_map<xls::Function*, dslx::Function*> ir_to_dslx;
  absl::flat_hash_set<xls::Function*> wrappers;
};

// Returns a status that indicates an error in the IR conversion process.
absl::Status ConversionErrorStatus(const absl::optional<Span>& span,
                                   absl::string_view message) {
  return absl::InternalError(
      absl::StrFormat("ConversionError: %s %s",
                      span ? span->ToString() : "<no span>", message));
}

// A function that creates/returns a predicate value -- since this is used
// frequently when making "chained" predicates through control constructs, we
// give it an alias.
using PredicateFun = std::function<BValue()>;

// If the function being converted has an implicit token; e.g. caused by
// the need to have `assert` IR ops (via DSLX `fail!`), it is created here.
// Whenever we have an entry token we also have an associated "was I activated"
// boolean which will help guard the associated assertion(s).
//
// Similarly when there's an entry token we need to keep track of current
// control predicates, and join all the assertion tokens together at the end,
// into the output token.
struct ImplicitTokenData {
  BValue entry_token;
  BValue activated;
  PredicateFun create_control_predicate;
  // Used for sequencing by both fail! and cover! ops.
  std::vector<BValue> control_tokens;
};

// Wrapper around the type information query for whether DSL function "f"
// requires an implicit token calling convention.
//
// This query is not necessary when emit_fail_as_assert is off, then we never
// use the "implicit token" calling convention.
static bool GetRequiresImplicitToken(dslx::Function* f, ImportData* import_data,
                                     const ConvertOptions& options) {
  if (!options.emit_fail_as_assert) {
    return false;
  }
  absl::optional<bool> requires_opt = import_data->GetRootTypeInfo(f->owner())
                                          .value()
                                          ->GetRequiresImplicitToken(f);
  XLS_CHECK(requires_opt.has_value());
  return requires_opt.value();
}

// Helper type that creates XLS IR for a function -- this is done within a
// Package, given a DSLX AST, its type information, and an entry point function.
//
// It mainly encapsulates DSLX=>IR mapping `node_to_ir_` used for resolution of
// FunctionBuilder values (BValue) in the conversion rules.
//
// Implementation note: these are "throwaway objects"; i.e. instantiate them
// once to visit a single function and then discard. It exists mostly to avoid
// needing to thread conversion state between a bunch of on-the-wall functions.
//
// A function's external constant dependencies (e.g. if a function refers to a
// module-level constant as a free variable) have to be noted from the outside
// world via calls to `AddConstantDep()`.
//
// Implementation note: we don't tack a `Function*` attribute on here because we
// use this same facility for translating the body of a for loop into a function
// -- for that "anonymous function" there's not a direct correspondence to a
// DSLX function.
class FunctionConverter {
 public:
  FunctionConverter(PackageData& package_data, Module* module,
                    ImportData* import_data, ConvertOptions options);

  // Main entry point to request conversion of the DSLX function "f" to an IR
  // function.
  absl::StatusOr<xls::Function*> HandleFunction(
      Function* node, TypeInfo* type_info,
      const SymbolicBindings* symbolic_bindings);

  // Notes a constant-definition dependency for the function (so it can
  // participate in the IR conversion).
  void AddConstantDep(ConstantDef* constant_def);

 private:
  // Helper class used for dispatching to IR conversion methods.
  friend class FunctionConverterVisitor;

  friend struct ScopedTypeInfoSwap;

  // Helper class used for chaining on/off control predicates.
  friend class ScopedControlPredicate;

  // Wraps up a type so it is (token, u1, type).
  static Type* WrapIrForImplicitTokenType(Type* type, Package* package) {
    Type* token_type = package->GetTokenType();
    Type* u1_type = package->GetBitsType(1);
    return package->GetTupleType({token_type, u1_type, type});
  }

  // Helper function used for adding a parameter type wrapped up in a
  // token/activation boolean.
  BValue AddTokenWrappedParam(Type* type) {
    FunctionBuilder& fb = *function_builder_;
    Type* wrapped_type = WrapIrForImplicitTokenType(type, package());
    BValue param = fb.Param("__token_wrapped", wrapped_type);
    BValue entry_token = fb.TupleIndex(param, 0);
    BValue activated = fb.TupleIndex(param, 1);
    auto create_control_predicate = [activated] { return activated; };
    implicit_token_data_ =
        ImplicitTokenData{entry_token, activated, create_control_predicate};
    BValue unwrapped = fb.TupleIndex(param, 2);
    return unwrapped;
  }

  // An IR-conversion-time-constant value, decorates a BValue with its evaluated
  // constant form.
  struct CValue {
    Value ir_value;
    BValue value;
  };

  // Every AST node has an "IR value" that is either a function builder value
  // (BValue) or its IR-conversion-time-constant-decorated cousin (CValue).
  using IrValue = absl::variant<BValue, CValue>;

  // Helper for converting an IR value to its BValue pointer for use in
  // debugging.
  static std::string ToString(const IrValue& value);

  void InstantiateFunctionBuilder(absl::string_view mangled_name);

  // See `GetRequiresImplicitToken(f, import_data, options)`.
  bool GetRequiresImplicitToken(dslx::Function* f) const {
    return xls::dslx::GetRequiresImplicitToken(f, import_data_, options_);
  }

  CallingConvention GetCallingConvention(Function* f) const {
    return GetRequiresImplicitToken(f) ? CallingConvention::kImplicitToken
                                       : CallingConvention::kTypical;
  }

  // Populates the implicit_token_data_.
  absl::Status AddImplicitTokenParams();

  // Aliases the (IR) result of AST node 'from' with AST node 'to'.
  //
  // That is, 'from' has already been emitted, and we want 'to' to just be
  // whatever 'from' has as an IR value.
  //
  // Returns the aliased IR BValue both AST nodes now map to.
  absl::StatusOr<BValue> DefAlias(AstNode* from, AstNode* to);

  // Returns the BValue previously noted as corresponding to "node" (via a
  // Def/DefAlias).
  absl::StatusOr<BValue> Use(AstNode* node) const;

  void SetNodeToIr(AstNode* node, IrValue value);
  absl::optional<IrValue> GetNodeToIr(AstNode* node) const;

  // Returns the constant value corresponding to the IrValue of "node", or
  // returns an error if it is not present (or not constant).
  absl::StatusOr<Value> GetConstValue(AstNode* node) const;

  // As above, but also checks it is a constant Bits value.
  absl::StatusOr<Bits> GetConstBits(AstNode* node) const;

  // Resolves "dim" (from a possible parametric) against the
  // symbolic_binding_map_.
  absl::StatusOr<ConcreteTypeDim> ResolveDim(ConcreteTypeDim dim);

  // As above, does ResolveDim() but then accesses the dimension value as an
  // expected int64_t.
  absl::StatusOr<int64_t> ResolveDimToInt(const ConcreteTypeDim& dim) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim resolved, ResolveDim(dim));
    if (absl::holds_alternative<InterpValue>(resolved.value())) {
      return absl::get<InterpValue>(resolved.value()).GetBitValueInt64();
    }
    return absl::InternalError(absl::StrFormat(
        "Expected resolved dimension of %s to be an integer, got: %s",
        dim.ToString(), resolved.ToString()));
  }

  // Resolves node's type and resolves all of its dimensions via `ResolveDim()`.
  absl::StatusOr<std::unique_ptr<ConcreteType>> ResolveType(AstNode* node);

  // Helper that composes ResolveType() and TypeToIr().
  absl::StatusOr<xls::Type*> ResolveTypeToIr(AstNode* node);

  // -- Accessors

  void SetSymbolicBindings(const SymbolicBindings* value) {
    if (value == nullptr) {
      symbolic_binding_map_.clear();
    } else {
      symbolic_binding_map_ = value->ToMap();
    }
  }
  void set_symbolic_binding_map(
      absl::flat_hash_map<std::string, InterpValue> map) {
    symbolic_binding_map_ = std::move(map);
  }

  // Gets the current counter of counted_for loops we've observed and bumps it.
  // This is useful for generating new symbols for the functions that serve as
  // XLS counted_for "bodies".
  int64_t GetAndBumpCountedForCount() { return counted_for_count_++; }

  // TODO(leary): 2020-12-22 Clean all this up to expose a minimal surface area
  // once everything is ported over to C++.
  absl::optional<InterpValue> get_symbolic_binding(
      absl::string_view identifier) const {
    auto it = symbolic_binding_map_.find(identifier);
    if (it == symbolic_binding_map_.end()) {
      return absl::nullopt;
    }
    return it->second;
  }
  // Note: this object holds a map, but we can return a SymbolicBindings object
  // on demand.
  SymbolicBindings GetSymbolicBindings() const {
    return SymbolicBindings(symbolic_binding_map_);
  }

  // Returns the symbolic bindings in symbolic_binding_map_ that are not
  // module-level constants -- the typechecker doesn't care about module-level
  // constants.
  //
  // TODO(leary): 2020-01-08 This seems broken, what if a local parametric
  // shadows a module-level constant? We should use "stacked" bindings that look
  // like a scope chain.
  SymbolicBindings GetSymbolicBindingsTuple() const;

  // Returns the symbolic bindings to be used in the callee for this invocation.
  //
  // We must provide the current evaluation context (module_name, function_name,
  // caller_symbolic_bindings) in order to retrieve the correct symbolic
  // bindings to use in the callee invocation.
  //
  // Args:
  //  invocation: Invocation that the bindings are being retrieved for.
  //
  // Returns:
  //  The symbolic bindings for the given invocation.
  absl::optional<const SymbolicBindings*> GetInstantiationCalleeBindings(
      Invocation* invocation) const {
    SymbolicBindings key = GetSymbolicBindingsTuple();
    return import_data_->GetRootTypeInfo(invocation->owner())
        .value()
        ->GetInstantiationCalleeBindings(invocation, key);
  }

  // Helpers for HandleBinop().
  absl::Status HandleConcat(Binop* node, BValue lhs, BValue rhs);
  absl::Status HandleEq(Binop* node, BValue lhs, BValue rhs);
  absl::Status HandleNe(Binop* node, BValue lhs, BValue rhs);

  using BuildTermFn =
      std::function<xls::BValue(xls::BValue lhs, xls::BValue rhs)>;

  // Helpers for HandleEq / HandleNe
  absl::StatusOr<std::vector<BValue>> BuildTerms(
      BValue lhs, BValue rhs, absl::optional<SourceLocation>& loc,
      const BuildTermFn& build_term);
  absl::StatusOr<BValue> BuildTest(BValue lhs, BValue rhs, Op op, Value base,
                                   absl::optional<SourceLocation>& loc,
                                   const BuildTermFn& build_term);

  // AstNode handlers.
  absl::Status HandleBinop(Binop* node);
  absl::Status HandleConstRef(ConstRef* node);
  absl::Status HandleNameRef(NameRef* node);
  absl::Status HandleNumber(Number* node);
  absl::Status HandleParam(Param* node);
  absl::Status HandleString(String* node);
  absl::Status HandleUnop(Unop* node);
  absl::Status HandleXlsTuple(XlsTuple* node);

  // AstNode handlers that recur "manually" internal to the handler.
  absl::Status HandleArray(Array* node);
  absl::Status HandleAttr(Attr* node);
  absl::Status HandleCast(Cast* node);
  absl::Status HandleColonRef(ColonRef* node);
  absl::Status HandleConstantArray(ConstantArray* node);
  absl::Status HandleConstantDef(ConstantDef* node);
  absl::Status HandleFor(For* node);
  absl::Status HandleIndex(Index* node);
  absl::Status HandleInvocation(Invocation* node);
  absl::Status HandleLet(Let* node);
  absl::Status HandleMatch(Match* node);
  absl::Status HandleSplatStructInstance(SplatStructInstance* node);
  absl::Status HandleStructInstance(StructInstance* node);
  absl::Status HandleTernary(Ternary* node);

  // Handles invocation of a user-defined function (UDF).
  absl::Status HandleUdfInvocation(Invocation* node, xls::Function* f,
                                   std::vector<BValue> args);

  // Handles the fail!() builtin invocation.
  absl::Status HandleFailBuiltin(Invocation* node, BValue arg);

  // Handles the cover!() builtin invocation.
  absl::Status HandleCoverBuiltin(Invocation* node, BValue condition);

  // Handles an arm of a match expression.
  absl::StatusOr<BValue> HandleMatcher(NameDefTree* matcher,
                                       absl::Span<const int64_t> index,
                                       const BValue& matched_value,
                                       const ConcreteType& matched_type);

  // Makes the specified builtin available to the package.
  absl::StatusOr<BValue> DefMapWithBuiltin(
      Invocation* parent_node, NameRef* node, AstNode* arg,
      const SymbolicBindings& symbolic_bindings);

  // Evaluates a constexpr AST Invocation via the DSLX interpreter.
  //
  // Evaluates an Invocation node whose argument values are all known at
  // compile/interpret time, yielding a constant value that can be inserted
  // into the IR.
  //
  // Args:
  //  node: The Invocation node to evaluate.
  //
  // Returns:
  //   The XLS (IR) Value containing the result.
  absl::StatusOr<Value> EvaluateConstFunction(Invocation* node);

  absl::StatusOr<BValue> HandleMap(Invocation* node);

  absl::StatusOr<BValue> HandleFail(Invocation* node);

  // Builtin invocation handlers.
  absl::Status HandleBuiltinAndReduce(Invocation* node);
  absl::Status HandleBuiltinArraySlice(Invocation* node);
  absl::Status HandleBuiltinBitSlice(Invocation* node);
  absl::Status HandleBuiltinBitSliceUpdate(Invocation* node);
  absl::Status HandleBuiltinClz(Invocation* node);
  absl::Status HandleBuiltinCtz(Invocation* node);
  absl::Status HandleBuiltinOneHot(Invocation* node);
  absl::Status HandleBuiltinOneHotSel(Invocation* node);
  absl::Status HandleBuiltinOrReduce(Invocation* node);
  absl::Status HandleBuiltinRev(Invocation* node);
  absl::Status HandleBuiltinScmp(SignedCmp cmp, Invocation* node);
  absl::Status HandleBuiltinSignex(Invocation* node);
  absl::Status HandleBuiltinUpdate(Invocation* node);
  absl::Status HandleBuiltinXorReduce(Invocation* node);

  // Signed comparisons.
  absl::Status HandleBuiltinSLt(Invocation* node) {
    return HandleBuiltinScmp(SignedCmp::kLt, node);
  }
  absl::Status HandleBuiltinSLe(Invocation* node) {
    return HandleBuiltinScmp(SignedCmp::kLe, node);
  }
  absl::Status HandleBuiltinSGe(Invocation* node) {
    return HandleBuiltinScmp(SignedCmp::kGe, node);
  }
  absl::Status HandleBuiltinSGt(Invocation* node) {
    return HandleBuiltinScmp(SignedCmp::kGt, node);
  }

  // Derefences the type definition to a struct definition.
  absl::StatusOr<StructDef*> DerefStruct(TypeDefinition node);
  absl::StatusOr<StructDef*> DerefStruct(NameRef* name_ref) {
    return DerefStructOrEnumFromNameRef<StructDef*>(
        name_ref, [this](TypeDefinition td) { return DerefStruct(td); });
  }

  // Derefences the type definition to a enum definition.
  absl::StatusOr<EnumDef*> DerefEnum(TypeDefinition node);
  absl::StatusOr<EnumDef*> DerefEnum(NameRef* name_ref) {
    return DerefStructOrEnumFromNameRef<EnumDef*>(
        name_ref, [this](TypeDefinition td) { return DerefEnum(td); });
  }

  absl::Status CastToArray(Cast* node, const ArrayType& output_type);
  absl::Status CastFromArray(Cast* node, const ConcreteType& output_type);

  // Returns the fully resolved (mangled) name for the callee of the given node,
  // with parametric values mangled in appropriately.
  absl::StatusOr<std::string> GetCalleeIdentifier(Invocation* node);

  // Determines whether the for loop node is of the general form:
  //
  //  `for ... in range(0, N)`
  //
  // Returns the value of N if so, or a conversion error if it is not.
  absl::StatusOr<int64_t> QueryConstRangeCall(For* node);

  template <typename T>
  absl::StatusOr<T> DerefStructOrEnumFromNameRef(
      NameRef* name_ref,
      const std::function<absl::StatusOr<T>(TypeDefinition)>& f) {
    AnyNameDef any_name_def = name_ref->name_def();
    auto* name_def = absl::get<NameDef*>(any_name_def);
    AstNode* definer = name_def->definer();
    XLS_ASSIGN_OR_RETURN(TypeDefinition td, ToTypeDefinition(definer));
    return f(td);
  }

  // Dereferences a type definition to either a struct definition or enum
  // definition.
  using DerefVariant = absl::variant<StructDef*, EnumDef*>;
  absl::StatusOr<DerefVariant> DerefStructOrEnum(TypeDefinition node);

  // Converts a concrete type to its corresponding IR representation.
  absl::StatusOr<xls::Type*> TypeToIr(const ConcreteType& concrete_type);

  absl::optional<SourceLocation> ToSourceLocation(
      const absl::optional<Span>& span) {
    if (!options_.emit_positions || !span.has_value()) {
      return absl::nullopt;
    }
    const Pos& start_pos = span->start();
    Lineno lineno(start_pos.lineno());
    Colno colno(start_pos.colno());
    // TODO(leary): 2020-12-20 Figure out the fileno based on the module owner
    // of node.
    return SourceLocation{fileno_, lineno, colno};
  }

  // Defines "node" to map the result of running "ir_func" with "args" -- if
  // emit_positions is on grabs the span from the node and uses it in the call.
  absl::StatusOr<BValue> DefWithStatus(
      AstNode* node, const std::function<absl::StatusOr<BValue>(
                         absl::optional<SourceLocation>)>& ir_func);

  // Specialization for Def() above when the "ir_func" is infallible.
  BValue Def(
      AstNode* node,
      const std::function<BValue(absl::optional<SourceLocation>)>& ir_func);

  // Def(), but for constant/constexpr values, adds a literal as the IR
  // function.
  CValue DefConst(AstNode* node, Value ir_value);

  absl::Status Visit(AstNode* node);

  // Creates a predicate that corresponds to reaching the current program point
  // being converted. Handlers that convert `match` constructs (and similar)
  // have a notion of implied control flow to the user -- we have to add control
  // predicates to side-effecting operations (like `fail!()`) that occur in
  // those positions.
  //
  // In other words, control constructs are responsible for squirreling away a
  // function that conjures the control predicate while they are being
  // translated (and this is how the conversion process can access that function
  // when needed).
  BValue CreateControlPredicate();

  Package* package() const { return package_data_.package; }

  // Package that IR is being generated into.
  PackageData& package_data_;

  // Module that contains the entry point function being converted.
  Module* module_;

  // The type information currently being used in IR conversion. Note that as we
  // visit derived parametrics, or traverse to imported modules, this will be
  // updated (in a stack-like fashion).
  TypeInfo* current_type_info_ = nullptr;

  // Import data which holds type information for imported modules.
  ImportData* import_data_;

  // Mapping from AST node to its corresponding IR value.
  absl::flat_hash_map<AstNode*, IrValue> node_to_ir_;

  // Various conversion options; e.g. whether or not to emit source code
  // positions into the XLS IR.
  ConvertOptions options_;

  // Constants that this translation depends upon (as determined externally).
  std::vector<ConstantDef*> constant_deps_;

  // Function builder being used to create BValues.
  absl::optional<FunctionBuilder> function_builder_;

  // When we have a fail!() operation we implicitly need to thread a token /
  // activation boolean -- the data for this is kept here if necessary.
  absl::optional<ImplicitTokenData> implicit_token_data_;

  // Mapping of symbolic bindings active in this translation (e.g. what integral
  // values parametrics are taking on).
  absl::flat_hash_map<std::string, InterpValue> symbolic_binding_map_;

  // File number for use in source positions.
  Fileno fileno_;

  // Number of "counted for" nodes we've observed in this function.
  int64_t counted_for_count_ = 0;
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
    parent_->implicit_token_data_->create_control_predicate = [&] {
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

// For all free variables of "node", adds them transitively for any required
// constant dependencies to the converter.
static absl::StatusOr<std::vector<ConstantDef*>> GetConstantDepFreevars(
    AstNode* node) {
  Span span = node->GetSpan().value();
  FreeVariables free_variables = node->GetFreeVariables(&span.start());
  std::vector<std::pair<std::string, AnyNameDef>> freevars =
      free_variables.GetNameDefTuples();
  std::vector<ConstantDef*> constant_deps;
  for (const auto& [identifier, any_name_def] : freevars) {
    if (absl::holds_alternative<BuiltinNameDef*>(any_name_def)) {
      continue;
    }
    auto* name_def = absl::get<NameDef*>(any_name_def);
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

struct ScopedTypeInfoSwap {
  ScopedTypeInfoSwap(FunctionConverter* converter, TypeInfo* new_type_info)
      : converter_(converter),
        original_type_info_(converter_->current_type_info_) {
    XLS_CHECK(new_type_info != nullptr);
    converter_->current_type_info_ = new_type_info;
  }

  ~ScopedTypeInfoSwap() {
    converter_->current_type_info_ = original_type_info_;
  }

  FunctionConverter* converter_;
  TypeInfo* original_type_info_;
};

// Helper that dispatches to the appropriate FunctionConverter handler for the
// AST node being visited.
class FunctionConverterVisitor : public AstNodeVisitor {
 public:
  explicit FunctionConverterVisitor(FunctionConverter* converter)
      : converter_(converter) {}

  // Causes node "n" to accept this visitor (basic double-dispatch).
  absl::Status Visit(AstNode* n) {
    XLS_VLOG(6) << this << " visiting: `" << n->ToString() << "` ("
                << n->GetNodeTypeName() << ")"
                << " @ " << SpanToString(n->GetSpan());
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
  // the FunctionConverter handler (i.e. postorder traversal).
#define TRAVERSE_DISPATCH(__type)                      \
  absl::Status Handle##__type(__type* node) override { \
    XLS_RETURN_IF_ERROR(VisitChildren(node));          \
    return converter_->Handle##__type(node);           \
  }

  TRAVERSE_DISPATCH(Unop)
  TRAVERSE_DISPATCH(Binop)
  TRAVERSE_DISPATCH(XlsTuple)

  // A macro used for AST types where we don't want to visit any children, just
  // call the FunctionConverter handler.
#define NO_TRAVERSE_DISPATCH(__type)                   \
  absl::Status Handle##__type(__type* node) override { \
    return converter_->Handle##__type(node);           \
  }

  NO_TRAVERSE_DISPATCH(Param)
  NO_TRAVERSE_DISPATCH(NameRef)
  NO_TRAVERSE_DISPATCH(ConstRef)
  NO_TRAVERSE_DISPATCH(Number)
  NO_TRAVERSE_DISPATCH(String)

  // A macro used for AST types where we don't want to visit any children, just
  // call the FunctionConverter handler.
#define NO_TRAVERSE_DISPATCH_VISIT(__type)             \
  absl::Status Handle##__type(__type* node) override { \
    return converter_->Handle##__type(node);           \
  }

  NO_TRAVERSE_DISPATCH_VISIT(Attr)
  NO_TRAVERSE_DISPATCH_VISIT(Array)
  NO_TRAVERSE_DISPATCH_VISIT(ConstantArray)
  NO_TRAVERSE_DISPATCH_VISIT(Cast)
  NO_TRAVERSE_DISPATCH_VISIT(ColonRef)
  NO_TRAVERSE_DISPATCH_VISIT(ConstantDef)
  NO_TRAVERSE_DISPATCH_VISIT(For)
  NO_TRAVERSE_DISPATCH_VISIT(Index)
  NO_TRAVERSE_DISPATCH_VISIT(Invocation)
  NO_TRAVERSE_DISPATCH_VISIT(Let)
  NO_TRAVERSE_DISPATCH_VISIT(Match)
  NO_TRAVERSE_DISPATCH_VISIT(SplatStructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(StructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(Ternary)

  // A macro used for AST types that we never expect to visit (if we do we
  // provide an error message noting it was unexpected).
#define INVALID(__type) \
  absl::Status Handle##__type(__type* node) override { return Invalid(node); }

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
  FunctionConverter* converter_;
};

absl::Status FunctionConverter::Visit(AstNode* node) {
  FunctionConverterVisitor visitor(this);
  return visitor.Visit(node);
}

/* static */ std::string FunctionConverter::ToString(const IrValue& value) {
  if (absl::holds_alternative<BValue>(value)) {
    return absl::StrFormat("%p", absl::get<BValue>(value).node());
  }
  return absl::StrFormat("%p", absl::get<CValue>(value).value.node());
}

FunctionConverter::FunctionConverter(PackageData& package_data, Module* module,
                                     ImportData* import_data,
                                     ConvertOptions options)
    : package_data_(package_data),
      module_(module),
      import_data_(import_data),
      options_(std::move(options)),
      // TODO(leary): 2019-07-19 Create a way to get the file path from the
      // module.
      fileno_(package_data.package->GetOrCreateFileno("fake_file.x")) {
  XLS_VLOG(5) << "Constructed IR converter: " << this;
}

void FunctionConverter::InstantiateFunctionBuilder(
    absl::string_view mangled_name) {
  XLS_CHECK(!function_builder_.has_value());
  function_builder_.emplace(mangled_name, package());
}

void FunctionConverter::AddConstantDep(ConstantDef* constant_def) {
  XLS_VLOG(2) << "Adding constant dep: " << constant_def->ToString();
  constant_deps_.push_back(constant_def);
}

absl::StatusOr<BValue> FunctionConverter::DefAlias(AstNode* from, AstNode* to) {
  XLS_RET_CHECK_NE(from, to);
  auto it = node_to_ir_.find(from);
  if (it == node_to_ir_.end()) {
    return absl::InternalError(absl::StrFormat(
        "TypeAliasError: %s internal error during IR conversion: could not "
        "find AST node for "
        "aliasing: %s (%s) to: %s (%s)",
        SpanToString(from->GetSpan()), from->ToString(),
        from->GetNodeTypeName(), to->ToString(), to->GetNodeTypeName()));
  }
  IrValue value = it->second;
  XLS_VLOG(6) << absl::StreamFormat(
      "Aliased node '%s' (%s) to be same as '%s' (%s): %s", to->ToString(),
      to->GetNodeTypeName(), from->ToString(), from->GetNodeTypeName(),
      ToString(value));
  node_to_ir_[to] = std::move(value);
  if (auto* name_def = dynamic_cast<NameDef*>(to)) {
    // Name the aliased node based on the identifier in the NameDef.
    if (absl::holds_alternative<BValue>(node_to_ir_.at(from))) {
      BValue ir_node = absl::get<BValue>(node_to_ir_.at(from));
      ir_node.SetName(name_def->identifier());
    } else {
      BValue ir_node = absl::get<CValue>(node_to_ir_.at(from)).value;
      ir_node.SetName(name_def->identifier());
    }
  }
  return Use(to);
}

absl::StatusOr<BValue> FunctionConverter::DefWithStatus(
    AstNode* node,
    const std::function<absl::StatusOr<BValue>(absl::optional<SourceLocation>)>&
        ir_func) {
  absl::optional<SourceLocation> loc = ToSourceLocation(node->GetSpan());
  XLS_ASSIGN_OR_RETURN(BValue result, ir_func(loc));
  XLS_VLOG(6) << absl::StreamFormat(
      "Define node '%s' (%s) to be %s @ %s", node->ToString(),
      node->GetNodeTypeName(), ToString(result), SpanToString(node->GetSpan()));
  SetNodeToIr(node, result);
  return result;
}

BValue FunctionConverter::Def(
    AstNode* node,
    const std::function<BValue(absl::optional<SourceLocation>)>& ir_func) {
  return DefWithStatus(node,
                       [&ir_func](absl::optional<SourceLocation> loc)
                           -> absl::StatusOr<BValue> { return ir_func(loc); })
      .value();
}

FunctionConverter::CValue FunctionConverter::DefConst(AstNode* node,
                                                      xls::Value ir_value) {
  auto ir_func = [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Literal(ir_value, loc);
  };
  BValue result = Def(node, ir_func);
  CValue c_value{ir_value, result};
  SetNodeToIr(node, c_value);
  return c_value;
}

absl::StatusOr<BValue> FunctionConverter::Use(AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Could not resolve IR value for %s node: %s",
                        node->GetNodeTypeName(), node->ToString()));
  }
  const IrValue& ir_value = it->second;
  XLS_VLOG(6) << absl::StreamFormat("Using node '%s' (%p) as IR value %s.",
                                    node->ToString(), node, ToString(ir_value));
  if (absl::holds_alternative<BValue>(ir_value)) {
    return absl::get<BValue>(ir_value);
  }
  XLS_RET_CHECK(absl::holds_alternative<CValue>(ir_value));
  return absl::get<CValue>(ir_value).value;
}

void FunctionConverter::SetNodeToIr(AstNode* node, IrValue value) {
  XLS_VLOG(6) << absl::StreamFormat("Setting node '%s' (%p) to IR value %s.",
                                    node->ToString(), node, ToString(value));
  node_to_ir_[node] = value;
}

absl::optional<FunctionConverter::IrValue> FunctionConverter::GetNodeToIr(
    AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::nullopt;
  }
  return it->second;
}

absl::Status FunctionConverter::HandleUnop(Unop* node) {
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
      absl::StrCat("Invalid UnopKind: ", static_cast<int64_t>(node->kind())));
}

absl::Status FunctionConverter::HandleConcat(Binop* node, BValue lhs,
                                             BValue rhs) {
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

absl::StatusOr<std::vector<BValue>> FunctionConverter::BuildTerms(
    BValue lhs, BValue rhs, absl::optional<SourceLocation>& loc,
    const FunctionConverter::BuildTermFn& build_term) {
  // Since we're building the expanded IR here, it is simpler to check that
  // the types match once at the beginning rather incrementally at each
  // expansion step.
  XLS_RET_CHECK(lhs.GetType() == rhs.GetType()) << absl::StreamFormat(
      "BuildTerms lhs %s and rhs %s types do not match in when building terms",
      lhs.GetType()->ToString(), rhs.GetType()->ToString());

  struct ToBuild {
    xls::BValue lhs;
    xls::BValue rhs;
  };

  std::vector<ToBuild> to_build = {ToBuild{lhs, rhs}};
  std::vector<BValue> result;

  while (!to_build.empty()) {
    ToBuild next = to_build.back();
    to_build.pop_back();

    xls::Type* term_type = next.lhs.GetType();

    switch (term_type->kind()) {
      case TypeKind::kToken:
        return absl::InvalidArgumentError(
            absl::StrFormat("Illegal token comparison lhs %s rhs %s",
                            lhs.ToString(), rhs.ToString()));
      case TypeKind::kBits:
        result.push_back(build_term(next.lhs, next.rhs));
        break;
      case TypeKind::kArray: {
        xls::ArrayType* array_type = term_type->AsArrayOrDie();
        // Cast the array size to uint64_t because it will be used as an
        // unsigned index in the generated IR.
        uint64_t array_size = array_type->size();
        for (uint64_t i = 0; i < array_size; i++) {
          BValue i_val = function_builder_->Literal(Value(UBits(i, 64)));
          BValue lhs_i = function_builder_->ArrayIndex(next.lhs, {i_val}, loc);
          BValue rhs_i = function_builder_->ArrayIndex(next.rhs, {i_val}, loc);
          to_build.push_back(ToBuild{lhs_i, rhs_i});
        }
        break;
      }
      case TypeKind::kTuple: {
        xls::TupleType* tuple_type = term_type->AsTupleOrDie();
        int64_t tuple_size = tuple_type->size();
        for (int64_t i = 0; i < tuple_size; i++) {
          BValue lhs_i = function_builder_->TupleIndex(next.lhs, i, loc);
          BValue rhs_i = function_builder_->TupleIndex(next.rhs, i, loc);
          to_build.push_back(ToBuild{lhs_i, rhs_i});
        }
        break;
      }
    }
  }
  return result;
}

absl::StatusOr<BValue> FunctionConverter::BuildTest(
    BValue lhs, BValue rhs, Op op, Value base,
    absl::optional<SourceLocation>& loc, const BuildTermFn& build_term) {
  XLS_ASSIGN_OR_RETURN(std::vector<BValue> terms,
                       BuildTerms(lhs, rhs, loc, build_term));
  if (terms.empty()) {
    return function_builder_->Literal(base);
  }
  if (terms.size() == 1) {
    return terms.front();
  } else {
    return function_builder_->AddNaryOp(op, terms, loc);
  }
}

absl::Status FunctionConverter::HandleEq(Binop* node, BValue lhs, BValue rhs) {
  return DefWithStatus(
             node,
             [&](absl::optional<SourceLocation> loc) -> absl::StatusOr<BValue> {
               return BuildTest(lhs, rhs, Op::kAnd, Value::Bool(true), loc,
                                [this, loc](BValue l, BValue r) {
                                  return function_builder_->Eq(l, r, loc);
                                });
             })
      .status();
}

absl::Status FunctionConverter::HandleNe(Binop* node, BValue lhs, BValue rhs) {
  return DefWithStatus(
             node,
             [&](absl::optional<SourceLocation> loc) -> absl::StatusOr<BValue> {
               return BuildTest(lhs, rhs, Op::kOr, Value::Bool(false), loc,
                                [this, loc](BValue l, BValue r) {
                                  return function_builder_->Ne(l, r, loc);
                                });
             })
      .status();
}

SymbolicBindings FunctionConverter::GetSymbolicBindingsTuple() const {
  absl::flat_hash_set<std::string> module_level_constant_identifiers;
  for (const ConstantDef* constant : module_->GetConstantDefs()) {
    module_level_constant_identifiers.insert(constant->identifier());
  }
  absl::flat_hash_map<std::string, InterpValue> sans_module_level_constants;
  for (const auto& item : symbolic_binding_map_) {
    if (module_level_constant_identifiers.contains(item.first)) {
      continue;
    }
    sans_module_level_constants.insert({item.first, item.second});
  }
  return SymbolicBindings(std::move(sans_module_level_constants));
}

absl::Status FunctionConverter::HandleNumber(Number* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim dim, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                       absl::get<InterpValue>(dim.value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, node->GetBits(bit_count));
  DefConst(node, Value(bits));
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleString(String* node) {
  std::vector<Value> elements;
  for (const char letter : node->text()) {
    elements.push_back(Value(UBits(letter, /*bit_count=*/8)));
  }
  XLS_ASSIGN_OR_RETURN(Value array, Value::Array(elements));
  DefConst(node, array);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleXlsTuple(XlsTuple* node) {
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

absl::Status FunctionConverter::HandleParam(Param* node) {
  XLS_VLOG(5) << "HandleParam: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(xls::Type * type,
                       ResolveTypeToIr(node->type_annotation()));
  Def(node->name_def(), [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Param(node->identifier(), type);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleConstRef(ConstRef* node) {
  return DefAlias(node->name_def(), /*to=*/node).status();
}

absl::Status FunctionConverter::HandleNameRef(NameRef* node) {
  AstNode* from = ToAstNode(node->name_def());
  return DefAlias(from, /*to=*/node).status();
}

absl::Status FunctionConverter::HandleConstantDef(ConstantDef* node) {
  XLS_VLOG(5) << "Visiting ConstantDef expr: " << node->value()->ToString();
  XLS_RETURN_IF_ERROR(Visit(node->value()));
  XLS_VLOG(5) << "Aliasing NameDef for constant: "
              << node->name_def()->ToString();
  return DefAlias(node->value(), /*to=*/node->name_def()).status();
}

absl::Status FunctionConverter::HandleLet(Let* node) {
  XLS_RETURN_IF_ERROR(Visit(node->rhs()));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->rhs()));

  // Verify that the RHS conforms to the annotation (if present).
  if (node->type_annotation() != nullptr) {
    XLS_ASSIGN_OR_RETURN(xls::Type * annotated_type,
                         ResolveTypeToIr(node->type_annotation()));
    xls::Type* value_type = rhs.GetType();
    XLS_RET_CHECK_EQ(annotated_type, value_type);
  }

  if (node->name_def_tree()->is_leaf()) {
    XLS_RETURN_IF_ERROR(
        DefAlias(node->rhs(), /*to=*/ToAstNode(node->name_def_tree()->leaf()))
            .status());
    XLS_RETURN_IF_ERROR(Visit(node->body()));
    XLS_RETURN_IF_ERROR(DefAlias(node->body(), node).status());
  } else {
    // Walk the tree of names we're trying to bind, performing tuple_index
    // operations on the RHS to get to the values we want to bind to those
    // names.
    std::vector<BValue> levels = {rhs};
    // Invoked at each level of the NameDefTree: binds the name in the
    // NameDefTree to the correponding value (being pattern matched).
    //
    // Args:
    //  x: Current subtree of the NameDefTree.
    //  level: Level (depth) in the NameDefTree, root is 0.
    //  index: Index of node in the current tree level (e.g. leftmost is 0).
    auto walk = [&](NameDefTree* x, int64_t level,
                    int64_t index) -> absl::Status {
      XLS_VLOG(6) << absl::StreamFormat("Walking level %d index %d: `%s`",
                                        level, index, x->ToString());
      levels.resize(level);
      levels.push_back(Def(x, [this, &levels, x,
                               index](absl::optional<SourceLocation> loc) {
        if (loc.has_value()) {
          loc = ToSourceLocation(x->is_leaf() ? ToAstNode(x->leaf())->GetSpan()
                                              : x->GetSpan());
        }
        BValue tuple = levels.back();
        xls::TupleType* tuple_type = tuple.GetType()->AsTupleOrDie();
        XLS_CHECK_LT(index, tuple_type->size())
            << "index: " << index << " type: " << tuple_type->ToString();
        return function_builder_->TupleIndex(tuple, index, loc);
      }));
      if (x->is_leaf()) {
        XLS_RETURN_IF_ERROR(DefAlias(x, ToAstNode(x->leaf())).status());
      }
      return absl::OkStatus();
    };

    XLS_RETURN_IF_ERROR(node->name_def_tree()->DoPreorder(walk));
    XLS_RETURN_IF_ERROR(Visit(node->body()));
    XLS_RETURN_IF_ERROR(DefAlias(node->body(), /*to=*/node).status());
  }

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleCast(Cast* node) {
  XLS_RETURN_IF_ERROR(Visit(node->expr()));
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
  XLS_ASSIGN_OR_RETURN(
      int64_t new_bit_count,
      absl::get<InterpValue>(new_bit_count_ctd.value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim input_bit_count_ctd,
                       input_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t old_bit_count,
      absl::get<InterpValue>(input_bit_count_ctd.value()).GetBitValueInt64());
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

absl::Status FunctionConverter::HandleMatch(Match* node) {
  if (node->arms().empty() ||
      !node->arms().back()->patterns()[0]->IsIrrefutable()) {
    return ConversionErrorStatus(
        node->span(),
        "Only matches with trailing irrefutable patterns (i.e. `_ => ...`) "
        "are currently supported for IR conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->matched()));
  XLS_ASSIGN_OR_RETURN(BValue matched, Use(node->matched()));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> matched_type,
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
    return absl::UnimplementedError(
        absl::StrFormat("ConversionError: %s Multiple patterns in default arm "
                        "is not currently supported for IR conversion.",
                        node->span().ToString()));
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
  SetNodeToIr(node, function_builder_->MatchTrue(arm_selectors, arm_values,
                                                 default_value));
  return absl::OkStatus();
}

absl::StatusOr<int64_t> FunctionConverter::QueryConstRangeCall(For* node) {
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

  absl::optional<InterpValue> interp_value =
      current_type_info_->GetConstExpr(start);
  if (!interp_value || !interp_value->IsBits() ||
      interp_value->GetBitValueUint64().value() != 0) {
    return error();
  }

  absl::optional<InterpValue> value = current_type_info_->GetConstExpr(limit);
  XLS_RET_CHECK(value.has_value() && value->IsBits());
  if (value->IsSigned()) {
    return value->GetBitValueInt64();
  } else {
    return value->GetBitValueUint64();
  }
}

// Convert a NameDefTree node variant to an AstNode pointer (either the leaf
// node or the interior NameDefTree node).
static AstNode* ToAstNode(
    const absl::variant<NameDefTree::Leaf, NameDefTree*>& x) {
  if (absl::holds_alternative<NameDefTree*>(x)) {
    return absl::get<NameDefTree*>(x);
  }
  return ToAstNode(absl::get<NameDefTree::Leaf>(x));
}

absl::Status FunctionConverter::HandleFor(For* node) {
  XLS_RETURN_IF_ERROR(Visit(node->init()));

  // TODO(leary): We currently only support counted loops with fixed upper
  // bounds that start at zero; i.e. those of the form like:
  //
  //  for (i, ...): (u32, ...) in range(u32:0, N) {
  //    ...
  //  }
  XLS_ASSIGN_OR_RETURN(int64_t trip_count, QueryConstRangeCall(node));

  XLS_VLOG(5) << "Converting for-loop @ " << node->span();
  FunctionConverter body_converter(package_data_, module_, import_data_,
                                   options_);
  body_converter.set_symbolic_binding_map(symbolic_binding_map_);

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
  body_converter.InstantiateFunctionBuilder(body_fn_name);

  // Grab the two tuple of `(ivar, accum)`.
  std::vector<absl::variant<NameDefTree::Leaf, NameDefTree*>> flat =
      node->names()->Flatten1();
  if (flat.size() != 2) {
    return absl::UnimplementedError(
        "Expect for loop to have counter (induction variable) and carry data "
        "for IR conversion.");
  }

  // Add the induction value (the "ranged" counter).
  AstNode* ivar = ToAstNode(flat[0]);
  auto* name_def = dynamic_cast<NameDef*>(ivar);
  XLS_RET_CHECK(name_def != nullptr);
  XLS_ASSIGN_OR_RETURN(xls::Type * ivar_type, ResolveTypeToIr(name_def));
  body_converter.SetNodeToIr(name_def, body_converter.function_builder_->Param(
                                           name_def->identifier(), ivar_type));

  // Add the loop carry value.
  AstNode* carry = ToAstNode(flat[1]);
  if (auto* name_def = dynamic_cast<NameDef*>(carry)) {
    // When the loop carry value is just a name; e.g. the `x` in `for (i, x)` we
    // can simply bind it.
    XLS_ASSIGN_OR_RETURN(xls::Type * type, ResolveTypeToIr(name_def));
    if (implicit_token_data_.has_value()) {
      // Note: this is somewhat conservative, even if the for-body does not
      // require an implicit token, for bodies are not marked independently from
      // their enclosing functions.
      BValue unwrapped = body_converter.AddTokenWrappedParam(type);
      body_converter.SetNodeToIr(name_def, unwrapped);
    } else {
      BValue param =
          body_converter.function_builder_->Param(name_def->identifier(), type);
      body_converter.SetNodeToIr(name_def, param);
    }
  } else {
    // For tuple loop carries we have to destructure names on entry.
    NameDefTree* accum = absl::get<NameDefTree*>(flat[1]);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> carry_type,
                         ResolveType(accum));
    XLS_ASSIGN_OR_RETURN(xls::Type * carry_ir_type, TypeToIr(*carry_type));
    BValue carry;
    if (implicit_token_data_.has_value()) {
      carry = body_converter.AddTokenWrappedParam(carry_ir_type);
    } else {
      carry = body_converter.function_builder_->Param("__loop_carry",
                                                      carry_ir_type);
    }
    body_converter.SetNodeToIr(accum, carry);
    // This will destructure the names for us in the body of the anonymous
    // function.
    XLS_RETURN_IF_ERROR(body_converter
                            .HandleMatcher(/*matcher=*/accum, /*index=*/{},
                                           /*matched_value=*/carry,
                                           /*matched_type=*/*carry_type)
                            .status());
  }

  // We need to capture the lexical scope and pass it to his loop body function.
  //
  // So we suffix free variables for the function body onto the function
  // parameters.
  FreeVariables freevars =
      node->body()->GetFreeVariables(&node->span().start());
  freevars = freevars.DropBuiltinDefs();
  std::vector<NameDef*> relevant_name_defs;
  for (const auto& any_name_def : freevars.GetNameDefs()) {
    auto* name_def = absl::get<NameDef*>(any_name_def);
    absl::optional<const ConcreteType*> type =
        current_type_info_->GetItem(name_def);
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
    XLS_VLOG(5) << "Converting freevar name: " << name_def->ToString();

    absl::optional<IrValue> ir_value = GetNodeToIr(name_def);
    if (!ir_value.has_value()) {
      return absl::InternalError(
          absl::StrFormat("AST node had no associated IR value: %s @ %s",
                          node->ToString(), SpanToString(node->GetSpan())));
    }

    // If free variable is a constant, create constant node inside body.
    // This preserves const-ness of loop body uses (e.g. loop bounds for
    // a nested loop).
    if (absl::holds_alternative<CValue>(*ir_value)) {
      Value constant_value = absl::get<CValue>(*ir_value).ir_value;
      body_converter.DefConst(name_def, constant_value);
    } else {
      // Otherwise, pass in the variable to the loop body function as
      // a parameter.
      relevant_name_defs.push_back(name_def);
      XLS_ASSIGN_OR_RETURN(xls::Type * name_def_type, TypeToIr(**type));
      body_converter.SetNodeToIr(name_def,
                                 body_converter.function_builder_->Param(
                                     name_def->identifier(), name_def_type));
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
    XLS_ASSIGN_OR_RETURN(BValue retval,
                         body_converter.function_builder_->GetLastValue());
    body_converter.function_builder_->Tuple(
        {body_converter.implicit_token_data_->entry_token,
         body_converter.implicit_token_data_->activated, retval});
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * body_function,
                       body_converter.function_builder_->Build());
  XLS_VLOG(5) << "Converted body function: " << body_function->name();

  std::vector<BValue> invariant_args;
  for (NameDef* name_def : relevant_name_defs) {
    XLS_ASSIGN_OR_RETURN(BValue value, Use(name_def));
    invariant_args.push_back(value);
  }

  XLS_ASSIGN_OR_RETURN(BValue init, Use(node->init()));
  if (implicit_token_data_.has_value()) {
    BValue activated = trip_count == 0 ? function_builder_->Literal(UBits(0, 1))
                                       : implicit_token_data_->activated;
    init = function_builder_->Tuple(
        {implicit_token_data_->entry_token, activated, init});
  }

  Def(node, [&](absl::optional<SourceLocation> loc) {
    BValue result = function_builder_->CountedFor(
        init, trip_count, /*stride=*/1, body_function, invariant_args);
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
    const BValue& matched_value, const ConcreteType& matched_type) {
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
      XLS_RETURN_IF_ERROR(Visit(ToAstNode(leaf)));
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
  for (int64_t i = 0; i < matched_tuple_type->size(); ++i) {
    const ConcreteType& element_type = matched_tuple_type->GetMemberType(i);
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
    Invocation* parent_node, NameRef* node, AstNode* arg,
    const SymbolicBindings& symbolic_bindings) {
  // Builtins always use the "typical" calling convention (are never "implicit
  // token").
  XLS_ASSIGN_OR_RETURN(const std::string mangled_name,
                       MangleDslxName(module_->name(), node->identifier(),
                                      CallingConvention::kTypical,
                                      /*free_keys=*/{}, &symbolic_bindings));
  XLS_ASSIGN_OR_RETURN(BValue arg_value, Use(arg));
  XLS_VLOG(5) << "Mapping with builtin; arg: "
              << arg_value.GetType()->ToString();
  auto* array_type = arg_value.GetType()->AsArrayOrDie();
  if (!package()->HasFunctionWithName(mangled_name)) {
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
    XLS_RETURN_IF_ERROR(fb.Build().status());
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * f, package()->GetFunction(mangled_name));
  return Def(parent_node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Map(arg_value, f);
  });
}

absl::StatusOr<BValue> FunctionConverter::HandleMap(Invocation* node) {
  for (Expr* arg : node->args().subspan(0, node->args().size() - 1)) {
    XLS_RETURN_IF_ERROR(Visit(arg));
  }
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Expr* fn_node = node->args()[1];
  XLS_VLOG(5) << "Function being mapped AST: " << fn_node->ToString();
  absl::optional<const SymbolicBindings*> node_sym_bindings =
      GetInstantiationCalleeBindings(node);

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
    Import* import_node = colon_ref->ResolveImportSubject().value();
    absl::optional<const ImportedInfo*> info =
        current_type_info_->GetImported(import_node);
    lookup_module = (*info)->module;
  } else {
    return absl::UnimplementedError("Unhandled function mapping: " +
                                    fn_node->ToString());
  }

  absl::optional<Function*> mapped_fn = lookup_module->GetFunction(map_fn_name);
  std::vector<std::string> free = (*mapped_fn)->GetFreeParametricKeys();
  absl::btree_set<std::string> free_set(free.begin(), free.end());
  CallingConvention convention = GetCallingConvention(mapped_fn.value());
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(lookup_module->name(), (*mapped_fn)->identifier(),
                     convention, free_set, node_sym_bindings.value()));
  XLS_VLOG(5) << "Getting function with mangled name: " << mangled_name
              << " from package: " << package()->name();
  XLS_ASSIGN_OR_RETURN(xls::Function * f, package()->GetFunction(mangled_name));
  return Def(node, [&](absl::optional<SourceLocation> loc) -> BValue {
    return function_builder_->Map(arg, f, loc);
  });
}

absl::Status FunctionConverter::HandleIndex(Index* node) {
  XLS_RETURN_IF_ERROR(Visit(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));

  absl::optional<const ConcreteType*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  if (dynamic_cast<const TupleType*>(lhs_type.value()) != nullptr) {
    // Tuple indexing requires a compile-time-constant RHS.
    XLS_RETURN_IF_ERROR(Visit(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(Bits rhs, GetConstBits(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(uint64_t index, rhs.ToUint64());
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->TupleIndex(lhs, index, loc);
    });
  } else if (dynamic_cast<const BitsType*>(lhs_type.value()) != nullptr) {
    IndexRhs rhs = node->rhs();
    if (absl::holds_alternative<WidthSlice*>(rhs)) {
      auto* width_slice = absl::get<WidthSlice*>(rhs);
      XLS_RETURN_IF_ERROR(Visit(width_slice->start()));
      XLS_ASSIGN_OR_RETURN(BValue start, Use(width_slice->start()));
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> output_type,
                           ResolveType(node));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim output_type_dim,
                           output_type->GetTotalBitCount());
      XLS_ASSIGN_OR_RETURN(int64_t width, output_type_dim.GetAsInt64());
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->DynamicBitSlice(lhs, start, width, loc);
      });
    } else {
      auto* slice = absl::get<Slice*>(rhs);
      absl::optional<StartAndWidth> saw =
          current_type_info_->GetSliceStartAndWidth(slice,
                                                    GetSymbolicBindingsTuple());
      XLS_RET_CHECK(saw.has_value());
      Def(node, [&](absl::optional<SourceLocation> loc) {
        return function_builder_->BitSlice(lhs, saw->start, saw->width, loc);
      });
    }
  } else {
    XLS_RETURN_IF_ERROR(Visit(ToAstNode(node->rhs())));
    XLS_ASSIGN_OR_RETURN(BValue index, Use(ToAstNode(node->rhs())));
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->ArrayIndex(lhs, {index}, loc);
    });
  }
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleArray(Array* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  const ArrayType* array_type = dynamic_cast<ArrayType*>(type.get());
  XLS_RET_CHECK(array_type != nullptr);
  std::vector<BValue> members;
  for (Expr* member : node->members()) {
    XLS_RETURN_IF_ERROR(Visit(member));
    XLS_ASSIGN_OR_RETURN(BValue member_value, Use(member));
    members.push_back(member_value);
  }

  if (node->has_ellipsis()) {
    ConcreteTypeDim array_size_ctd = array_type->size();
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_size_ctd.GetAsInt64());
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

absl::Status FunctionConverter::HandleUdfInvocation(Invocation* node,
                                                    xls::Function* f,
                                                    std::vector<BValue> args) {
  XLS_VLOG(5) << "HandleUdfInvocation: " << f->name() << " via "
              << node->ToString();
  XLS_RET_CHECK(package_data_.ir_to_dslx.contains(f)) << f->name();
  dslx::Function* dslx_callee = package_data_.ir_to_dslx.at(f);

  const bool callee_requires_implicit_token =
      GetRequiresImplicitToken(dslx_callee);
  XLS_VLOG(6) << "HandleUdfInvocation: callee: " << dslx_callee->ToString()
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

  Def(node, [&](absl::optional<SourceLocation> loc) {
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

absl::Status FunctionConverter::HandleFailBuiltin(Invocation* node,
                                                  BValue arg) {
  if (options_.emit_fail_as_assert) {
    // For a fail node we both create a predicate that corresponds to the
    // "control" leading to this DSL program point.
    XLS_RET_CHECK(implicit_token_data_.has_value())
        << "Invoking fail!(), but no implicit token is present for caller @ "
        << node->span();
    XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
    BValue control_predicate = implicit_token_data_->create_control_predicate();
    std::string message = absl::StrFormat("Assertion failure via fail! @ %s",
                                          node->span().ToString());
    BValue assert_result_token = function_builder_->Assert(
        implicit_token_data_->entry_token,
        function_builder_->Not(control_predicate), message);
    implicit_token_data_->control_tokens.push_back(assert_result_token);
  }
  // The result of the failure call is the argument given; e.g. if we were to
  // remove assertions this is the value that would flow in the case that the
  // assertion was hit.
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Identity(arg);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleCoverBuiltin(Invocation* node,
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
    XLS_RET_CHECK_EQ(node->args().size(), 2);
    String* label = dynamic_cast<String*>(node->args()[0]);
    XLS_RET_CHECK(label != nullptr)
        << "cover!() argument 0 must be a literal string "
        << "(should have been typechecked?).";
    BValue cover_result_token = function_builder_->Cover(
        implicit_token_data_->entry_token, condition, label->text());
    implicit_token_data_->control_tokens.push_back(cover_result_token);
  }

  // The result of the cover call is the argument given; e.g. if we were to
  // turn off coverpoints, this is the value that would be used.
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Tuple(std::vector<BValue>());
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleInvocation(Invocation* node) {
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

  if (package()->HasFunctionWithName(called_name)) {
    XLS_ASSIGN_OR_RETURN(xls::Function * f,
                         package()->GetFunction(called_name));
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    return HandleUdfInvocation(node, f, std::move(args));
  }

  // A few builtins are handled specially.

  if (called_name == "fail!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 1)
        << called_name << " builtin only accepts a single argument";
    return HandleFailBuiltin(node, std::move(args[0]));
  } else if (called_name == "cover!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 2)
        << called_name << " builtin requires two arguments";
    return HandleCoverBuiltin(node, std::move(args[1]));
  } else if (called_name == "trace!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 1)
        << called_name << " builtin only accepts a single argument";
    Def(node, [&](absl::optional<SourceLocation> loc) {
      return function_builder_->Identity(args[0], loc);
    });
    return absl::OkStatus();
  } else if (called_name == "map") {
    return HandleMap(node).status();
  }

  // The rest of the builtins have "handle" methods we can resolve.
  absl::flat_hash_map<std::string,
                      decltype(&FunctionConverter::HandleBuiltinClz)>
      map = {
          {"clz", &FunctionConverter::HandleBuiltinClz},
          {"ctz", &FunctionConverter::HandleBuiltinCtz},
          {"sgt", &FunctionConverter::HandleBuiltinSGt},
          {"sge", &FunctionConverter::HandleBuiltinSGe},
          {"slt", &FunctionConverter::HandleBuiltinSLt},
          {"sle", &FunctionConverter::HandleBuiltinSLe},
          {"signex", &FunctionConverter::HandleBuiltinSignex},
          {"one_hot", &FunctionConverter::HandleBuiltinOneHot},
          {"one_hot_sel", &FunctionConverter::HandleBuiltinOneHotSel},
          {"slice", &FunctionConverter::HandleBuiltinArraySlice},
          {"bit_slice", &FunctionConverter::HandleBuiltinBitSlice},
          {"bit_slice_update", &FunctionConverter::HandleBuiltinBitSliceUpdate},
          {"rev", &FunctionConverter::HandleBuiltinRev},
          {"and_reduce", &FunctionConverter::HandleBuiltinAndReduce},
          {"or_reduce", &FunctionConverter::HandleBuiltinOrReduce},
          {"xor_reduce", &FunctionConverter::HandleBuiltinXorReduce},
          {"update", &FunctionConverter::HandleBuiltinUpdate},
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
  return absl::OkStatus();
}

// Creates a function that wraps up `implicit_token_f`.
//
// Precondition: `implicit_token_f` must use the "implicit token" calling
// convention, see `CallingConvention` for details.
//
// The wrapped function exposes the implicit token function as if it were a
// normal function, so it can be called by the outside world in a typical
// fashion as an entry point (e.g. the IR JIT, Verilog module signature, etc).
static absl::StatusOr<xls::Function*> EmitImplicitTokenEntryWrapper(
    xls::Function* implicit_token_f, dslx::Function* dslx_function) {
  XLS_RET_CHECK_GE(implicit_token_f->params().size(), 2);
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(dslx_function->owner()->name(),
                     dslx_function->identifier(), CallingConvention::kTypical,
                     /*free_keys=*/{}, /*symbolic_bindings=*/nullptr));
  FunctionBuilder fb(mangled_name, implicit_token_f->package());

  // Clone all the params except for the leading `(token, bool)`.
  std::vector<BValue> params;
  for (const xls::Param* p : implicit_token_f->params().subspan(2)) {
    params.push_back(fb.Param(p->name(), p->GetType()));
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

// As a postprocessing step for converting a module to a package, we check and
// see if the entry point has the "implicit token" calling convention, to see if
// it should be wrapped up.
//
// Note: we do this as a postprocessing step because we can't know what the
// module entry point is _until_ all functions have been converted.
static absl::Status WrapEntryIfImplicitToken(const PackageData& package_data,
                                             ImportData* import_data,
                                             const ConvertOptions& options) {
  absl::StatusOr<xls::Function*> entry_or =
      package_data.package->EntryFunction();
  if (!entry_or.ok()) {  // Entry point not found.
    XLS_RET_CHECK_EQ(entry_or.status().code(), absl::StatusCode::kNotFound);
    return absl::OkStatus();
  }

  xls::Function* entry = entry_or.value();
  if (package_data.wrappers.contains(entry)) {
    // Already created!
    return absl::OkStatus();
  }

  dslx::Function* dslx_entry = package_data.ir_to_dslx.at(entry);
  if (GetRequiresImplicitToken(dslx_entry, import_data, options)) {
    return EmitImplicitTokenEntryWrapper(entry, dslx_entry).status();
  }
  return absl::OkStatus();
}

absl::StatusOr<xls::Function*> FunctionConverter::HandleFunction(
    Function* node, TypeInfo* type_info,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RET_CHECK(type_info != nullptr);

  XLS_VLOG(5) << "HandleFunction: " << node->ToString();

  if (symbolic_bindings != nullptr) {
    SetSymbolicBindings(symbolic_bindings);
  }

  ScopedTypeInfoSwap stis(this, type_info);

  // We use a function builder for the duration of converting this AST Function.
  const bool requires_implicit_token = GetRequiresImplicitToken(node);
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(module_->name(), node->identifier(),
                     requires_implicit_token ? CallingConvention::kImplicitToken
                                             : CallingConvention::kTypical,
                     node->GetFreeParametricKeySet(), symbolic_bindings));
  InstantiateFunctionBuilder(mangled_name);

  XLS_VLOG(6) << "Function " << node->identifier()
              << " requires_implicit_token? "
              << (requires_implicit_token ? "true" : "false");
  if (requires_implicit_token) {
    XLS_RETURN_IF_ERROR(AddImplicitTokenParams());
    XLS_RET_CHECK(implicit_token_data_.has_value());
  }

  for (Param* param : node->params()) {
    XLS_RETURN_IF_ERROR(Visit(param));
  }

  for (ParametricBinding* parametric_binding : node->parametric_bindings()) {
    XLS_VLOG(5) << "Resolving parametric binding: "
                << parametric_binding->ToString();

    absl::optional<InterpValue> sb_value =
        get_symbolic_binding(parametric_binding->identifier());
    XLS_RET_CHECK(sb_value.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_type,
                         ResolveType(parametric_binding->type_annotation()));
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim parametric_width_ctd,
                         parametric_type->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, parametric_width_ctd.GetAsInt64());
    Value param_value;
    if (sb_value->IsSigned()) {
      XLS_ASSIGN_OR_RETURN(int64_t bit_value, sb_value->GetBitValueInt64());
      param_value = Value(SBits(bit_value, bit_count));
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t bit_value, sb_value->GetBitValueUint64());
      param_value = Value(UBits(bit_value, bit_count));
    }
    DefConst(parametric_binding, param_value);
    XLS_RETURN_IF_ERROR(
        DefAlias(parametric_binding, /*to=*/parametric_binding->name_def())
            .status());
  }

  XLS_VLOG(3) << "Function has " << constant_deps_.size() << " constant deps";
  for (ConstantDef* dep : constant_deps_) {
    XLS_VLOG(5) << "Visiting constant dep: " << dep->ToString();
    XLS_RETURN_IF_ERROR(Visit(dep));
  }

  XLS_VLOG(5) << "body: " << node->body()->ToString();
  XLS_RETURN_IF_ERROR(Visit(node->body()));

  XLS_ASSIGN_OR_RETURN(BValue return_value, Use(node->body()));

  if (requires_implicit_token) {
    // Now join all the assertion tokens together to make the output token.
    XLS_RET_CHECK(!implicit_token_data_->control_tokens.empty())
        << "Function " << node->ToString()
        << " has no assertion tokens to join!";
    BValue join_token =
        function_builder_->AfterAll(implicit_token_data_->control_tokens);
    std::vector<BValue> elements = {join_token, return_value};
    return_value = function_builder_->Tuple(std::move(elements));
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * f,
                       function_builder_->BuildWithReturnValue(return_value));
  XLS_VLOG(5) << "Built function: " << f->name();
  XLS_RETURN_IF_ERROR(VerifyFunction(f));

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
  if (requires_implicit_token && node->is_public() && !node->IsParametric()) {
    XLS_ASSIGN_OR_RETURN(xls::Function * wrapper,
                         EmitImplicitTokenEntryWrapper(f, node));
    package_data_.wrappers.insert(wrapper);
  }

  package_data_.ir_to_dslx[f] = node;
  return f;
}

absl::Status FunctionConverter::HandleColonRef(ColonRef* node) {
  // Implementation note: ColonRef "invocations" are handled in Invocation (by
  // resolving the mangled callee name, which should have been IR converted in
  // dependency order).
  if (absl::optional<Import*> import = node->ResolveImportSubject()) {
    absl::optional<const ImportedInfo*> imported =
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
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data_->GetRootTypeInfo(enum_def->owner()));
  ScopedTypeInfoSwap stis(this, type_info);
  XLS_ASSIGN_OR_RETURN(Expr * value, enum_def->GetValue(node->attr()));

  // Find and visit transitive dependencies as above.
  XLS_ASSIGN_OR_RETURN(auto constant_deps, GetConstantDepFreevars(value));
  for (const auto& dep : constant_deps) {
    XLS_RETURN_IF_ERROR(Visit(dep));
  }

  XLS_RETURN_IF_ERROR(Visit(value));
  return DefAlias(/*from=*/value, /*to=*/node).status();
}

absl::Status FunctionConverter::HandleSplatStructInstance(
    SplatStructInstance* node) {
  XLS_RETURN_IF_ERROR(Visit(node->splatted()));
  XLS_ASSIGN_OR_RETURN(BValue original, Use(node->splatted()));

  absl::flat_hash_map<std::string, BValue> updates;
  for (const auto& item : node->members()) {
    XLS_RETURN_IF_ERROR(Visit(item.second));
    XLS_ASSIGN_OR_RETURN(updates[item.first], Use(item.second));
  }

  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefStruct(ToTypeDefinition(node->struct_ref())));
  std::vector<BValue> members;
  for (int64_t i = 0; i < struct_def->members().size(); ++i) {
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

absl::Status FunctionConverter::HandleStructInstance(StructInstance* node) {
  std::vector<BValue> operands;
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefStruct(ToTypeDefinition(node->struct_def())));
  std::vector<Value> const_operands;
  for (auto [_, member_expr] : node->GetOrderedMembers(struct_def)) {
    XLS_RETURN_IF_ERROR(Visit(member_expr));
    XLS_ASSIGN_OR_RETURN(BValue operand, Use(member_expr));
    operands.push_back(operand);
  }

  Def(node, [this, &operands](absl::optional<SourceLocation> loc) {
    return function_builder_->Tuple(std::move(operands), loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<std::string> FunctionConverter::GetCalleeIdentifier(
    Invocation* node) {
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
    absl::optional<const ImportedInfo*> info =
        current_type_info_->GetImported(*import);
    m = (*info)->module;
  } else {
    return absl::InternalError("Invalid callee: " + callee->ToString());
  }

  absl::optional<dslx::Function*> f = m->GetFunction(callee_name);
  if (!f.has_value()) {
    // For e.g. builtins that are not in the module we just provide the name
    // directly.
    return callee_name;
  }

  // We have to mangle the symbolic bindings into the name to get the fully
  // resolved symbol.
  absl::btree_set<std::string> free_keys = (*f)->GetFreeParametricKeySet();
  const CallingConvention convention = GetCallingConvention(f.value());
  if (!(*f)->IsParametric()) {
    return MangleDslxName(m->name(), (*f)->identifier(), convention, free_keys);
  }

  absl::optional<const SymbolicBindings*> resolved_symbolic_bindings =
      GetInstantiationCalleeBindings(node);
  XLS_RET_CHECK(resolved_symbolic_bindings.has_value());
  XLS_VLOG(5) << absl::StreamFormat("Node `%s` (%s) @ %s symbolic bindings %s",
                                    node->ToString(), node->GetNodeTypeName(),
                                    node->span().ToString(),
                                    (*resolved_symbolic_bindings)->ToString());
  XLS_RET_CHECK(!(*resolved_symbolic_bindings)->empty());
  return MangleDslxName(m->name(), (*f)->identifier(), convention, free_keys,
                        resolved_symbolic_bindings.value());
}

absl::Status FunctionConverter::HandleBinop(Binop* node) {
  XLS_VLOG(5) << "HandleBinop: " << node->ToString();
  absl::optional<const ConcreteType*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* bits_type = dynamic_cast<const BitsType*>(lhs_type.value());
  bool signed_input = bits_type != nullptr && bits_type->is_signed();
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue rhs, Use(node->rhs()));
  std::function<BValue(absl::optional<SourceLocation>)> ir_func;

  switch (node->kind()) {
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
        if (signed_input) {
          return function_builder_->SDiv(lhs, rhs, loc);
        }
        return function_builder_->UDiv(lhs, rhs, loc);
      };
      break;
    // Non-equality comparisons.
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
    case BinopKind::kShr:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        if (signed_input) {
          return function_builder_->Shra(lhs, rhs, loc);
        }
        return function_builder_->Shrl(lhs, rhs, loc);
      };
      break;
    case BinopKind::kShl:
      ir_func = [&](absl::optional<SourceLocation> loc) {
        return function_builder_->Shll(lhs, rhs, loc);
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

absl::Status FunctionConverter::HandleAttr(Attr* node) {
  XLS_RETURN_IF_ERROR(Visit(node->lhs()));
  absl::optional<const ConcreteType*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* struct_type = dynamic_cast<const StructType*>(lhs_type.value());
  const std::string& identifier = node->attr()->identifier();
  XLS_ASSIGN_OR_RETURN(int64_t index, struct_type->GetMemberIndex(identifier));
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

absl::Status FunctionConverter::HandleTernary(Ternary* node) {
  XLS_RETURN_IF_ERROR(Visit(node->test()));
  XLS_ASSIGN_OR_RETURN(BValue arg0, Use(node->test()));

  {
    ScopedControlPredicate scp(
        this, [&](const PredicateFun& orig_control_predicate) {
          BValue activated = orig_control_predicate();
          XLS_CHECK_EQ(activated.GetType()->AsBitsOrDie()->bit_count(), 1);
          return function_builder_->And(activated, arg0);
        });
    XLS_RETURN_IF_ERROR(Visit(node->consequent()));
  }

  XLS_ASSIGN_OR_RETURN(BValue arg1, Use(node->consequent()));

  {
    ScopedControlPredicate scp(
        this, [&](const PredicateFun& orig_control_predicate) {
          BValue activated = orig_control_predicate();
          XLS_CHECK_EQ(activated.GetType()->AsBitsOrDie()->bit_count(), 1);
          return function_builder_->And(orig_control_predicate(),
                                        function_builder_->Not(arg0));
        });
    XLS_RETURN_IF_ERROR(Visit(node->alternate()));
  }

  XLS_ASSIGN_OR_RETURN(BValue arg2, Use(node->alternate()));

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Select(arg0, arg1, arg2, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinAndReduce(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->AndReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinArraySlice(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue start, Use(node->args()[1]));

  auto* width_array = dynamic_cast<Array*>(node->args()[2]);
  XLS_RET_CHECK_NE(width_array, nullptr);
  XLS_RET_CHECK_NE(width_array->type_annotation(), nullptr);
  auto* width_type =
      dynamic_cast<ArrayTypeAnnotation*>(width_array->type_annotation());
  XLS_ASSIGN_OR_RETURN(uint64_t width,
                       dynamic_cast<Number*>(width_type->dim())->GetAsUint64());

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->ArraySlice(arg, start, width, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinBitSlice(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(Bits start_bits, GetConstBits(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(uint64_t start, start_bits.ToUint64());
  XLS_ASSIGN_OR_RETURN(Bits width_bits, GetConstBits(node->args()[2]));
  XLS_ASSIGN_OR_RETURN(uint64_t width, width_bits.ToUint64());
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->BitSlice(arg, start, width, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinBitSliceUpdate(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue start, Use(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(BValue update_value, Use(node->args()[2]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->BitSliceUpdate(arg, start, update_value, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinClz(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Clz(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinCtz(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Ctz(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinOneHot(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue input, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(Bits lsb_prio, GetConstBits(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(uint64_t lsb_prio_value, lsb_prio.ToUint64());

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->OneHot(
        input, lsb_prio_value ? LsbOrMsb::kLsb : LsbOrMsb::kMsb, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinOneHotSel(Invocation* node) {
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
    return function_builder_->OneHotSelect(selector, cases, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinOrReduce(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->OrReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinRev(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->Reverse(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinSignex(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 2);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));

  // Remember - it's the _type_ of the RHS of a signex that gives the new bit
  // count, not the value!
  auto* bit_count = dynamic_cast<Number*>(node->args()[1]);
  XLS_RET_CHECK_NE(bit_count, nullptr);
  XLS_RET_CHECK(bit_count->type_annotation());
  auto* type_annotation =
      dynamic_cast<BuiltinTypeAnnotation*>(bit_count->type_annotation());
  int64_t new_bit_count = type_annotation->GetBitCount();

  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->SignExtend(arg, new_bit_count, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinUpdate(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(BValue index, Use(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(BValue new_value, Use(node->args()[2]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->ArrayUpdate(arg, new_value, {index}, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinXorReduce(Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Def(node, [&](absl::optional<SourceLocation> loc) {
    return function_builder_->XorReduce(arg, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinScmp(SignedCmp cmp,
                                                  Invocation* node) {
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

absl::Status FunctionConverter::CastToArray(Cast* node,
                                            const ArrayType& output_type) {
  XLS_ASSIGN_OR_RETURN(BValue bits, Use(node->expr()));
  std::vector<BValue> slices;
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim element_bit_count_dim,
                       output_type.element_type().GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t element_bit_count,
      ConcreteTypeDim::GetAs64Bits(element_bit_count_dim.value()));
  XLS_ASSIGN_OR_RETURN(int64_t array_size, ConcreteTypeDim::GetAs64Bits(
                                               output_type.size().value()));
  // MSb becomes lowest-indexed array element.
  for (int64_t i = 0; i < array_size; ++i) {
    slices.push_back(function_builder_->BitSlice(bits, i * element_bit_count,
                                                 element_bit_count));
  }
  std::reverse(slices.begin(), slices.end());
  xls::Type* element_type = package()->GetBitsType(element_bit_count);
  Def(node, [this, &slices, element_type](absl::optional<SourceLocation> loc) {
    return function_builder_->Array(std::move(slices), element_type, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::CastFromArray(Cast* node,
                                              const ConcreteType& output_type) {
  XLS_ASSIGN_OR_RETURN(BValue array, Use(node->expr()));
  XLS_ASSIGN_OR_RETURN(xls::Type * input_type, ResolveTypeToIr(node->expr()));
  xls::ArrayType* array_type = input_type->AsArrayOrDie();
  const int64_t array_size = array_type->size();
  std::vector<BValue> pieces;
  for (int64_t i = 0; i < array_size; ++i) {
    BValue index = function_builder_->Literal(UBits(i, 32));
    pieces.push_back(function_builder_->ArrayIndex(array, {index}));
  }
  Def(node, [this, &pieces](absl::optional<SourceLocation> loc) {
    return function_builder_->Concat(std::move(pieces), loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<FunctionConverter::DerefVariant>
FunctionConverter::DerefStructOrEnum(TypeDefinition node) {
  while (absl::holds_alternative<TypeDef*>(node)) {
    auto* type_def = absl::get<TypeDef*>(node);
    TypeAnnotation* annotation = type_def->type_annotation();
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
  absl::optional<const ImportedInfo*> info =
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
  XLS_RET_CHECK(absl::holds_alternative<StructDef*>(v));
  return absl::get<StructDef*>(v);
}

absl::StatusOr<EnumDef*> FunctionConverter::DerefEnum(TypeDefinition node) {
  XLS_ASSIGN_OR_RETURN(DerefVariant v, DerefStructOrEnum(node));
  XLS_RET_CHECK(absl::holds_alternative<EnumDef*>(v));
  return absl::get<EnumDef*>(v);
}

absl::StatusOr<ConcreteTypeDim> FunctionConverter::ResolveDim(
    ConcreteTypeDim dim) {
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

absl::StatusOr<std::unique_ptr<ConcreteType>> FunctionConverter::ResolveType(
    AstNode* node) {
  XLS_RET_CHECK(current_type_info_ != nullptr);
  absl::optional<const ConcreteType*> t = current_type_info_->GetItem(node);
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

absl::StatusOr<Value> FunctionConverter::GetConstValue(AstNode* node) const {
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

absl::StatusOr<Bits> FunctionConverter::GetConstBits(AstNode* node) const {
  XLS_ASSIGN_OR_RETURN(Value value, GetConstValue(node));
  return value.GetBitsWithStatus();
}

absl::Status FunctionConverter::HandleConstantArray(ConstantArray* node) {
  // Note: previously we would force constant evaluation here, but because all
  // constexprs should be evaluated during typechecking, we shouldn't need to
  // forcibly do constant evaluation at IR conversion time; therefore, we just
  // build BValues and let XLS opt constant fold them.
  return HandleArray(node);
}

absl::StatusOr<xls::Type*> FunctionConverter::ResolveTypeToIr(AstNode* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_type,
                       ResolveType(node));
  return TypeToIr(*concrete_type);
}

absl::StatusOr<xls::Type*> FunctionConverter::TypeToIr(
    const ConcreteType& concrete_type) {
  XLS_VLOG(5) << "Converting concrete type to IR: " << concrete_type;
  if (auto* array_type = dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(xls::Type * element_type,
                         TypeToIr(array_type->element_type()));
    XLS_ASSIGN_OR_RETURN(int64_t element_count,
                         ResolveDimToInt(array_type->size()));
    xls::Type* result = package()->GetArrayType(element_count, element_type);
    XLS_VLOG(5) << "Converted type to IR; concrete type: " << concrete_type
                << " ir: " << result->ToString()
                << " element_count: " << element_count;
    return result;
  }
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, ResolveDimToInt(bits_type->size()));
    return package()->GetBitsType(bit_count);
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    XLS_ASSIGN_OR_RETURN(int64_t bit_count, enum_type->size().GetAsInt64());
    return package()->GetBitsType(bit_count);
  }
  if (dynamic_cast<const TokenType*>(&concrete_type)) {
    return package()->GetTokenType();
  }
  std::vector<xls::Type*> members;
  if (auto* struct_type = dynamic_cast<const StructType*>(&concrete_type)) {
    for (const std::unique_ptr<ConcreteType>& m : struct_type->members()) {
      XLS_ASSIGN_OR_RETURN(xls::Type * type, TypeToIr(*m));
      members.push_back(type);
    }
    return package()->GetTupleType(std::move(members));
  }
  auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type);
  XLS_RET_CHECK(tuple_type != nullptr) << concrete_type;
  for (const std::unique_ptr<ConcreteType>& m : tuple_type->members()) {
    XLS_ASSIGN_OR_RETURN(xls::Type * type, TypeToIr(*m));
    members.push_back(type);
  }
  return package()->GetTupleType(std::move(members));
}

absl::Status ConvertOneFunctionInternal(
    PackageData& package_data, Module* module, Function* function,
    TypeInfo* type_info, ImportData* import_data,
    const SymbolicBindings* symbolic_bindings, const ConvertOptions& options) {
  // Validate the requested conversion looks sound in terms of provided
  // parametrics.
  if (symbolic_bindings != nullptr) {
    XLS_RETURN_IF_ERROR(
        ConversionRecord::ValidateParametrics(function, *symbolic_bindings));
  }

  absl::flat_hash_map<std::string, Function*> function_by_name =
      module->GetFunctionByName();
  absl::flat_hash_map<std::string, ConstantDef*> constant_by_name =
      module->GetConstantByName();
  absl::flat_hash_map<std::string, TypeDefinition> type_definition_by_name =
      module->GetTypeDefinitionByName();
  absl::flat_hash_map<std::string, Import*> import_by_name =
      module->GetImportByName();

  FunctionConverter converter(package_data, module, import_data, options);

  XLS_ASSIGN_OR_RETURN(auto constant_deps,
                       GetConstantDepFreevars(function->body()));
  for (const auto& dep : constant_deps) {
    converter.AddConstantDep(dep);
  }

  XLS_VLOG(3) << absl::StreamFormat("Converting function: %s",
                                    function->ToString());
  XLS_RETURN_IF_ERROR(
      converter.HandleFunction(function, type_info, symbolic_bindings)
          .status());
  return absl::OkStatus();
}

}  // namespace

// Converts the functions in the call graph in a specified order.
//
// Args:
//   order: order for conversion
//   import_data: Contains type information used in conversion.
//   options: Conversion option flags.
//   package: output of function
static absl::Status ConvertCallGraph(absl::Span<const ConversionRecord> order,
                                     ImportData* import_data,
                                     const ConvertOptions& options,
                                     PackageData& package_data) {
  XLS_VLOG(3) << "Conversion order: ["
              << absl::StrJoin(
                     order, ", ",
                     [](std::string* out, const ConversionRecord& record) {
                       absl::StrAppend(out, record.ToString());
                     })
              << "]";
  for (const ConversionRecord& record : order) {
    XLS_VLOG(3) << "Converting to IR: " << record.ToString();
    XLS_RETURN_IF_ERROR(ConvertOneFunctionInternal(
        package_data, record.module(), record.f(), record.type_info(),
        import_data, &record.symbolic_bindings(), options));
  }

  XLS_VLOG(3) << "Verifying converted package";
  XLS_RETURN_IF_ERROR(VerifyPackage(package_data.package));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Package>> ConvertModuleToPackage(
    Module* module, ImportData* import_data, const ConvertOptions& options,
    bool traverse_tests) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * root_type_info,
                       import_data->GetRootTypeInfo(module));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrder(module, root_type_info, traverse_tests));
  auto package = absl::make_unique<Package>(module->name());
  PackageData package_data{package.get()};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));

  XLS_RETURN_IF_ERROR(
      WrapEntryIfImplicitToken(package_data, import_data, options));

  return std::move(package);
}

absl::StatusOr<std::string> ConvertModule(Module* module,
                                          ImportData* import_data,
                                          const ConvertOptions& options) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ConvertModuleToPackage(module, import_data, options));
  return package->DumpIr();
}

absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, absl::string_view entry_function_name,
    ImportData* import_data, const SymbolicBindings* symbolic_bindings,
    const ConvertOptions& options) {
  absl::optional<Function*> f = module->GetFunction(entry_function_name);
  XLS_RET_CHECK(f.has_value());
  XLS_ASSIGN_OR_RETURN(TypeInfo * func_type_info,
                       import_data->GetRootTypeInfoForNode(f.value()));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrderForEntry(f.value(), func_type_info));
  Package package(module->name());
  PackageData package_data{&package};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));
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
      auto get_type = [&](int64_t i) -> const ConcreteType* {
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
      for (int64_t i = 0; i < v.elements().size(); ++i) {
        const Value& e = v.elements()[i];
        XLS_ASSIGN_OR_RETURN(InterpValue iv,
                             ValueToInterpValue(e, get_type(i)));
        members.push_back(iv);
      }
      if (v.kind() == ValueKind::kTuple) {
        return InterpValue::MakeTuple(std::move(members));
      }
      return InterpValue::MakeArray(std::move(members));
    }
    default:
      return absl::InvalidArgumentError(
          "Cannot convert IR value to interpreter value: " + v.ToString());
  }
}

}  // namespace xls::dslx
