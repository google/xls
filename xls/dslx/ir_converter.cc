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
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "xls/common/visitor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/ast_utils.h"
#include "xls/dslx/builtins_metadata.h"
#include "xls/dslx/constexpr_evaluator.h"
#include "xls/dslx/extract_conversion_order.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_conversion_utils.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/proc_config_ir_converter.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"

namespace xls::dslx {
namespace {

// Bundles together a package pointer with a supplementary map we keep that
// shows the DSLX function that led to IR functions in the package.
struct PackageData {
  Package* package;
  absl::flat_hash_map<xls::FunctionBase*, dslx::Function*> ir_to_dslx;
  absl::flat_hash_set<xls::Function*> wrappers;
};

// Returns a status that indicates an error in the IR conversion process.
absl::Status ConversionErrorStatus(const std::optional<Span>& span,
                                   std::string_view message) {
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
static bool GetRequiresImplicitToken(dslx::Function* f, ImportData* import_data,
                                     const ConvertOptions& options) {
  std::optional<bool> requires_opt = import_data->GetRootTypeInfo(f->owner())
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
                    ImportData* import_data, ConvertOptions options,
                    ProcConversionData* proc_data, bool is_top);

  // Main entry point to request conversion of the DSLX function "f" to an IR
  // function.
  absl::Status HandleFunction(Function* node, TypeInfo* type_info,
                              const SymbolicBindings* symbolic_bindings);

  absl::Status HandleProcNextFunction(
      Function* pf, const Invocation* invocation, TypeInfo* type_info,
      ImportData* import_data, const SymbolicBindings* symbolic_bindings,
      const ProcId& proc_id, ProcConversionData* proc_data);

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
    BuilderBase* fb = function_builder_.get();
    Type* wrapped_type = WrapIrForImplicitTokenType(type, package());
    BValue param = fb->Param("__token_wrapped", wrapped_type);
    BValue entry_token = fb->TupleIndex(param, 0);
    BValue activated = fb->TupleIndex(param, 1);
    auto create_control_predicate = [activated] { return activated; };
    implicit_token_data_ =
        ImplicitTokenData{entry_token, activated, create_control_predicate};
    BValue unwrapped = fb->TupleIndex(param, 2);
    return unwrapped;
  }

  // An IR-conversion-time-constant value, decorates a BValue with its evaluated
  // constant form.
  struct CValue {
    Value ir_value;
    BValue value;
  };

  // Every AST node has an "IR value" that is either a function builder value
  // (BValue) or its IR-conversion-time-constant-decorated cousin (CValue), or
  // an inter-proc Channel.
  using IrValue = std::variant<BValue, CValue, Channel*>;

  // Helper for converting an IR value to its BValue pointer for use in
  // debugging.
  static std::string ToString(const IrValue& value);

  void SetFunctionBuilder(std::unique_ptr<BuilderBase> builder);

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
  absl::Status DefAlias(const AstNode* from, const AstNode* to);

  // Returns the BValue previously noted as corresponding to "node" (via a
  // Def/DefAlias).
  absl::StatusOr<BValue> Use(const AstNode* node) const;

  void SetNodeToIr(const AstNode* node, IrValue value);
  std::optional<IrValue> GetNodeToIr(const AstNode* node) const;

  // Returns the constant value corresponding to the IrValue of "node", or
  // returns an error if it is not present (or not constant).
  absl::StatusOr<Value> GetConstValue(const AstNode* node) const;

  // As above, but also checks it is a constant Bits value.
  absl::StatusOr<Bits> GetConstBits(const AstNode* node) const;

  // Resolves node's type and resolves all of its dimensions via `ResolveDim()`.
  absl::StatusOr<std::unique_ptr<ConcreteType>> ResolveType(
      const AstNode* node);

  // Helper that composes ResolveType() and TypeToIr().
  absl::StatusOr<xls::Type*> ResolveTypeToIr(AstNode* node);

  // -- Accessors

  void SetSymbolicBindings(const SymbolicBindings* value) {
    symbolic_binding_map_ = value->ToMap();
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
  std::optional<InterpValue> get_symbolic_binding(
      std::string_view identifier) const {
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
  //  The symbolic bindings for the given instantiation (function call or proc
  //  spawn).
  std::optional<const SymbolicBindings*> GetInvocationCalleeBindings(
      const Invocation* invocation) const {
    SymbolicBindings key = GetSymbolicBindings();
    return import_data_->GetRootTypeInfo(invocation->owner())
        .value()
        ->GetInvocationCalleeBindings(invocation, key);
  }

  // Helpers for HandleBinop().
  absl::Status HandleConcat(const Binop* node, BValue lhs, BValue rhs);
  absl::Status HandleEq(const Binop* node, BValue lhs, BValue rhs);
  absl::Status HandleNe(const Binop* node, BValue lhs, BValue rhs);

  // AstNode handlers.
  absl::Status HandleBinop(const Binop* node);
  absl::Status HandleConstRef(const ConstRef* node);
  absl::Status HandleNameRef(const NameRef* node);
  absl::Status HandleNumber(const Number* node);
  absl::Status HandleParam(const Param* node);
  absl::Status HandleString(const String* node);
  absl::Status HandleUnop(const Unop* node);
  absl::Status HandleXlsTuple(const XlsTuple* node);

  // AstNode handlers that recur "manually" internal to the handler.
  absl::Status HandleArray(const Array* node);
  absl::Status HandleAttr(const Attr* node);
  absl::Status HandleBlock(const Block* node);
  absl::Status HandleCast(const Cast* node);
  absl::Status HandleColonRef(const ColonRef* node);
  absl::Status HandleConstantArray(const ConstantArray* node);
  absl::Status HandleConstantDef(const ConstantDef* node);
  absl::Status HandleFor(const For* node);
  absl::Status HandleFormatMacro(const FormatMacro* node);
  absl::Status HandleIndex(const Index* node);
  absl::Status HandleInvocation(const Invocation* node);
  absl::Status HandleJoin(const Join* node);
  absl::Status HandleLet(const Let* node);
  absl::Status HandleMatch(const Match* node);
  absl::Status HandleRange(const Range* node);
  absl::Status HandleRecv(const Recv* node);
  absl::Status HandleRecvIf(const RecvIf* node);
  absl::Status HandleRecvIfNonBlocking(const RecvIfNonBlocking* node);
  absl::Status HandleRecvNonBlocking(const RecvNonBlocking* node);
  absl::Status HandleSend(const Send* node);
  absl::Status HandleSendIf(const SendIf* node);
  absl::Status HandleSplatStructInstance(const SplatStructInstance* node);
  absl::Status HandleStructInstance(const StructInstance* node);
  absl::Status HandleTernary(const Ternary* node);
  absl::Status HandleTupleIndex(const TupleIndex* node);

  // Handles invocation of a user-defined function (UDF).
  absl::Status HandleUdfInvocation(const Invocation* node, xls::Function* f,
                                   std::vector<BValue> args);

  // Handles the fail!() builtin invocation.
  absl::Status HandleFailBuiltin(const Invocation* node, BValue arg);

  // Handles the cover!() builtin invocation.
  absl::Status HandleCoverBuiltin(const Invocation* node, BValue condition);

  // Handles an arm of a match expression.
  absl::StatusOr<BValue> HandleMatcher(NameDefTree* matcher,
                                       absl::Span<const int64_t> index,
                                       const BValue& matched_value,
                                       const ConcreteType& matched_type);

  // Makes the specified builtin available to the package.
  absl::StatusOr<BValue> DefMapWithBuiltin(
      const Invocation* parent_node, NameRef* node, AstNode* arg,
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
  absl::StatusOr<Value> EvaluateConstFunction(const Invocation* node);

  absl::StatusOr<BValue> HandleMap(const Invocation* node);

  absl::StatusOr<BValue> HandleFail(const Invocation* node);

  // Builtin invocation handlers.
  absl::Status HandleBuiltinAndReduce(const Invocation* node);
  absl::Status HandleBuiltinArraySlice(const Invocation* node);
  absl::Status HandleBuiltinBitSlice(const Invocation* node);
  absl::Status HandleBuiltinBitSliceUpdate(const Invocation* node);
  absl::Status HandleBuiltinClz(const Invocation* node);
  absl::Status HandleBuiltinCtz(const Invocation* node);
  absl::Status HandleBuiltinGate(const Invocation* node);
  absl::Status HandleBuiltinOneHot(const Invocation* node);
  absl::Status HandleBuiltinOneHotSel(const Invocation* node);
  absl::Status HandleBuiltinPrioritySel(const Invocation* node);
  absl::Status HandleBuiltinOrReduce(const Invocation* node);
  absl::Status HandleBuiltinRev(const Invocation* node);
  absl::Status HandleBuiltinSignex(const Invocation* node);
  absl::Status HandleBuiltinSMulp(const Invocation* node);
  absl::Status HandleBuiltinUpdate(const Invocation* node);
  absl::Status HandleBuiltinUMulp(const Invocation* node);
  absl::Status HandleBuiltinXorReduce(const Invocation* node);

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

  absl::Status CastToArray(const Cast* node, const ArrayType& output_type);
  absl::Status CastFromArray(const Cast* node, const ConcreteType& output_type);

  // Returns the fully resolved (mangled) name for the callee of the given node,
  // with parametric values mangled in appropriately.
  absl::StatusOr<std::string> GetCalleeIdentifier(const Invocation* node);

  // Contains/returns the pertinent information about a range expression (either
  // a Range node or the range() builtin).
  struct RangeData {
    int64_t start_value;
    int64_t trip_count;
    int64_t bit_width;
  };
  absl::StatusOr<RangeData> GetRangeData(const Expr* iterable);

  template <typename T>
  absl::StatusOr<T> DerefStructOrEnumFromNameRef(
      NameRef* name_ref,
      const std::function<absl::StatusOr<T>(TypeDefinition)>& f) {
    AnyNameDef any_name_def = name_ref->name_def();
    const auto* name_def = std::get<const NameDef*>(any_name_def);
    AstNode* definer = name_def->definer();
    XLS_ASSIGN_OR_RETURN(TypeDefinition td, ToTypeDefinition(definer));
    return f(td);
  }

  // Dereferences a type definition to either a struct definition or enum
  // definition.
  using DerefVariant = std::variant<StructDef*, EnumDef*>;
  absl::StatusOr<DerefVariant> DerefStructOrEnum(TypeDefinition node);

  SourceInfo ToSourceInfo(const std::optional<Span>& span) {
    if (!options_.emit_positions || !span.has_value()) {
      return SourceInfo();
    }
    const Pos& start_pos = span->start();
    Lineno lineno(start_pos.lineno());
    Colno colno(start_pos.colno());
    // TODO(leary): 2020-12-20 Figure out the fileno based on the module owner
    // of node.
    return SourceInfo(SourceLocation{fileno_, lineno, colno});
  }

  // Defines "node" to map the result of running "ir_func" with "args" -- if
  // emit_positions is on grabs the span from the node and uses it in the call.
  absl::StatusOr<BValue> DefWithStatus(
      const AstNode* node,
      const std::function<absl::StatusOr<BValue>(const SourceInfo&)>& ir_func);

  // Specialization for Def() above when the "ir_func" is infallible.
  BValue Def(const AstNode* node,
             const std::function<BValue(const SourceInfo&)>& ir_func);

  // Def(), but for constant/constexpr values, adds a literal as the IR
  // function.
  CValue DefConst(const AstNode* node, Value ir_value);

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
  absl::flat_hash_map<const AstNode*, IrValue> node_to_ir_;

  // Various conversion options; e.g. whether or not to emit source code
  // positions into the XLS IR.
  ConvertOptions options_;

  // Constants that this translation depends upon (as determined externally).
  std::vector<ConstantDef*> constant_deps_;

  // Function builder being used to create BValues.
  std::unique_ptr<BuilderBase> function_builder_;

  // When we have a fail!() operation we implicitly need to thread a token /
  // activation boolean -- the data for this is kept here if necessary.
  std::optional<ImplicitTokenData> implicit_token_data_;

  // Mapping of symbolic bindings active in this translation (e.g. what integral
  // values parametrics are taking on).
  absl::flat_hash_map<std::string, InterpValue> symbolic_binding_map_;

  // File number for use in source positions.
  Fileno fileno_;

  // This is only set for procs; it holds the final recurrent state.
  BValue next_value_;

  // Number of "counted for" nodes we've observed in this function.
  int64_t counted_for_count_ = 0;

  // Uses id_to_members to resolve a proc's [constant] member values.
  ProcConversionData* proc_data_;

  std::vector<BValue> tokens_;

  // The ID of the unique Proc instance currently being converted. Only valid if
  // a Proc is being converted.
  ProcId proc_id_;

  // If the function is the entry function resulting in a top entity in the IR.
  bool is_top_;
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
  absl::Status VisitChildren(const AstNode* node) {
    for (AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(Visit(child));
    }
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
  NO_TRAVERSE_DISPATCH_VISIT(Join)
  NO_TRAVERSE_DISPATCH_VISIT(Let)
  NO_TRAVERSE_DISPATCH_VISIT(Match)
  NO_TRAVERSE_DISPATCH_VISIT(Range)
  NO_TRAVERSE_DISPATCH_VISIT(Recv)
  NO_TRAVERSE_DISPATCH_VISIT(RecvIf)
  NO_TRAVERSE_DISPATCH_VISIT(RecvIfNonBlocking)
  NO_TRAVERSE_DISPATCH_VISIT(RecvNonBlocking)
  NO_TRAVERSE_DISPATCH_VISIT(Send)
  NO_TRAVERSE_DISPATCH_VISIT(SendIf)
  NO_TRAVERSE_DISPATCH_VISIT(SplatStructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(StructInstance)
  NO_TRAVERSE_DISPATCH_VISIT(Ternary)
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
  INVALID(TypeDef)
  INVALID(Proc)
  INVALID(Module)
  INVALID(QuickCheck)
  INVALID(Spawn)
  INVALID(StructDef)

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

absl::Status FunctionConverter::Visit(AstNode* node) {
  FunctionConverterVisitor visitor(this);
  return visitor.Visit(node);
}

/* static */ std::string FunctionConverter::ToString(const IrValue& value) {
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
      options_(std::move(options)),
      // TODO(leary): 2019-07-19 Create a way to get the file path from the
      // module.
      fileno_(package_data.package->GetOrCreateFileno("fake_file.x")),
      proc_data_(proc_data),
      is_top_(is_top) {
  XLS_VLOG(5) << "Constructed IR converter: " << this;
}

void FunctionConverter::SetFunctionBuilder(
    std::unique_ptr<BuilderBase> builder) {
  XLS_CHECK(function_builder_ == nullptr);
  function_builder_ = std::move(builder);
}

void FunctionConverter::AddConstantDep(ConstantDef* constant_def) {
  XLS_VLOG(2) << "Adding constant dep: " << constant_def->ToString();
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
  XLS_VLOG(6) << absl::StreamFormat(
      "Aliased node '%s' (%s) to be same as '%s' (%s): %s", to->ToString(),
      to->GetNodeTypeName(), from->ToString(), from->GetNodeTypeName(),
      ToString(value));
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
  XLS_VLOG(6) << absl::StreamFormat(
      "Define node '%s' (%s) to be %s @ %s", node->ToString(),
      node->GetNodeTypeName(), ToString(result), SpanToString(node->GetSpan()));
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
  XLS_VLOG(6) << absl::StreamFormat("Using node '%s' (%p) as IR value %s.",
                                    node->ToString(), node, ToString(ir_value));
  if (std::holds_alternative<BValue>(ir_value)) {
    return std::get<BValue>(ir_value);
  }
  XLS_RET_CHECK(std::holds_alternative<CValue>(ir_value));
  return std::get<CValue>(ir_value).value;
}

void FunctionConverter::SetNodeToIr(const AstNode* node, IrValue value) {
  XLS_VLOG(6) << absl::StreamFormat("Setting node '%s' (%p) to IR value %s.",
                                    node->ToString(), node, ToString(value));
  node_to_ir_[node] = value;
}

std::optional<FunctionConverter::IrValue> FunctionConverter::GetNodeToIr(
    const AstNode* node) const {
  auto it = node_to_ir_.find(node);
  if (it == node_to_ir_.end()) {
    return absl::nullopt;
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
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> output_type,
                       ResolveType(node));
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
                       [&](SourceInfo loc) -> absl::StatusOr<BValue> {
                         return function_builder_->Eq(lhs, rhs, loc);
                       })
      .status();
}

absl::Status FunctionConverter::HandleNe(const Binop* node, BValue lhs,
                                         BValue rhs) {
  return DefWithStatus(node,
                       [&](SourceInfo loc) -> absl::StatusOr<BValue> {
                         return function_builder_->Ne(lhs, rhs, loc);
                       })
      .status();
}

absl::Status FunctionConverter::HandleNumber(const Number* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> type, ResolveType(node));
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim dim, type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                       std::get<InterpValue>(dim.value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, node->GetBits(bit_count));
  DefConst(node, Value(bits));
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleString(const String* node) {
  std::vector<Value> elements;
  for (const char letter : node->text()) {
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
    return function_builder_->Tuple(std::move(operands), loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleParam(const Param* node) {
  XLS_VLOG(5) << "FunctionConverter::HandleParam: " << node->ToString();
  XLS_ASSIGN_OR_RETURN(xls::Type * type,
                       ResolveTypeToIr(node->type_annotation()));
  Def(node->name_def(), [&](const SourceInfo& loc) {
    return function_builder_->Param(node->identifier(), type);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleConstRef(const ConstRef* node) {
  return DefAlias(node->name_def(), /*to=*/node);
}

absl::Status FunctionConverter::HandleNameRef(const NameRef* node) {
  AstNode* from = ToAstNode(node->name_def());

  if (!node_to_ir_.contains(from)) {
    for (const auto& [k, v] : proc_data_->id_to_members.at(proc_id_)) {
      if (k == node->identifier()) {
        if (std::holds_alternative<Value>(v)) {
          XLS_VLOG(4) << "Reference to Proc member: " << k
                      << " : Value : " << std::get<Value>(v).ToString();
          CValue cvalue;
          cvalue.ir_value = std::get<Value>(v);
          cvalue.value = function_builder_->Literal(cvalue.ir_value);
          SetNodeToIr(from, cvalue);
        } else {
          XLS_VLOG(4) << "Reference to Proc member: " << k
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
  XLS_VLOG(5) << "Visiting ConstantDef expr: " << node->value()->ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue iv,
                       current_type_info_->GetConstExpr(node->value()));
  XLS_ASSIGN_OR_RETURN(Value value, InterpValueToValue(iv));
  Def(node->value(), [this, &value](const SourceInfo& loc) {
    return function_builder_->Literal(value, loc);
  });
  XLS_VLOG(5) << "Aliasing NameDef for constant: "
              << node->name_def()->ToString();
  return DefAlias(node->value(), /*to=*/node->name_def());
}

absl::Status FunctionConverter::HandleLet(const Let* node) {
  XLS_VLOG(5) << "FunctionConverter::HandleLet: " << node->ToString();
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
        DefAlias(node->rhs(), /*to=*/ToAstNode(node->name_def_tree()->leaf())));
    XLS_RETURN_IF_ERROR(Visit(node->body()));
    XLS_RETURN_IF_ERROR(DefAlias(node->body(), node));
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
      levels.push_back(Def(x, [this, &levels, x, index](SourceInfo loc) {
        if (!loc.Empty()) {
          loc = ToSourceInfo(x->is_leaf() ? ToAstNode(x->leaf())->GetSpan()
                                          : x->GetSpan());
        }
        BValue tuple = levels.back();
        xls::TupleType* tuple_type = tuple.GetType()->AsTupleOrDie();
        XLS_CHECK_LT(index, tuple_type->size())
            << "index: " << index << " type: " << tuple_type->ToString();
        BValue tuple_index = function_builder_->TupleIndex(tuple, index, loc);
        return tuple_index;
      }));
      if (x->is_leaf()) {
        XLS_RETURN_IF_ERROR(DefAlias(x, ToAstNode(x->leaf())));
      }
      return absl::OkStatus();
    };

    XLS_RETURN_IF_ERROR(node->name_def_tree()->DoPreorder(walk));
    XLS_RETURN_IF_ERROR(Visit(node->body()));
    XLS_RETURN_IF_ERROR(DefAlias(node->body(), /*to=*/node));
  }

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleCast(const Cast* node) {
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
      std::get<InterpValue>(new_bit_count_ctd.value()).GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim input_bit_count_ctd,
                       input_type->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(
      int64_t old_bit_count,
      std::get<InterpValue>(input_bit_count_ctd.value()).GetBitValueInt64());
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

absl::Status FunctionConverter::HandleMatch(const Match* node) {
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

absl::StatusOr<FunctionConverter::RangeData> FunctionConverter::GetRangeData(
    const Expr* iterable) {
  auto error = [&] {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConversionError: %s iterable (%s) "
                        "must be bits-typed, constexpr, and its start must be "
                        "less than or equal to its limit.",
                        iterable->span().ToString(), iterable->ToString()));
  };

  // Easy case first: using the `..` range operator.
  InterpValue start_value(InterpValue::MakeToken());
  InterpValue limit_value(InterpValue::MakeToken());
  SymbolicBindings bindings(symbolic_binding_map_);

  const auto* range_op = dynamic_cast<const Range*>(iterable);
  if (range_op != nullptr) {
    XLS_ASSIGN_OR_RETURN(start_value,
                         ConstexprEvaluator::EvaluateToValue(
                             import_data_, current_type_info_, bindings,
                             range_op->start(), nullptr));
    XLS_ASSIGN_OR_RETURN(limit_value, ConstexprEvaluator::EvaluateToValue(
                                          import_data_, current_type_info_,
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

    XLS_ASSIGN_OR_RETURN(start_value, ConstexprEvaluator::EvaluateToValue(
                                          import_data_, current_type_info_,
                                          bindings, start, nullptr));
    XLS_ASSIGN_OR_RETURN(limit_value, ConstexprEvaluator::EvaluateToValue(
                                          import_data_, current_type_info_,
                                          bindings, limit, nullptr));
  }

  if (!start_value.IsBits() || !limit_value.IsBits()) {
    return error();
  }

  XLS_ASSIGN_OR_RETURN(int64_t start_int, start_value.GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(int64_t bit_width, start_value.GetBitCount());

  XLS_ASSIGN_OR_RETURN(InterpValue start_ge_limit, start_value.Ge(limit_value));
  if (start_ge_limit.IsTrue()) {
    return RangeData{start_int, 0, bit_width};
  }

  XLS_ASSIGN_OR_RETURN(InterpValue trip_count, limit_value.Sub(start_value));
  XLS_RET_CHECK(trip_count.IsBits());
  int64_t trip_count_int;
  if (trip_count.IsSigned()) {
    XLS_ASSIGN_OR_RETURN(trip_count_int, trip_count.GetBitValueInt64());
  } else {
    XLS_ASSIGN_OR_RETURN(trip_count_int, trip_count.GetBitValueUint64());
  }

  return RangeData{start_int, trip_count_int, bit_width};
}

// Convert a NameDefTree node variant to an AstNode pointer (either the leaf
// node or the interior NameDefTree node).
static AstNode* ToAstNode(
    const std::variant<NameDefTree::Leaf, NameDefTree*>& x) {
  if (std::holds_alternative<NameDefTree*>(x)) {
    return std::get<NameDefTree*>(x);
  }
  return ToAstNode(std::get<NameDefTree::Leaf>(x));
}

absl::Status FunctionConverter::HandleFor(const For* node) {
  XLS_RETURN_IF_ERROR(Visit(node->init()));

  XLS_ASSIGN_OR_RETURN(RangeData range_data, GetRangeData(node->iterable()));

  XLS_VLOG(5) << "Converting for-loop @ " << node->span();
  FunctionConverter body_converter(package_data_, module_, import_data_,
                                   options_, proc_data_, /*is_top=*/false);
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
  AstNode* ivar = ToAstNode(flat[0]);
  auto* name_def = dynamic_cast<NameDef*>(ivar);
  XLS_RET_CHECK(name_def != nullptr);
  XLS_ASSIGN_OR_RETURN(xls::Type * ivar_type, ResolveTypeToIr(name_def));
  BValue loop_index = body_converter.function_builder_->Param(
      name_def->identifier(), ivar_type);

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
  body_converter.SetNodeToIr(name_def, offset_sum);

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
    NameDefTree* accum = std::get<NameDefTree*>(flat[1]);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> carry_type,
                         ResolveType(accum));
    XLS_ASSIGN_OR_RETURN(xls::Type * carry_ir_type,
                         TypeToIr(package_data_.package, *carry_type,
                                  SymbolicBindings(symbolic_binding_map_)));
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
  std::vector<const NameDef*> relevant_name_defs;
  for (const auto& any_name_def : freevars.GetNameDefs()) {
    const auto* name_def = std::get<const NameDef*>(any_name_def);
    std::optional<const ConcreteType*> type =
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

    std::optional<IrValue> ir_value = GetNodeToIr(name_def);
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
      body_converter.DefConst(name_def, constant_value);
    } else {
      // Otherwise, pass in the variable to the loop body function as
      // a parameter.
      relevant_name_defs.push_back(name_def);
      XLS_ASSIGN_OR_RETURN(xls::Type * name_def_type,
                           TypeToIr(package_data_.package, **type,
                                    SymbolicBindings(symbolic_binding_map_)));
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
                       body_builder_ptr->Build());
  XLS_VLOG(5) << "Converted body function: " << body_function->name();

  std::vector<BValue> invariant_args;
  for (const NameDef* name_def : relevant_name_defs) {
    XLS_ASSIGN_OR_RETURN(BValue value, Use(name_def));
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
    const BValue& matched_value, const ConcreteType& matched_type) {
  if (matcher->is_leaf()) {
    NameDefTree::Leaf leaf = matcher->leaf();
    XLS_VLOG(5) << absl::StreamFormat("Matcher is leaf: %s (%s)",
                                      ToAstNode(leaf)->ToString(),
                                      ToAstNode(leaf)->GetNodeTypeName());
    if (std::holds_alternative<WildcardPattern*>(leaf)) {
      return Def(matcher, [&](const SourceInfo& loc) {
        return function_builder_->Literal(UBits(1, 1), loc);
      });
    }
    if (std::holds_alternative<Number*>(leaf) ||
        std::holds_alternative<ColonRef*>(leaf)) {
      XLS_RETURN_IF_ERROR(Visit(ToAstNode(leaf)));
      XLS_ASSIGN_OR_RETURN(BValue to_match, Use(ToAstNode(leaf)));
      return Def(matcher, [&](const SourceInfo& loc) {
        return function_builder_->Eq(to_match, matched_value);
      });
    }
    if (std::holds_alternative<NameRef*>(leaf)) {
      // Comparing for equivalence to a (referenced) name.
      auto* name_ref = std::get<NameRef*>(leaf);
      const auto* name_def = std::get<const NameDef*>(name_ref->name_def());
      XLS_ASSIGN_OR_RETURN(BValue to_match, Use(name_def));
      BValue result = Def(matcher, [&](const SourceInfo& loc) {
        return function_builder_->Eq(to_match, matched_value);
      });
      XLS_RETURN_IF_ERROR(DefAlias(name_def, name_ref));
      return result;
    }
    XLS_RET_CHECK(std::holds_alternative<NameDef*>(leaf));
    auto* name_def = std::get<NameDef*>(leaf);
    BValue ok = Def(name_def, [&](const SourceInfo& loc) {
      return function_builder_->Literal(UBits(1, 1));
    });
    SetNodeToIr(matcher, matched_value);
    SetNodeToIr(ToAstNode(leaf), matched_value);
    return ok;
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
    const Invocation* parent_node, NameRef* node, AstNode* arg,
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
  return Def(parent_node, [&](const SourceInfo& loc) {
    return function_builder_->Map(arg_value, f);
  });
}

absl::StatusOr<BValue> FunctionConverter::HandleMap(const Invocation* node) {
  for (Expr* arg : node->args().subspan(0, node->args().size() - 1)) {
    XLS_RETURN_IF_ERROR(Visit(arg));
  }
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  Expr* fn_node = node->args()[1];
  XLS_VLOG(5) << "Function being mapped AST: " << fn_node->ToString();
  std::optional<const SymbolicBindings*> node_sym_bindings =
      GetInvocationCalleeBindings(node);

  std::string map_fn_name;
  Module* lookup_module = nullptr;
  if (auto* name_ref = dynamic_cast<NameRef*>(fn_node)) {
    map_fn_name = name_ref->identifier();
    if (IsNameParametricBuiltin(map_fn_name)) {
      XLS_VLOG(5) << "Map of parametric builtin: " << map_fn_name;
      return DefMapWithBuiltin(node, name_ref, node->args()[0],
                               *node_sym_bindings.value());
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
                     free_set, node_sym_bindings.value()));
  XLS_VLOG(5) << "Getting function with mangled name: " << mangled_name
              << " from package: " << package()->name();
  XLS_ASSIGN_OR_RETURN(xls::Function * f, package()->GetFunction(mangled_name));
  return Def(node, [&](const SourceInfo& loc) -> BValue {
    return function_builder_->Map(arg, f, loc);
  });
}

absl::Status FunctionConverter::HandleIndex(const Index* node) {
  XLS_RETURN_IF_ERROR(Visit(node->lhs()));
  XLS_ASSIGN_OR_RETURN(BValue lhs, Use(node->lhs()));

  std::optional<const ConcreteType*> lhs_type =
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
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> output_type,
                           ResolveType(node));
      XLS_ASSIGN_OR_RETURN(ConcreteTypeDim output_type_dim,
                           output_type->GetTotalBitCount());
      XLS_ASSIGN_OR_RETURN(int64_t width, output_type_dim.GetAsInt64());
      Def(node, [&](const SourceInfo& loc) {
        return function_builder_->DynamicBitSlice(lhs, start, width, loc);
      });
    } else {
      auto* slice = std::get<Slice*>(rhs);
      std::optional<StartAndWidth> saw =
          current_type_info_->GetSliceStartAndWidth(slice,
                                                    GetSymbolicBindings());
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
  Def(node, [&](const SourceInfo& loc) {
    xls::Type* type = members[0].GetType();
    return function_builder_->Array(std::move(members), type, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleUdfInvocation(const Invocation* node,
                                                    xls::Function* f,
                                                    std::vector<BValue> args) {
  XLS_VLOG(5) << "HandleUdfInvocation: " << f->name() << " via "
              << node->ToString();
  XLS_RET_CHECK(package_data_.ir_to_dslx.contains(f)) << f->name();
  dslx::Function* dslx_callee =
      dynamic_cast<dslx::Function*>(package_data_.ir_to_dslx.at(f));

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

absl::Status FunctionConverter::HandleFailBuiltin(const Invocation* node,
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
    tokens_.push_back(assert_result_token);
  }
  // The result of the failure call is the argument given; e.g. if we were to
  // remove assertions this is the value that would flow in the case that the
  // assertion was hit.
  Def(node,
      [&](const SourceInfo& loc) { return function_builder_->Identity(arg); });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleFormatMacro(const FormatMacro* node) {
  XLS_RET_CHECK(implicit_token_data_.has_value())
      << "Invoking trace_fmt!(), but no implicit token is present for caller @ "
      << node->span();
  XLS_RET_CHECK(implicit_token_data_->create_control_predicate != nullptr);
  BValue control_predicate = implicit_token_data_->create_control_predicate();

  std::vector<BValue> ir_args = {};
  for (auto* arg : node->args()) {
    XLS_RETURN_IF_ERROR(Visit(arg));
    XLS_ASSIGN_OR_RETURN(BValue ir_arg, Use(arg));
    ir_args.push_back(ir_arg);
  }

  BValue trace_result_token =
      function_builder_->Trace(implicit_token_data_->entry_token,
                               control_predicate, ir_args, node->format());
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
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->Tuple(std::vector<BValue>());
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleInvocation(const Invocation* node) {
  XLS_VLOG(5) << "FunctionConverter::HandleInvocation: " << node->ToString();
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
    XLS_RET_CHECK_EQ(args.size(), 2)
        << called_name << " builtin requires two arguments";
    return HandleFailBuiltin(node, std::move(args[1]));
  }
  if (called_name == "cover!") {
    XLS_ASSIGN_OR_RETURN(std::vector<BValue> args, accept_args());
    XLS_RET_CHECK_EQ(args.size(), 2)
        << called_name << " builtin requires two arguments";
    return HandleCoverBuiltin(node, std::move(args[1]));
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

  // The rest of the builtins have "handle" methods we can resolve.
  absl::flat_hash_map<std::string,
                      decltype(&FunctionConverter::HandleBuiltinClz)>
      map = {
          {"clz", &FunctionConverter::HandleBuiltinClz},
          {"ctz", &FunctionConverter::HandleBuiltinCtz},
          {"gate!", &FunctionConverter::HandleBuiltinGate},
          {"signex", &FunctionConverter::HandleBuiltinSignex},
          {"one_hot", &FunctionConverter::HandleBuiltinOneHot},
          {"one_hot_sel", &FunctionConverter::HandleBuiltinOneHotSel},
          {"priority_sel", &FunctionConverter::HandleBuiltinPrioritySel},
          {"slice", &FunctionConverter::HandleBuiltinArraySlice},
          {"bit_slice", &FunctionConverter::HandleBuiltinBitSlice},
          {"bit_slice_update", &FunctionConverter::HandleBuiltinBitSliceUpdate},
          {"rev", &FunctionConverter::HandleBuiltinRev},
          {"and_reduce", &FunctionConverter::HandleBuiltinAndReduce},
          {"or_reduce", &FunctionConverter::HandleBuiltinOrReduce},
          {"xor_reduce", &FunctionConverter::HandleBuiltinXorReduce},
          {"update", &FunctionConverter::HandleBuiltinUpdate},
          {"umulp", &FunctionConverter::HandleBuiltinUMulp},
          {"smulp", &FunctionConverter::HandleBuiltinSMulp},
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

absl::Status FunctionConverter::HandleSend(const Send* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Send nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->channel()));
  XLS_RETURN_IF_ERROR(Visit(node->payload()));
  if (!node_to_ir_.contains(node->channel())) {
    return absl::InternalError("Send channel not found!");
  }
  IrValue ir_value = node_to_ir_[node->channel()];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }
  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->token()));
  XLS_ASSIGN_OR_RETURN(BValue data, Use(node->payload()));
  BValue result = builder_ptr->Send(std::get<Channel*>(ir_value), token, data);
  node_to_ir_[node] = result;
  tokens_.push_back(result);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleSendIf(const SendIf* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Send nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->channel()));
  if (!node_to_ir_.contains(node->channel())) {
    return absl::InternalError("Send channel not found!");
  }
  IrValue ir_value = node_to_ir_[node->channel()];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->condition()));
  XLS_ASSIGN_OR_RETURN(BValue predicate, Use(node->condition()));

  XLS_RETURN_IF_ERROR(Visit(node->payload()));
  XLS_ASSIGN_OR_RETURN(BValue data, Use(node->payload()));
  BValue result =
      builder_ptr->SendIf(std::get<Channel*>(ir_value), token, predicate, data);
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

absl::Status FunctionConverter::HandleRecv(const Recv* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->channel()));
  if (!node_to_ir_.contains(node->channel())) {
    return absl::InternalError("Recv channel not found!");
  }
  IrValue ir_value = node_to_ir_[node->channel()];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->token()));
  BValue value = builder_ptr->Receive(std::get<Channel*>(ir_value), token);
  BValue token_value = builder_ptr->TupleIndex(value, 0);
  tokens_.push_back(token_value);
  node_to_ir_[node] = value;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleRecvNonBlocking(
    const RecvNonBlocking* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->channel()));
  if (!node_to_ir_.contains(node->channel())) {
    return absl::InternalError("Recv channel not found!");
  }
  IrValue ir_value = node_to_ir_[node->channel()];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->token()));
  BValue value =
      builder_ptr->ReceiveNonBlocking(std::get<Channel*>(ir_value), token);
  BValue token_value = builder_ptr->TupleIndex(value, 0);
  tokens_.push_back(token_value);
  node_to_ir_[node] = value;

  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleRecvIf(const RecvIf* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->channel()));
  if (!node_to_ir_.contains(node->channel())) {
    return absl::InternalError("Recv channel not found!");
  }
  IrValue ir_value = node_to_ir_[node->channel()];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->condition()));
  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->token()));
  XLS_ASSIGN_OR_RETURN(BValue predicate, Use(node->condition()));
  BValue value =
      builder_ptr->ReceiveIf(std::get<Channel*>(ir_value), token, predicate);
  BValue token_value = builder_ptr->TupleIndex(value, 0);
  tokens_.push_back(token_value);
  node_to_ir_[node] = value;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleRecvIfNonBlocking(
    const RecvIfNonBlocking* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Recv nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->token()));
  XLS_RETURN_IF_ERROR(Visit(node->channel()));
  if (!node_to_ir_.contains(node->channel())) {
    return absl::InternalError("Recv channel not found!");
  }
  IrValue ir_value = node_to_ir_[node->channel()];
  if (!std::holds_alternative<Channel*>(ir_value)) {
    return absl::InvalidArgumentError(
        "Expected channel, got BValue or CValue.");
  }

  XLS_RETURN_IF_ERROR(Visit(node->condition()));
  XLS_ASSIGN_OR_RETURN(BValue token, Use(node->token()));
  XLS_ASSIGN_OR_RETURN(BValue predicate, Use(node->condition()));
  BValue value = builder_ptr->ReceiveIfNonBlocking(std::get<Channel*>(ir_value),
                                                   token, predicate);
  BValue token_value = builder_ptr->TupleIndex(value, 0);
  tokens_.push_back(token_value);
  node_to_ir_[node] = value;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleJoin(const Join* node) {
  ProcBuilder* builder_ptr =
      dynamic_cast<ProcBuilder*>(function_builder_.get());
  if (builder_ptr == nullptr) {
    return absl::InternalError(
        "Join nodes should only be encountered during Proc conversion; "
        "we seem to be in function conversion.");
  }

  std::vector<BValue> ir_tokens;
  ir_tokens.reserve(node->tokens().size());
  for (auto* token : node->tokens()) {
    XLS_RETURN_IF_ERROR(Visit(token));
    XLS_ASSIGN_OR_RETURN(BValue ir_token, Use(token));
    ir_tokens.push_back(ir_token);
  }
  BValue value = builder_ptr->AfterAll(ir_tokens);
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
    xls::Function* implicit_token_f, dslx::Function* dslx_function,
    bool is_top) {
  XLS_RET_CHECK_GE(implicit_token_f->params().size(), 2);
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(dslx_function->owner()->name(),
                     dslx_function->identifier(), CallingConvention::kTypical,
                     /*free_keys=*/{}, /*symbolic_bindings=*/nullptr));
  FunctionBuilder fb(mangled_name, implicit_token_f->package());
  // Entry is a top entity.
  if (is_top) {
    XLS_RETURN_IF_ERROR(fb.SetAsTop());
  }
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

// Tries four heuristics as names for potential entry functions of the package.
// Returns the first found heuristic. Otherwise, returns an absl::NotFoundError.
static absl::StatusOr<xls::Function*> GetEntryFunction(xls::Package* package) {
  constexpr char kMain[] = "main";
  // Try a few possibilities of names for the canonical entry function.
  const std::vector<std::string> to_try = {
      kMain,
      package->name(),
      absl::StrCat("__", package->name(), "__", kMain),
      absl::StrCat("__", package->name(), "__", package->name()),
  };

  for (const std::string& attempt : to_try) {
    auto func_or = package->GetFunction(attempt);
    if (func_or.ok()) {
      return func_or.value();
    }
  }

  std::vector<xls::Function*> functions;
  // Finally we use the only function if only one exists.
  for (xls::FunctionBase* fb : package->GetFunctionBases()) {
    if (fb->IsFunction()) {
      functions.emplace_back(fb->AsFunctionOrDie());
    }
  }
  if (functions.size() == 1) {
    return functions.front();
  }

  return absl::NotFoundError(absl::StrFormat(
      "Could not find an entry function for the \"%s\" package.",
      package->name()));
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
      GetEntryFunction(package_data.package);
  if (!entry_or.ok()) {  // Entry point not found.
    XLS_RET_CHECK_EQ(entry_or.status().code(), absl::StatusCode::kNotFound);
    return absl::OkStatus();
  }

  xls::Function* entry = entry_or.value();
  if (package_data.wrappers.contains(entry)) {
    // Already created!
    return absl::OkStatus();
  }

  dslx::Function* dslx_entry =
      dynamic_cast<Function*>(package_data.ir_to_dslx.at(entry));
  if (GetRequiresImplicitToken(dslx_entry, import_data, options)) {
    // Only create implicit token wrapper.
    return EmitImplicitTokenEntryWrapper(entry, dslx_entry, /*is_top=*/false)
        .status();
  }
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleFunction(
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
  std::string fn_name = node->identifier();
  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(module_->name(), node->identifier(),
                     requires_implicit_token ? CallingConvention::kImplicitToken
                                             : CallingConvention::kTypical,
                     node->GetFreeParametricKeySet(), symbolic_bindings));
  auto builder = std::make_unique<FunctionBuilder>(mangled_name, package());
  auto* builder_ptr = builder.get();
  SetFunctionBuilder(std::move(builder));
  // Function is a top entity.
  if (is_top_ && !requires_implicit_token) {
    XLS_RETURN_IF_ERROR(builder_ptr->SetAsTop());
  }

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

    std::optional<InterpValue> sb_value =
        get_symbolic_binding(parametric_binding->identifier());
    XLS_RET_CHECK(sb_value.has_value());
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> parametric_type,
                         ResolveType(parametric_binding->type_annotation()));
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim parametric_width_ctd,
                         parametric_type->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(Value param_value, InterpValueToValue(*sb_value));
    DefConst(parametric_binding, param_value);
    XLS_RETURN_IF_ERROR(
        DefAlias(parametric_binding, /*to=*/parametric_binding->name_def()));
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
    // This set may be empty if "emit_fail_as_assert" is false.
    BValue join_token =
        function_builder_->AfterAll(implicit_token_data_->control_tokens);
    std::vector<BValue> elements = {join_token, return_value};
    return_value = function_builder_->Tuple(std::move(elements));
  }

  XLS_ASSIGN_OR_RETURN(xls::Function * f,
                       builder_ptr->BuildWithReturnValue(return_value));
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
  if (requires_implicit_token && (node->is_public() || is_top_) &&
      !node->IsParametric()) {
    XLS_ASSIGN_OR_RETURN(xls::Function * wrapper,
                         EmitImplicitTokenEntryWrapper(f, node, is_top_));
    package_data_.wrappers.insert(wrapper);
  }

  package_data_.ir_to_dslx[f] = node;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleProcNextFunction(
    Function* f, const Invocation* invocation, TypeInfo* type_info,
    ImportData* import_data, const SymbolicBindings* symbolic_bindings,
    const ProcId& proc_id, ProcConversionData* proc_data) {
  XLS_RET_CHECK(type_info != nullptr);
  XLS_VLOG(5) << "HandleProcNextFunction: " << f->ToString();

  if (symbolic_bindings != nullptr) {
    SetSymbolicBindings(symbolic_bindings);
  }

  ScopedTypeInfoSwap stis(this, type_info);

  XLS_ASSIGN_OR_RETURN(
      std::string mangled_name,
      MangleDslxName(module_->name(), proc_id.ToString(),
                     CallingConvention::kProcNext, f->GetFreeParametricKeySet(),
                     symbolic_bindings));
  std::string token_name = "__token";
  std::string state_name = "__state";
  // If this function is parametric, then the constant args will be defined in
  // the _parent_ type info.
  if (type_info->parent() != nullptr) {
    type_info = type_info->parent();
  }

  Value initial_element = Value::Tuple({});
  if (proc_data_->id_to_initial_value.contains(proc_id)) {
    initial_element = proc_data_->id_to_initial_value.at(proc_id);
  }

  auto builder =
      std::make_unique<ProcBuilder>(mangled_name, token_name, package());
  builder->StateElement(state_name, initial_element);
  tokens_.push_back(builder->GetTokenParam());
  SetNodeToIr(f->params()[0]->name_def(), builder->GetTokenParam());
  auto* builder_ptr = builder.get();
  SetFunctionBuilder(std::move(builder));
  // Proc is a top entity.
  if (is_top_) {
    XLS_RETURN_IF_ERROR(builder_ptr->SetAsTop());
  }

  // "next" functions actually have _explicit_ tokens. We can unconditionally
  // hook into the implicit token stuff in case any downstream functions need
  // it.
  implicit_token_data_.emplace();
  implicit_token_data_->entry_token = builder_ptr->GetTokenParam();
  implicit_token_data_->activated = builder_ptr->Literal(Value::Bool(true));
  implicit_token_data_->create_control_predicate = [this]() {
    return implicit_token_data_->activated;
  };

  // Now bind the recurrent state element.
  // We need to shift indices by one to handle the token arg.
  if (f->params().size() == 2) {
    BValue state = builder_ptr->GetStateParam(0);
    SetNodeToIr(f->params()[1]->name_def(), state);
  }

  proc_id_ = proc_id;

  for (ParametricBinding* parametric_binding : f->parametric_bindings()) {
    XLS_VLOG(5) << "Resolving parametric binding: "
                << parametric_binding->ToString();

    std::optional<InterpValue> sb_value =
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
        DefAlias(parametric_binding, /*to=*/parametric_binding->name_def()));
  }

  XLS_VLOG(3) << "Proc has " << constant_deps_.size() << " constant deps";
  for (ConstantDef* dep : constant_deps_) {
    XLS_VLOG(5) << "Visiting constant dep: " << dep->ToString();
    XLS_RETURN_IF_ERROR(Visit(dep));
  }

  XLS_RETURN_IF_ERROR(Visit(f->body()));

  BValue result = std::get<BValue>(node_to_ir_[f->body()]);
  BValue final_token = builder_ptr->AfterAll(tokens_);
  XLS_ASSIGN_OR_RETURN(xls::Proc * p,
                       builder_ptr->Build(final_token, {result}));
  package_data_.ir_to_dslx[p] = f;
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleColonRef(const ColonRef* node) {
  // Implementation note: ColonRef "invocations" are handled in Invocation (by
  // resolving the mangled callee name, which should have been IR converted in
  // dependency order).
  if (std::optional<Import*> import = node->ResolveImportSubject()) {
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

  XLS_ASSIGN_OR_RETURN(
      auto subject,
      ResolveColonRefSubject(import_data_, current_type_info_, node));
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
          }},
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

  Def(node, [this, &members](const SourceInfo& loc) {
    return function_builder_->Tuple(std::move(members), loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleStructInstance(
    const StructInstance* node) {
  std::vector<BValue> operands;
  XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                       DerefStruct(ToTypeDefinition(node->struct_def())));
  std::vector<Value> const_operands;
  for (auto [_, member_expr] : node->GetOrderedMembers(struct_def)) {
    XLS_RETURN_IF_ERROR(Visit(member_expr));
    XLS_ASSIGN_OR_RETURN(BValue operand, Use(member_expr));
    operands.push_back(operand);
  }

  Def(node, [this, &operands](const SourceInfo& loc) {
    return function_builder_->Tuple(std::move(operands), loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<std::string> FunctionConverter::GetCalleeIdentifier(
    const Invocation* node) {
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

  // We have to mangle the symbolic bindings into the name to get the fully
  // resolved symbol.
  absl::btree_set<std::string> free_keys = f->GetFreeParametricKeySet();
  const CallingConvention convention = GetCallingConvention(f);
  if (!f->IsParametric()) {
    return MangleDslxName(m->name(), f->identifier(), convention, free_keys);
  }

  std::optional<const SymbolicBindings*> resolved_symbolic_bindings =
      GetInvocationCalleeBindings(node);
  XLS_RET_CHECK(resolved_symbolic_bindings.has_value());
  XLS_VLOG(5) << absl::StreamFormat("Node `%s` (%s) @ %s symbolic bindings %s",
                                    node->ToString(), node->GetNodeTypeName(),
                                    node->span().ToString(),
                                    (*resolved_symbolic_bindings)->ToString());
  XLS_RET_CHECK(!(*resolved_symbolic_bindings)->empty());
  return MangleDslxName(m->name(), f->identifier(), convention, free_keys,
                        resolved_symbolic_bindings.value());
}

absl::Status FunctionConverter::HandleBinop(const Binop* node) {
  XLS_VLOG(5) << "HandleBinop: " << node->ToString();
  std::optional<const ConcreteType*> lhs_type =
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
  XLS_VLOG(5) << "FunctionConverter::HandleAttr: " << node->ToString() << " @ "
              << node->span().ToString();
  XLS_RETURN_IF_ERROR(Visit(node->lhs()));
  std::optional<const ConcreteType*> lhs_type =
      current_type_info_->GetItem(node->lhs());
  XLS_RET_CHECK(lhs_type.has_value());
  auto* struct_type = dynamic_cast<const StructType*>(lhs_type.value());
  const std::string& identifier = node->attr()->identifier();
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
  XLS_RETURN_IF_ERROR(Visit(node->body()));
  XLS_ASSIGN_OR_RETURN(BValue bvalue, Use(node->body()));
  SetNodeToIr(node, bvalue);
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleTernary(const Ternary* node) {
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

absl::Status FunctionConverter::HandleBuiltinArraySlice(
    const Invocation* node) {
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

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->ArraySlice(arg, start, width, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinBitSlice(const Invocation* node) {
  XLS_RET_CHECK_EQ(node->args().size(), 3);
  XLS_ASSIGN_OR_RETURN(BValue arg, Use(node->args()[0]));
  XLS_ASSIGN_OR_RETURN(Bits start_bits, GetConstBits(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(uint64_t start, start_bits.ToUint64());
  XLS_ASSIGN_OR_RETURN(Bits width_bits, GetConstBits(node->args()[2]));
  XLS_ASSIGN_OR_RETURN(uint64_t width, width_bits.ToUint64());
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->BitSlice(arg, start, width, loc);
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

  const Expr* cases_arg = node->args()[1];
  std::vector<BValue> cases;
  const auto* array = dynamic_cast<const Array*>(cases_arg);
  XLS_RET_CHECK_NE(array, nullptr);
  for (const auto& sel_case : array->members()) {
    XLS_ASSIGN_OR_RETURN(BValue bvalue_case, Use(sel_case));
    cases.push_back(bvalue_case);
  }

  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->OneHotSelect(selector, cases, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::HandleBuiltinPrioritySel(
    const Invocation* node) {
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

  Def(node, [&](const SourceInfo& loc) {
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

  Def(node, [&](const SourceInfo& loc) {
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
  XLS_ASSIGN_OR_RETURN(BValue index, Use(node->args()[1]));
  XLS_ASSIGN_OR_RETURN(BValue new_value, Use(node->args()[2]));
  Def(node, [&](const SourceInfo& loc) {
    return function_builder_->ArrayUpdate(arg, new_value, {index}, loc);
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
  Def(node, [this, &slices, element_type](const SourceInfo& loc) {
    return function_builder_->Array(std::move(slices), element_type, loc);
  });
  return absl::OkStatus();
}

absl::Status FunctionConverter::CastFromArray(const Cast* node,
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
  Def(node, [this, &pieces](const SourceInfo& loc) {
    return function_builder_->Concat(std::move(pieces), loc);
  });
  return absl::OkStatus();
}

absl::StatusOr<FunctionConverter::DerefVariant>
FunctionConverter::DerefStructOrEnum(TypeDefinition node) {
  while (std::holds_alternative<TypeDef*>(node)) {
    auto* type_def = std::get<TypeDef*>(node);
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

absl::StatusOr<std::unique_ptr<ConcreteType>> FunctionConverter::ResolveType(
    const AstNode* node) {
  XLS_RET_CHECK(current_type_info_ != nullptr);
  std::optional<const ConcreteType*> t = current_type_info_->GetItem(node);
  if (!t.has_value()) {
    return ConversionErrorStatus(
        node->GetSpan(),
        absl::StrFormat(
            "Failed to convert IR because type was missing for AST node: %s",
            node->ToString()));
  }

  return t.value()->MapSize([this](ConcreteTypeDim dim) {
    return ResolveDim(dim, SymbolicBindings(symbolic_binding_map_));
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

absl::StatusOr<xls::Type*> FunctionConverter::ResolveTypeToIr(AstNode* node) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> concrete_type,
                       ResolveType(node));
  return TypeToIr(package_data_.package, *concrete_type,
                  SymbolicBindings(symbolic_binding_map_));
}

absl::Status ConvertOneFunctionInternal(PackageData& package_data,
                                        const ConversionRecord& record,
                                        ImportData* import_data,
                                        ProcConversionData* proc_data,
                                        const ConvertOptions& options) {
  // Validate the requested conversion looks sound in terms of provided
  // parametrics.
  XLS_RETURN_IF_ERROR(ConversionRecord::ValidateParametrics(
      record.f(), record.symbolic_bindings()));

  FunctionConverter converter(package_data, record.module(), import_data,
                              options, proc_data, record.IsTop());
  XLS_ASSIGN_OR_RETURN(auto constant_deps,
                       GetConstantDepFreevars(record.f()->body()));
  for (const auto& dep : constant_deps) {
    converter.AddConstantDep(dep);
  }

  Function* f = record.f();
  if (f->tag() == Function::Tag::kProcConfig) {
    // TODO(rspringer): 2021-09-29: Probably need to pass constants in here.
    ProcConfigIrConverter converter(
        package_data.package, f, record.type_info(), import_data, proc_data,
        record.symbolic_bindings(), record.proc_id().value());
    XLS_RETURN_IF_ERROR(f->Accept(&converter));
    XLS_RETURN_IF_ERROR(converter.Finalize());
    return absl::OkStatus();
  }

  if (f->tag() == Function::Tag::kProcNext) {
    if (!proc_data->id_to_initial_value.contains(record.proc_id().value())) {
      Proc* p = f->proc().value();
      XLS_ASSIGN_OR_RETURN(
          ConcreteType * foo,
          record.type_info()->GetItemOrError(p->init()->body()));

      // If there's no value in the map, then this should be a top-level proc.
      // Verify that there are no symbolic bindings.
      XLS_RET_CHECK(record.symbolic_bindings().empty());
      XLS_ASSIGN_OR_RETURN(InterpValue iv, ConstexprEvaluator::EvaluateToValue(
                                               import_data, record.type_info(),
                                               record.symbolic_bindings(),
                                               p->init()->body(), foo));
      XLS_ASSIGN_OR_RETURN(Value ir_value, InterpValueToValue(iv));
      proc_data->id_to_initial_value[record.proc_id().value()] = ir_value;
    }

    return converter.HandleProcNextFunction(
        f, record.invocation(), record.type_info(), import_data,
        &record.symbolic_bindings(), record.proc_id().value(), proc_data);
  }

  return converter.HandleFunction(f, record.type_info(),
                                  &record.symbolic_bindings());
}

// Creates the recv- and send-only channels needed for the top-level proc - it
// needs to communicate with the outside world somehow.
absl::Status CreateBoundaryChannels(absl::Span<Param* const> params,
                                    const ProcId& proc_id, TypeInfo* type_info,
                                    const PackageData& package_data,
                                    ProcConversionData* proc_data) {
  for (const Param* param : params) {
    TypeAnnotation* type = param->type_annotation();
    if (auto* channel_type = dynamic_cast<ChannelTypeAnnotation*>(type);
        type != nullptr) {
      auto maybe_type = type_info->GetItem(channel_type->payload());
      XLS_RET_CHECK(maybe_type.has_value());
      ConcreteType* ct = maybe_type.value();
      XLS_ASSIGN_OR_RETURN(Type * ir_type, TypeToIr(package_data.package, *ct,
                                                    SymbolicBindings()));
      ChannelOps op = channel_type->direction() == ChannelTypeAnnotation::kIn
                          ? ChannelOps::kReceiveOnly
                          : ChannelOps::kSendOnly;
      std::string channel_name =
          absl::StrCat(package_data.package->name(), "__", param->identifier());
      XLS_ASSIGN_OR_RETURN(StreamingChannel * channel,
                           package_data.package->CreateStreamingChannel(
                               channel_name, op, ir_type));
      proc_data->id_to_config_args[proc_id].push_back(channel);
    }
  }
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
                     order, "\n\t",
                     [](std::string* out, const ConversionRecord& record) {
                       absl::StrAppend(out, record.ToString());
                     })
              << "]";
  // We need to convert Functions before procs: Channels are declared inside
  // Functions, but exist as "global" entities in the IR. By processing
  // Functions first, we can collect the declarations of these global data so we
  // can resolve them when processing Procs. GetOrder() (in
  // extract_conversion_order.h) handles evaluation ordering. In addition, for
  // procs, the config block must be evaluated before the next block, as config
  // sets channels and constants needed by next. This is handled inside
  // ConvertOneFunctionInternal().
  ProcConversionData proc_data;

  // The top-level proc's input/output channels need to come from _somewhere_.
  // At conversion time, though, we won't have that info. To enable forward
  // progress, we collect any channel args to the top-level proc and declare
  // them as send- or recv-only, as appropriate.
  const ConversionRecord* first_proc_config = nullptr;
  const ConversionRecord* first_proc_next = nullptr;
  Proc* top_proc = nullptr;
  // Only "next" functions are marked as "top", but we need to treat "config"
  // functions specially too, so we need to find the associated proc.
  for (const auto& record : order) {
    if (record.IsTop() && record.f()->tag() == Function::Tag::kProcNext) {
      top_proc = record.f()->proc().value();
    }
  }

  // If there's no top proc, then we'll just do our best & pick the first one.
  if (top_proc == nullptr) {
    for (const auto& record : order) {
      if (record.f()->tag() == Function::Tag::kProcConfig) {
        top_proc = record.f()->proc().value();
        break;
      }
    }
  }

  for (const auto& record : order) {
    if (record.f()->tag() == Function::Tag::kProcConfig &&
        record.f()->proc().value() == top_proc) {
      first_proc_config = &record;
    } else if (record.IsTop() &&
               record.f()->tag() == Function::Tag::kProcNext) {
      first_proc_next = &record;
    }

    if (first_proc_config != nullptr && first_proc_next != nullptr) {
      break;
    }
  }

  // Set first proc's initial value.
  if (first_proc_config != nullptr) {
    XLS_RET_CHECK(first_proc_config->proc_id().has_value());
    ProcId proc_id = first_proc_config->proc_id().value();
    XLS_RETURN_IF_ERROR(CreateBoundaryChannels(
        first_proc_config->f()->params(), proc_id,
        first_proc_config->type_info(), package_data, &proc_data));
  }

  for (const ConversionRecord& record : order) {
    XLS_VLOG(3) << "Converting to IR: " << record.ToString();
    XLS_RETURN_IF_ERROR(ConvertOneFunctionInternal(
        package_data, record, import_data, &proc_data, options));
  }

  XLS_VLOG(3) << "Verifying converted package";
  if (options.verify_ir) {
    XLS_RETURN_IF_ERROR(VerifyPackage(package_data.package));
  }
  return absl::OkStatus();
}

absl::Status ConvertModuleIntoPackage(Module* module, ImportData* import_data,
                                      const ConvertOptions& options,
                                      bool traverse_tests, Package* package) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * root_type_info,
                       import_data->GetRootTypeInfo(module));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrder(module, root_type_info, traverse_tests));
  PackageData package_data{package};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));

  XLS_RETURN_IF_ERROR(
      WrapEntryIfImplicitToken(package_data, import_data, options));

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Package>> ConvertModuleToPackage(
    Module* module, ImportData* import_data, const ConvertOptions& options,
    bool traverse_tests) {
  auto package = std::make_unique<Package>(module->name());
  XLS_RETURN_IF_ERROR(ConvertModuleIntoPackage(module, import_data, options,
                                               traverse_tests, package.get()));
  return package;
}

absl::StatusOr<std::string> ConvertModule(Module* module,
                                          ImportData* import_data,
                                          const ConvertOptions& options) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ConvertModuleToPackage(module, import_data, options));
  return package->DumpIr();
}

template <typename BlockT>
absl::Status ConvertOneFunctionIntoPackageInternal(
    Module* module, BlockT* block, ImportData* import_data,
    const SymbolicBindings* symbolic_bindings, const ConvertOptions& options,
    Package* package) {
  XLS_ASSIGN_OR_RETURN(TypeInfo * func_type_info,
                       import_data->GetRootTypeInfoForNode(block));
  XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> order,
                       GetOrderForEntry(block, func_type_info));
  PackageData package_data{package};
  XLS_RETURN_IF_ERROR(
      ConvertCallGraph(order, import_data, options, package_data));
  return absl::OkStatus();
}

absl::Status ConvertOneFunctionIntoPackage(
    Module* module, std::string_view entry_function_name,
    ImportData* import_data, const SymbolicBindings* symbolic_bindings,
    const ConvertOptions& options, Package* package) {
  std::optional<Function*> fn_or = module->GetFunction(entry_function_name);
  if (fn_or.has_value()) {
    return ConvertOneFunctionIntoPackageInternal(module, fn_or.value(),
                                                 import_data, symbolic_bindings,
                                                 options, package);
  }

  auto proc_or = module->GetMemberOrError<Proc>(entry_function_name);
  if (proc_or.ok()) {
    return ConvertOneFunctionIntoPackageInternal(module, proc_or.value(),
                                                 import_data, symbolic_bindings,
                                                 options, package);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Entry \"%s\" is not present in "
                      "DSLX module %s as a Function or a Proc.",
                      entry_function_name, module->name()));
}

absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, std::string_view entry_function_name,
    ImportData* import_data, const SymbolicBindings* symbolic_bindings,
    const ConvertOptions& options) {
  Package package(module->name());
  XLS_RETURN_IF_ERROR(
      ConvertOneFunctionIntoPackage(module, entry_function_name, import_data,
                                    symbolic_bindings, options, &package));
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

}  // namespace xls::dslx
