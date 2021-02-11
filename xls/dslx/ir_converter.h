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

#ifndef XLS_DSLX_IR_CONVERTER_H_
#define XLS_DSLX_IR_CONVERTER_H_

#include <memory>

#include "absl/container/btree_set.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_info.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls::dslx {
namespace internal {

// Helper type that creates XLS IR in a Package, given a DSLX AST, its type
// information, and an entry point function.
//
// TODO(leary): 2020-12-22 Conceptually this class walks the AST and converts
// its operations into XLS IR in a fairly simple way (though it needs to resolve
// parametrics and such). Right now this mostly holds the state required, and
// most of the functionality is in Python ir_converter.py, but when it is fully
// ported over this class will be responsible for everything -- this allows us
// to port incrementally.
class IrConverter {
 public:
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

  IrConverter(Package* package, Module* module, TypeInfo* type_info,
              ImportCache* import_cache, bool emit_positions);

  ~IrConverter() { XLS_VLOG(5) << "Destroying IR converter: " << this; }

  // Main entry point to request conversion of the DSLX function "f" to an IR
  // function.
  absl::StatusOr<xls::Function*> VisitFunction(
      Function* f, const SymbolicBindings* symbolic_bindings = nullptr);

  void InstantiateFunctionBuilder(absl::string_view mangled_name);

  // Notes a constant-definition dependency for the converted IR.
  void AddConstantDep(ConstantDef* constant_def);
  void ClearConstantDeps() { constant_deps_.clear(); }

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
  // expected int64.
  absl::StatusOr<int64> ResolveDimToInt(const ConcreteTypeDim& dim) {
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim resolved, ResolveDim(dim));
    if (absl::holds_alternative<int64>(resolved.value())) {
      return absl::get<int64>(resolved.value());
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

  const std::vector<ConstantDef*>& constant_deps() const {
    return constant_deps_;
  }

  void SetSymbolicBindings(const SymbolicBindings* value) {
    if (value == nullptr) {
      clear_symbolic_binding_map();
    } else {
      set_symbolic_binding_map(value->ToMap());
    }
  }

  // Gets the current counter of counted_for loops we've observed and bumps it.
  // This is useful for generating new symbols for the functions that serve as
  // XLS counted_for "bodies".
  int64 GetAndBumpCountedForCount() { return counted_for_count_++; }

  // TODO(leary): 2020-12-22 Clean all this up to expose a minimal surface area
  // once everything is ported over to C++.
  absl::optional<int64> get_symbolic_binding(
      absl::string_view identifier) const {
    auto it = symbolic_binding_map_.find(identifier);
    if (it == symbolic_binding_map_.end()) {
      return absl::nullopt;
    }
    return it->second;
  }
  void set_symbolic_binding(std::string symbol, int64 value) {
    symbolic_binding_map_[symbol] = value;
  }
  void set_symbolic_binding_map(absl::flat_hash_map<std::string, int64> map) {
    symbolic_binding_map_ = std::move(map);
  }
  const absl::flat_hash_map<std::string, int64>& symbolic_binding_map() const {
    return symbolic_binding_map_;
  }
  void clear_symbolic_binding_map() { symbolic_binding_map_.clear(); }
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
  absl::optional<const SymbolicBindings*> GetInvocationBindings(
      Invocation* invocation) const {
    SymbolicBindings key = GetSymbolicBindingsTuple();
    return type_info_->GetInvocationSymbolicBindings(invocation, key);
  }

  const Fileno& fileno() const { return fileno_; }

  // AstNode handlers.
  absl::Status HandleBinop(Binop* node);
  absl::Status HandleConstRef(ConstRef* node);
  absl::Status HandleNameRef(NameRef* node);
  absl::Status HandleNumber(Number* node);
  absl::Status HandleParam(Param* node);
  absl::Status HandleTernary(Ternary* node);
  absl::Status HandleUnop(Unop* node);
  absl::Status HandleXlsTuple(XlsTuple* node);

  absl::Status HandleConcat(Binop* node, BValue lhs, BValue rhs);

  // Callback signature to visit a node for IR conversion (e.g. used in handlers
  // that employ a custom visitation order).
  using VisitFunc = std::function<absl::Status(AstNode*)>;

  // Callback signature to visit a node for IR conversion given an IR converter.
  // This is used, for example, when building a for loop's body function.
  using VisitIrConverterFunc =
      std::function<absl::Status(IrConverter*, AstNode*)>;

  // AstNode handlers that recur "manually" internal to the handler (via the
  // "visit" callback).
  absl::Status HandleAttr(Attr* node, const VisitFunc& visit);
  absl::Status HandleSplatStructInstance(SplatStructInstance* node,
                                         const VisitFunc& visit);
  absl::Status HandleStructInstance(StructInstance* node,
                                    const VisitFunc& visit);
  absl::Status HandleColonRef(ColonRef* node, const VisitFunc& visit);
  absl::Status HandleConstantDef(ConstantDef* node, const VisitFunc& visit);
  absl::Status HandleLet(Let* node, const VisitFunc& visit);
  absl::Status HandleCast(Cast* node, const VisitFunc& visit);
  absl::Status HandleMatch(Match* node, const VisitFunc& visit);
  absl::Status HandleIndex(Index* node, const VisitFunc& visit);
  absl::Status HandleConstantArray(ConstantArray* node, const VisitFunc& visit);
  absl::Status HandleArray(Array* node, const VisitFunc& visit);
  absl::Status HandleInvocation(Invocation* node, const VisitFunc& visit);

  absl::StatusOr<xls::Function*> HandleFunction(
      Function* node, const SymbolicBindings* symbolic_bindings,
      const VisitFunc& visit);

  absl::Status HandleFor(For* node, const VisitFunc& visit,
                         const VisitIrConverterFunc& visit_converter);

  // Handles an arm of a match expression.
  absl::StatusOr<BValue> HandleMatcher(NameDefTree* matcher,
                                       absl::Span<const int64> index,
                                       const BValue& matched_value,
                                       const ConcreteType& matched_type,
                                       const VisitFunc& visit);

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

  absl::StatusOr<BValue> HandleMap(Invocation* node, const VisitFunc& visit);

  // Builtin invocation handlers.
  absl::Status HandleBuiltinAndReduce(Invocation* node);
  absl::Status HandleBuiltinBitSlice(Invocation* node);
  absl::Status HandleBuiltinBitSliceUpdate(Invocation* node);
  absl::Status HandleBuiltinClz(Invocation* node);
  absl::Status HandleBuiltinCtz(Invocation* node);
  absl::Status HandleBuiltinOneHot(Invocation* node);
  absl::Status HandleBuiltinOneHotSel(Invocation* node);
  absl::Status HandleBuiltinOrReduce(Invocation* node);
  absl::Status HandleBuiltinRev(Invocation* node);
  absl::Status HandleBuiltinSignex(Invocation* node);
  absl::Status HandleBuiltinUpdate(Invocation* node);
  absl::Status HandleBuiltinXorReduce(Invocation* node);
  absl::Status HandleBuiltinScmp(SignedCmp cmp, Invocation* node);

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

  absl::StatusOr<std::string> GetCalleeIdentifier(Invocation* node);

 private:
  // Determines whether the for loop node is of the general form:
  //
  //  `for ... in range(0, N)`
  //
  // Returns the value of N if so, or a conversion error if it is not.
  absl::StatusOr<int64> QueryConstRangeCall(For* node, const VisitFunc& visit);

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
    if (!emit_positions_ || !span.has_value()) {
      return absl::nullopt;
    }
    const Pos& start_pos = span->start();
    Lineno lineno(start_pos.lineno());
    Colno colno(start_pos.colno());
    // TODO(leary): 2020-12-20 Figure out the fileno based on the module owner
    // of node.
    return SourceLocation{fileno_, lineno, colno};
  }

  // Defines "node" to map the the result of running "ir_func" with "args" -- if
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

  // Package that IR is being generated into.
  Package* package_;

  // Module that contains the entry point function being converted.
  Module* module_;

  // Type information for this IR conversion (determined by the type inference
  // phase).
  TypeInfo* type_info_;

  ImportCache* import_cache_;

  // Whether or not to emit source code positions into the XLS IR.
  //
  // Stripping positions can be useful for less fragile string matching in
  // development, e.g. tests.
  bool emit_positions_;

  // Mapping from AST node to its corresponding IR value.
  absl::flat_hash_map<AstNode*, IrValue> node_to_ir_;

  // Constants that this translation depends upon (as determined externally).
  std::vector<ConstantDef*> constant_deps_;

  // Function builder being used to create BValues.
  absl::optional<FunctionBuilder> function_builder_;

  // Mapping of symbolic bindings active in this translation (e.g. what integral
  // values parametrics are taking on).
  absl::flat_hash_map<std::string, int64> symbolic_binding_map_;

  // File number for use in source positions.
  Fileno fileno_;

  // Number of "counted for" nodes we've observed in this function.
  int64 counted_for_count_ = 0;

  // The last expression emitted as part of IR conversion, used to help
  // determine which expression produces the return value.
  Expr* last_expression_ = nullptr;
};

// Returns a status that indicates an error in the IR conversion process.
absl::Status ConversionErrorStatus(const absl::optional<Span>& span,
                                   absl::string_view message);

}  // namespace internal

// Returns the mangled name of function with the given parametric bindings.
absl::StatusOr<std::string> MangleDslxName(
    absl::string_view function_name,
    const absl::btree_set<std::string>& free_keys, Module* module,
    const SymbolicBindings* symbolic_bindings = nullptr);

// Converts the contents of a module to IR form.
//
// Args:
//   module: Module to convert.
//   type_info: Concrete type information used in conversion.
//   import_cache: Cache of modules potentially referenced by "module" above.
//   emit_positions: Whether to emit positional metadata into the output IR.
//   traverse_tests: Whether to convert functions called in DSLX test
//   constructs.
//     Note that this does NOT convert the test constructs themselves.
//
// Returns:
//   The IR package that corresponds to this module.
absl::StatusOr<std::unique_ptr<xls::Package>> ConvertModuleToPackage(
    Module* module, TypeInfo* type_info, ImportCache* import_cache,
    bool emit_positions = true, bool traverse_tests = false);

// Wrapper around ConvertModuleToPackage that converts to IR text.
absl::StatusOr<std::string> ConvertModule(Module* module, TypeInfo* type_info,
                                          ImportCache* import_cache,
                                          bool emit_positions = true);

// As with internal::ConvertOneFunction above, but takes an entry_function_name
// and creates a temporary IR package based on module's name.
absl::StatusOr<std::string> ConvertOneFunction(
    Module* module, absl::string_view entry_function_name, TypeInfo* type_info,
    ImportCache* import_cache, const SymbolicBindings* symbolic_bindings,
    bool emit_positions);

// Converts an interpreter value to an IR value.
absl::StatusOr<Value> InterpValueToValue(const InterpValue& v);

// Converts an (IR) value to an interpreter value.
absl::StatusOr<InterpValue> ValueToInterpValue(
    const Value& v, const ConcreteType* type = nullptr);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERTER_H_
