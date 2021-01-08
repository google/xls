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
#include "xls/dslx/cpp_ast.h"
#include "xls/dslx/type_info.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls::dslx {

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

  IrConverter(const std::shared_ptr<Package>& package, Module* module,
              const std::shared_ptr<TypeInfo>& type_info, bool emit_positions);

  ~IrConverter() { XLS_VLOG(5) << "Destroying IR converter: " << this; }

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

  // Returns the constant bits corresponding to the IrValue of "node", or
  // returns an error if it is not present (or not constant).
  absl::StatusOr<Bits> GetConstBits(AstNode* node) const;

  // Resolves "dim" (from a possible parametric) against the
  // symbolic_binding_map_.
  absl::StatusOr<ConcreteTypeDim> ResolveDim(ConcreteTypeDim dim);

  // Resolves node's type and resolves all of its dimensions via `ResolveDim()`.
  absl::StatusOr<std::unique_ptr<ConcreteType>> ResolveType(AstNode* node);

  // Helper that composes ResolveType() and TypeToIr().
  absl::StatusOr<xls::Type*> ResolveTypeToIr(AstNode* node);

  // -- Accessors

  // IR package being populated with IR entities as part of the conversion.
  const std::shared_ptr<Package>& package() const { return package_; }

  // The AST module wherein things are being converted to IR.
  Module* module() const { return module_; }

  // Type information for this IR conversion process (e.g. that describes the
  // nodes in module()).
  const std::shared_ptr<TypeInfo>& type_info() const { return type_info_; }

  bool emit_positions() const { return emit_positions_; }

  const std::shared_ptr<FunctionBuilder>& function_builder() const {
    return function_builder_;
  }

  Expr* last_expression() const { return last_expression_; }
  void set_last_expression(Expr* e) {
    XLS_CHECK_EQ(e->owner(), module_);
    last_expression_ = e;
  }

  const std::vector<ConstantDef*>& constant_deps() const {
    return constant_deps_;
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

  const Fileno& fileno() const { return fileno_; }

  // AstNode handlers.
  absl::Status HandleUnop(Unop* node);
  absl::Status HandleBinop(Binop* node);
  absl::Status HandleAttr(Attr* node);
  absl::Status HandleTernary(Ternary* node);
  absl::Status HandleConstantArray(ConstantArray* node);
  absl::Status HandleConcat(Binop* node, BValue lhs, BValue rhs);

  // Builtin invocation handlers.
  absl::Status HandleBuiltinAndReduce(Invocation* node);
  absl::Status HandleBuiltinBitSlice(Invocation* node);
  absl::Status HandleBuiltinClz(Invocation* node);
  absl::Status HandleBuiltinCtz(Invocation* node);
  absl::Status HandleBuiltinOneHot(Invocation* node);
  absl::Status HandleBuiltinOneHotSel(Invocation* node);
  absl::Status HandleBuiltinOrReduce(Invocation* node);
  absl::Status HandleBuiltinRev(Invocation* node);
  absl::Status HandleBuiltinSignex(Invocation* node);
  absl::Status HandleBuiltinUpdate(Invocation* node);
  absl::Status HandleBuiltinXorReduce(Invocation* node);

 private:
  // Converts a concrete type to its corresponding IR representation.
  absl::StatusOr<xls::Type*> TypeToIr(const ConcreteType& concrete_type);

  static std::string SpanToString(const absl::optional<Span>& span) {
    if (!span.has_value()) {
      return "<no span>";
    }
    return span->ToString();
  }

  // Defines "node" to map the the result of running "ir_func" with "args" -- if
  // emit_positions is on grabs the span from the node and uses it in the call.
  BValue Def(
      AstNode* node,
      const std::function<BValue(absl::optional<SourceLocation>)>& ir_func);

  // Def(), but for constant/constexpr values.
  CValue DefConst(
      AstNode* node, Value ir_value,
      const std::function<BValue(absl::optional<SourceLocation>)>& ir_func);

  // Package that IR is being generated into.
  std::shared_ptr<Package> package_;

  // Module that contains the entry point function being converted.
  Module* module_;

  // Type information for this IR conversion (determined by the type inference
  // phase).
  std::shared_ptr<TypeInfo> type_info_;

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
  std::shared_ptr<FunctionBuilder> function_builder_;

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

// Returns the mangled name of function with the given parametric bindings.
absl::StatusOr<std::string> MangleDslxName(
    absl::string_view function_name,
    const absl::btree_set<std::string>& free_keys, Module* module,
    SymbolicBindings* symbolic_bindings = nullptr);

// Returns a status that indicates an error in the IR conversion process.
absl::Status ConversionErrorStatus(const absl::optional<Span>& span,
                                   absl::string_view message);

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERTER_H_
