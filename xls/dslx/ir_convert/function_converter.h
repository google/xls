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

#ifndef XLS_DSLX_IR_CONVERT_FUNCTION_CONVERTER_H_
#define XLS_DSLX_IR_CONVERT_FUNCTION_CONVERTER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/ir_convert/channel_scope.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/proc_config_ir_converter.h"
#include "xls/dslx/ir_convert/proc_scoped_channel_scope.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {

// Converts an interpreter value to an IR value.
absl::StatusOr<Value> InterpValueToValue(const InterpValue& v);

// For all free variables of "node", adds them transitively for any required
// constant dependencies to the converter.
//
// Warning: the `ConstantDef`s may be owned by different modules, e.g. if we had
// to traverse a name that was `use`d into the current module.
absl::StatusOr<std::vector<ConstantDef*>> GetConstantDepFreevars(
    AstNode* node, TypeInfo& type_info);

// Wrapper around the type information query for whether DSL function "f"
// requires an implicit token calling convention, including overrides provided
// via ConvertOptions.
bool GetRequiresImplicitToken(const dslx::Function& f, ImportData* import_data,
                              const ConvertOptions& options);

// Creates a function that wraps up `implicit_token_f`.
//
// Precondition: `implicit_token_f` must use the "implicit token" calling
// convention, see `CallingConvention` for details.
//
// The wrapped function exposes the implicit token function as if it were a
// normal function, so it can be called by the outside world in a typical
// fashion as an entry point (e.g. the IR JIT, Verilog module signature, etc).
absl::StatusOr<xls::Function*> EmitImplicitTokenEntryWrapper(
    xls::Function* implicit_token_f, dslx::Function* dslx_function, bool is_top,
    PackageInterfaceProto* interface_proto,
    const PackageInterfaceProto::Function& implicit_token_proto);

// Bundles together a package pointer with a supplementary map we keep that
// shows the DSLX function that led to IR functions in the package.
struct PackageData {
  PackageConversionData* conversion_info;
  absl::flat_hash_map<xls::FunctionBase*, dslx::Function*> ir_to_dslx;
  absl::flat_hash_map<const dslx::Invocation*, xls::Proc*>
      invocation_to_ir_proc;
  absl::flat_hash_set<xls::Function*> wrappers;
};

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
                    ProcConversionData* proc_data, ChannelScope* channel_scope,
                    bool is_top);

  // Main entry point to request conversion of the DSLX function "f" to an IR
  // function.
  absl::Status HandleFunction(Function* node, TypeInfo* type_info,
                              const ParametricEnv* parametric_env);

  absl::Status HandleProcNextFunction(
      Function* pf, const Invocation* invocation, TypeInfo* type_info,
      ImportData* import_data, const ParametricEnv* parametric_env,
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

  // Contains/returns the pertinent information about a range expression (either
  // a Range node or the range() builtin).
  struct RangeData {
    int64_t start_value;
    int64_t trip_count;
    int64_t bit_width;
  };

  const FileTable& file_table() const { return import_data_->file_table(); }

  // Wraps up a type so it is (token, u1, type).
  static xls::Type* WrapIrForImplicitTokenType(xls::Type* type,
                                               Package* package) {
    xls::Type* token_type = package->GetTokenType();
    xls::Type* u1_type = package->GetBitsType(1);
    return package->GetTupleType({token_type, u1_type, type});
  }

  // Helper function used for adding a parameter type in for-loops and similar.
  BValue AddParam(std::string_view name, xls::Type* type) {
    CHECK(function_proto_);
    auto* param_proto = function_proto_.value()->add_parameters();
    param_proto->set_name(name);
    *param_proto->mutable_type() = type->ToProto();
    // This is only for internal 'for-loop' and similar impl functions so they
    // won't ever be emitted as verilog.
    return function_builder_->Param(name, type);
  }

  // Helper function used for adding a parameter type wrapped up in a
  // token/activation boolean.
  BValue AddTokenWrappedParam(xls::Type* type) {
    BuilderBase* fb = function_builder_.get();
    xls::Type* wrapped_type = WrapIrForImplicitTokenType(type, package());
    BValue param = fb->Param("__token_wrapped", wrapped_type);
    BValue entry_token = fb->TupleIndex(param, 0);
    BValue activated = fb->TupleIndex(param, 1);
    auto create_control_predicate = [activated] { return activated; };
    implicit_token_data_ =
        ImplicitTokenData{entry_token, activated, create_control_predicate};
    BValue unwrapped = fb->TupleIndex(param, 2);
    CHECK(function_proto_);
    auto* param_proto = function_proto_.value()->add_parameters();
    param_proto->set_name(param.GetName());
    *param_proto->mutable_type() = param.GetType()->ToProto();
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
  using IrValue = std::variant<BValue, CValue, Channel*, ChannelInterface*>;

  // Helper for converting an IR value to its BValue pointer for use in
  // debugging.
  static std::string IrValueToString(const IrValue& value);

  // Helper that checks that the given IrValue is a Channel or ChannelInterface,
  // or holds a ChannelInterface.
  static absl::Status CheckValueIsChannel(const IrValue& ir_value);

  // Helper that converts the IrValue to the desired ChannelRef variant.
  template <typename NodeT, typename ChanRef, typename ChanInt>
  static absl::StatusOr<ChanRef> IrValueToChannelRef(const IrValue& ir_value);

  static absl::StatusOr<SendChannelRef> IrValueToSendChannelRef(
      const IrValue& ir_value);
  static absl::StatusOr<ReceiveChannelRef> IrValueToReceiveChannelRef(
      const IrValue& ir_value);
  static absl::StatusOr<ChannelInterface*> IrValueToChannelInterface(
      const IrValue& ir_value);

  void SetFunctionBuilder(std::unique_ptr<BuilderBase> builder);

  // See `GetRequiresImplicitToken(f, import_data, options)`.
  bool GetRequiresImplicitToken(const dslx::Function* f) const;

  CallingConvention GetCallingConvention(const Function* f) const {
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
  absl::StatusOr<std::unique_ptr<Type>> ResolveType(const AstNode* node);

  // Helper that composes ResolveType() and TypeToIr().
  absl::StatusOr<xls::Type*> ResolveTypeToIr(const AstNode* node);

  // -- Accessors

  void SetParametricEnv(const ParametricEnv* value) {
    parametric_env_map_ = value->ToMap();
  }
  void set_parametric_env_map(
      absl::flat_hash_map<std::string, InterpValue> map) {
    parametric_env_map_ = std::move(map);
  }

  // Gets the current counter of counted_for loops we've observed and bumps it.
  // This is useful for generating new symbols for the functions that serve as
  // XLS counted_for "bodies".
  int64_t GetAndBumpCountedForCount() { return counted_for_count_++; }

  // Gets the unrolled version of the given for loop for the current context.
  std::optional<const Expr*> GetUnrolledForLoop(const UnrollFor* loop);

  std::optional<InterpValue> GetParametricBinding(
      std::string_view identifier) const {
    auto it = parametric_env_map_.find(identifier);
    if (it == parametric_env_map_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // Note: this object holds a map, but we can return a ParametricEnv object
  // on demand.
  ParametricEnv GetParametricEnv() const {
    return ParametricEnv(parametric_env_map_);
  }

  // Returns the parametric env to be used in the callee for this invocation.
  //
  // We must provide the current evaluation context (module_name, function_name,
  // caller_parametric_env) in order to retrieve the correct parametric
  // bindings to use in the callee invocation.
  //
  // Args:
  //  invocation: Invocation that the bindings are being retrieved for.
  //
  // Returns:
  //  The parametric bindings for the given instantiation (function call or proc
  //  spawn).
  std::optional<const ParametricEnv*> GetInvocationCalleeBindings(
      const Invocation* invocation) const {
    ParametricEnv key = GetParametricEnv();
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
  absl::Status HandleChannelDecl(const ChannelDecl* node);
  absl::Status HandleNameRef(const NameRef* node);

  absl::Status HandleExternNameRef(const NameRef* node,
                                   const UseTreeEntry* use_tree_entry);

  absl::Status HandleNumber(const Number* node);
  absl::Status HandleParam(const Param* node);
  absl::Status HandleString(const String* node);
  absl::Status HandleUnop(const Unop* node);
  absl::Status HandleXlsTuple(const XlsTuple* node);
  absl::Status HandleZeroMacro(const ZeroMacro* node);
  absl::Status HandleAllOnesMacro(const AllOnesMacro* node);
  absl::Status HandleUnrollFor(const UnrollFor* node);
  absl::Status HandleSpawn(const Spawn* node);

  // AstNode handlers that recur "manually" internal to the handler.
  absl::Status HandleArray(const Array* node);
  absl::Status HandleAttr(const Attr* node);
  absl::Status HandleStatementBlock(const StatementBlock* node);
  absl::Status HandleCast(const Cast* node);
  absl::Status HandleColonRef(const ColonRef* node);
  absl::Status HandleConstantDef(const ConstantDef* node);

  absl::Status HandleFor(const For* node);

  // Helper that adds a parameter for the induction variable node for a ranged
  // for loop, e.g. when we have a pattern like: `for (i, accum) in 1..10` this
  // handles the `ivar` named `i`.
  absl::StatusOr<BValue> HandleRangedForInductionVariable(
      const For* node, FunctionConverter& body_converter,
      NameDefTree::Leaf ivar);

  // Helpers that adds a parameter for the loop carried accumulator value.
  // Handles the fact that we may need to destructure a pattern for
  // `carry_node`.
  absl::Status HandleForLoopCarry(const For* node,
                                  FunctionConverter& body_converter,
                                  const std::optional<RangeData>& range_data,
                                  BValue loop_index, NameDefTree::Leaf ivar,
                                  AstNode* carry_node);

  // Returns the relevant name definitions for the lexical scope of the for
  // loop.
  //
  // This lets us convert the body of the for loop into a function and pass all
  // of the lexically required bindings as invariant arguments.
  absl::StatusOr<std::vector<const NameDef*>> HandleForLexicalScope(
      const For* node, FunctionConverter& body_converter);

  // Runs conversion on the body block of `node`, placing the built IR into
  // `body_converter`.
  //
  // This encapsulates any token threading that has to happen if the body is
  // implicit-token calling convention.
  //
  // Returns the `init` BValue (which is also appropriately adapted for the
  // calling convention).
  absl::StatusOr<BValue> HandleForBody(const For* node,
                                       FunctionConverter& body_converter,
                                       int64_t trip_count);

  absl::Status HandleFormatMacro(const FormatMacro* node);
  absl::Status HandleIndex(const Index* node);
  absl::Status HandleInvocation(const Invocation* node);
  absl::Status HandleLambda(const Lambda* node);
  absl::Status HandleLet(const Let* node);
  absl::Status HandleLetChannelDecl(const Let* node);
  absl::Status HandleMatch(const Match* node);
  absl::Status HandleRange(const Range* node);
  absl::Status HandleSplatStructInstance(const SplatStructInstance* node);
  absl::Status HandleStatement(const Statement* node);
  absl::Status HandleStructInstance(const StructInstance* node);
  absl::Status HandleConditional(const Conditional* node);
  absl::Status HandleTupleIndex(const TupleIndex* node);

  // Handles invocation of a user-defined function (UDF).
  absl::Status HandleUdfInvocation(const Invocation* node, xls::Function* f,
                                   std::vector<BValue> args);

  // Handles the `fail!()` builtin invocation.
  absl::Status HandleFailBuiltin(const Invocation* node, Expr* label_expr,
                                 BValue arg);

  // Handles the `assert!()` builtin invocation.
  absl::Status HandleAssertBuiltin(const Invocation* node, BValue predicate,
                                   Expr* label_expr);

  // Handles the `assert_eq` builtin invocation.
  absl::Status HandleAssertEqBuiltin(const Invocation* node, BValue lhs,
                                     BValue rhs);

  // Handles the `assert_lt` builtin invocation.
  absl::Status HandleAssertLtBuiltin(const Invocation* node, BValue lhs,
                                     BValue rhs);
  // Handles the `cover!()` builtin invocation.
  absl::Status HandleCoverBuiltin(const Invocation* node, BValue condition);

  // Handles an arm of a match expression.
  absl::StatusOr<BValue> HandleMatcher(NameDefTree* matcher,
                                       const BValue& matched_value,
                                       const Type& matched_type);

  // Makes the specified builtin available to the package.
  absl::StatusOr<BValue> DefMapWithBuiltin(const Invocation* parent_node,
                                           NameRef* node, AstNode* arg,
                                           const ParametricEnv& parametric_env);

  absl::StatusOr<BValue> HandleMap(const Invocation* node);

  // Builtin invocation handlers.
  // keep-sorted start
  absl::Status HandleBuiltinAndReduce(const Invocation* node);
  absl::Status HandleBuiltinArrayRev(const Invocation* node);
  absl::Status HandleBuiltinArraySize(const Invocation* node);
  absl::Status HandleBuiltinArraySlice(const Invocation* node);
  absl::Status HandleBuiltinBitCount(const Invocation* node);
  absl::Status HandleBuiltinBitSliceUpdate(const Invocation* node);
  absl::Status HandleBuiltinCheckedCast(const Invocation* node);
  absl::Status HandleBuiltinClz(const Invocation* node);
  absl::Status HandleBuiltinConfiguredValueOr(const Invocation* node);
  absl::Status HandleBuiltinCtz(const Invocation* node);
  absl::Status HandleBuiltinDecode(const Invocation* node);
  absl::Status HandleBuiltinElementCount(const Invocation* node);
  absl::Status HandleBuiltinEncode(const Invocation* node);
  absl::Status HandleBuiltinGate(const Invocation* node);
  absl::Status HandleBuiltinJoin(const Invocation* node);
  absl::Status HandleBuiltinOneHot(const Invocation* node);
  absl::Status HandleBuiltinOneHotSel(const Invocation* node);
  absl::Status HandleBuiltinOrReduce(const Invocation* node);
  absl::Status HandleBuiltinPrioritySel(const Invocation* node);
  absl::Status HandleBuiltinRecv(const Invocation* node);
  absl::Status HandleBuiltinRecvIf(const Invocation* node);
  absl::Status HandleBuiltinRecvIfNonBlocking(const Invocation* node);
  absl::Status HandleBuiltinRecvNonBlocking(const Invocation* node);
  absl::Status HandleBuiltinRev(const Invocation* node);
  absl::Status HandleBuiltinSMulp(const Invocation* node);
  absl::Status HandleBuiltinSend(const Invocation* node);
  absl::Status HandleBuiltinSendIf(const Invocation* node);
  absl::Status HandleBuiltinSignex(const Invocation* node);
  absl::Status HandleBuiltinToken(const Invocation* node);
  absl::Status HandleBuiltinUMulp(const Invocation* node);
  absl::Status HandleBuiltinUpdate(const Invocation* node);
  absl::Status HandleBuiltinWideningCast(const Invocation* node);
  absl::Status HandleBuiltinXorReduce(const Invocation* node);
  absl::Status HandleBuiltinZip(const Invocation* node);
  // keep-sorted end

  // Derefences the type definition to a struct definition.
  absl::StatusOr<StructDef*> DerefStruct(TypeDefinition node);

  // Derefences the type definition to a enum definition.
  absl::StatusOr<EnumDef*> DerefEnum(TypeDefinition node);

  absl::Status CastToArray(const Cast* node, const ArrayType& output_type);
  absl::Status CastFromArray(const Cast* node, const Type& output_type);

  // Returns the fully resolved (mangled) name for the callee of the given node,
  // with parametric values mangled in appropriately.
  absl::StatusOr<std::string> GetCalleeIdentifier(const Invocation* node);

  absl::StatusOr<RangeData> GetRangeData(const Expr* iterable);

  struct AssertionLabelData {
    // The codegen-level label we'll apply to the corresponding assertion.
    std::string label;

    // The (more arbitrary text) message we'll display if the assertion fails.
    std::string message;
  };

  // Helper that provides the label we'll use for an emitted assertion as well
  // as the message we'll use in building the IR node.
  absl::StatusOr<AssertionLabelData> GetAssertionLabel(
      std::string_view caller_name, const Expr* label_expr, const Span& span);

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
    return SourceInfo(SourceLocation{ir_fileno_, lineno, colno});
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

  absl::Status Visit(const AstNode* node);

  Package* package() const {
    return package_data_.conversion_info->package.get();
  }

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

  // The in-progress proto describing types of the function.
  std::optional<PackageInterfaceProto::Function*> function_proto_;
  // The in-progress proto describing types of the proc.
  std::optional<PackageInterfaceProto::Proc*> proc_proto_;

  // When we have a fail!() operation we implicitly need to thread a token /
  // activation boolean -- the data for this is kept here if necessary.
  std::optional<ImplicitTokenData> implicit_token_data_;

  // Mapping of parametric bindings active in this translation (e.g. what
  // integral values parametrics are taking on).
  absl::flat_hash_map<std::string, InterpValue> parametric_env_map_;

  // File number for use in source positions.
  xls::Fileno ir_fileno_;

  // This is only set for procs; it holds the final recurrent state.
  BValue next_value_;

  // Number of "counted for" nodes we've observed in this function.
  int64_t counted_for_count_ = 0;

  // Uses id_to_members to resolve a proc's [constant] member values.
  ProcConversionData* proc_data_;

  // Used to handle channel array accesses.
  ChannelScope* channel_scope_;

  std::vector<BValue> tokens_;

  // The ID of the unique Proc instance currently being converted. Only valid if
  // a Proc is being converted.
  std::optional<ProcId> proc_id_;

  // If the function is the entry function resulting in a top entity in the IR.
  bool is_top_;

  // The current type of function being processed.
  FunctionTag current_fn_tag_ = FunctionTag::kNormal;

  // The last tuple converted. Used for mapping the return tuple of a proc
  // `config` method to actual proc members.
  std::vector<BValue> last_tuple_;

  std::unique_ptr<ProcScopedChannelScope> proc_scoped_channel_scope_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_IR_CONVERT_FUNCTION_CONVERTER_H_
