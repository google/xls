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

#ifndef XLS_FUZZER_CPP_AST_GENERATOR_H_
#define XLS_FUZZER_CPP_AST_GENERATOR_H_

#include <optional>
#include <random>
#include <stack>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/strings/match.h"
#include "xls/common/test_macros.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/scanner.h"
#include "xls/fuzzer/value_generator.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

struct TypedExpr {
  Expr* expr;
  TypeAnnotation* type;
};

struct BitsAndSignedness {
  int64_t bits;
  bool signedness;
};

struct ProcProperties {
  // A list of the state types in the proc's next function. Currently, at most a
  // single state is supported. The order of types as they appear in the
  // container mirrors the order present in the proc's next function.
  std::vector<TypeAnnotation*> state_types;
  // Parameters of the proc.
  std::vector<Param*> params;
};

// Options that are used to configure the AST generator.
struct AstGeneratorOptions {
  // Emit signed types (currently not connected, signed types are always
  // generated).
  bool emit_signed_types = true;

  // The maximum (inclusive) width for bits types.
  int64_t max_width_bits_types = 64;

  // The maximum (inclusive) width for aggregate types; e.g. tuples.
  int64_t max_width_aggregate_types = 1024;

  // Emit loops (currently not connected, loops are always generated).
  bool emit_loops = true;

  // Whether to emit `gate!()` builtin calls.
  bool emit_gate = true;

  // Whether to generate a proc.
  bool generate_proc = false;

  // Whether to emit a stateless proc. When true, the state type of the proc is
  // an empty tuple. Otherwise, a random state type is generated (which may also
  // include an empty tuple). Its value is only meaningful when generate_proc is
  // `true`.
  bool emit_stateless_proc = false;
};

// Type that generates a random module for use in fuzz testing; i.e.
//
//    std::mt19937 rng;
//    AstGenerator g(AstGeneratorOptions(), &rng);
//    auto [f, module] = g.GenerateFunctionInModule().value();
//
// Where if is the main entry point inside of the returned module.
class AstGenerator {
 public:
  // The value generator must be alive for the lifetime of the object.
  AstGenerator(AstGeneratorOptions options, ValueGenerator* value_gen);

  // Generates the entity with name "name" in a module named "module_name".
  absl::StatusOr<std::unique_ptr<Module>> Generate(std::string top_entity_name,
                                                   std::string module_name);

  bool RandomBool() { return value_gen_->RandomBool(); }

  // Returns a random float uniformly distributed over [0, 1).
  float RandomFloat() { return value_gen_->RandomFloat(); }

  // Returns a random integer over the range [0, limit).
  int64_t RandRange(int64_t limit) { return value_gen_->RandRange(0, limit); }

  // Returns a random integer over the range [start, limit).
  int64_t RandRange(int64_t start, int64_t limit) {
    return value_gen_->RandRange(start, limit);
  }

  // Returns a random integer with the given expected value from a distribution
  // which tails off exponentially in both directions over the range
  // [lower_limit, inf). Useful for picking a number around some value with
  // decreasing likelihood of picking something far away from the expected
  // value.  The underlying distribution is a Poisson distribution. See:
  // https://en.wikipedia.org/wiki/Poisson_distribution
  int64_t RandomIntWithExpectedValue(float expected_value,
                                     int64_t lower_limit = 0) {
    return value_gen_->RandomIntWithExpectedValue(expected_value, lower_limit);
  }

 private:
  XLS_FRIEND_TEST(AstGeneratorTest, GeneratesParametricBindings);
  XLS_FRIEND_TEST(AstGeneratorTest, BitsTypeGetMetadata);

  // Note: we use a btree for a stable iteration order; i.e. so we can stably
  // select a random value from the environment across something like different
  // hash function seed states. That is, ideally different process invocation
  // would all produce identical generated functions for the same seed.
  using Env = absl::btree_map<std::string, TypedExpr>;

  // The context contains information for an instance in the call stack.
  struct Context {
    // TODO(https://github.com/google/xls/issues/789).
    // TODO(https://github.com/google/xls/issues/790).
    Env env;
    bool is_generating_proc;
  };

  static bool IsBits(TypeAnnotation* t);
  static bool IsUBits(TypeAnnotation* t);
  static bool IsArray(TypeAnnotation* t);
  static bool IsTuple(TypeAnnotation* t);
  static bool IsToken(TypeAnnotation* t);
  static bool IsChannel(TypeAnnotation* t);
  static bool IsNil(TypeAnnotation* t);
  static bool IsBuiltinBool(TypeAnnotation* type) {
    if (auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
      return builtin_type->GetBitCount() == 1;
    }
    return false;
  }
  // Helper that returns the signedness of type when converted to a builtin bits
  // type. Returns an error status if the type is not a builtin bits type.
  static absl::StatusOr<bool> BitsTypeIsSigned(TypeAnnotation* type);

  // Helper that returns the bit count of type when converted to a builtin bits
  // type. Returns an error status if the type is not a builtin bits type.
  absl::StatusOr<int64_t> BitsTypeGetBitCount(TypeAnnotation* type);

  static std::pair<std::vector<Expr*>, std::vector<TypeAnnotation*>> Unzip(
      absl::Span<const TypedExpr> typed_exprs);

  static bool EnvContainsArray(const Env& e);
  static bool EnvContainsTuple(const Env& e);
  static bool EnvContainsToken(const Env& e);
  static bool EnvContainsChannel(const Env& e);
  static absl::StatusOr<bool> ContainsToken(TypeAnnotation* type);

  // Generates a function with name "name".
  absl::Status GenerateFunctionInModule(std::string name);

  // Generates a proc with name "name".
  absl::Status GenerateProcInModule(std::string name);

  // Generate an DSLX function with the given name. call_depth is the current
  // depth of the call stack (if any) calling this function to be
  // generated. param_types, if given, defines the number and types of the
  // parameters.
  absl::StatusOr<Function*> GenerateFunction(
      std::string name, int64_t call_depth = 0,
      std::optional<absl::Span<TypeAnnotation* const>> param_types =
          absl::nullopt);

  // Generate the proc's config function with the given name and proc parameters
  // (proc members).
  absl::StatusOr<Function*> GenerateProcConfigFunction(
      std::string name, absl::Span<Param* const> proc_params);

  // Generate the proc's next function with the given name.
  absl::StatusOr<Function*> GenerateProcNextFunction(std::string name);

  // Generate a function to return a constant with the given TypeAnnotation to
  // serve as a Proc's [required] init function.
  absl::StatusOr<Function*> GenerateProcInitFunction(
      std::string_view name, TypeAnnotation* return_type);

  // Generate an DSLX proc with the given name.
  absl::StatusOr<Proc*> GenerateProc(std::string name);

  // Chooses a value from the environment that satisfies the predicate "take",
  // or returns nullopt if none exists.
  std::optional<TypedExpr> ChooseEnvValueOptional(
      Env* env, std::function<bool(const TypedExpr&)> take = nullptr);

  absl::StatusOr<TypedExpr> ChooseEnvValue(
      Env* env, std::function<bool(const TypedExpr&)> take = nullptr);

  // As above, but takes a type to compare for equality.
  absl::StatusOr<TypedExpr> ChooseEnvValue(Env* env, TypeAnnotation* type) {
    return ChooseEnvValue(env, [type](const TypedExpr& e) {
      return e.type->ToString() == type->ToString();
    });
  }

  // Return all values from the environment which satisty the given predicate.
  std::vector<TypedExpr> GatherAllValues(
      Env* env, std::function<bool(const TypedExpr&)> take);

  // Returns a random bits-types value from the environment.
  absl::StatusOr<TypedExpr> ChooseEnvValueBits(
      Env* env, std::optional<int64_t> bit_count = absl::nullopt) {
    auto is_bits = [&](const TypedExpr& e) -> bool {
      return IsBits(e.type) &&
             (bit_count.has_value() ? GetTypeBitCount(e.type) == bit_count
                                    : true);
    };
    return ChooseEnvValue(env, is_bits);
  }

  // Returns a random bits-types value from the environment within a certain
  // range [from, to] inclusive.
  absl::StatusOr<TypedExpr> ChooseEnvValueBitsInRange(Env* env, int64_t from,
                                                      int64_t to) {
    auto is_bits = [&](const TypedExpr& e) -> bool {
      return IsBits(e.type) && GetTypeBitCount(e.type) >= from &&
             GetTypeBitCount(e.type) <= to;
    };
    return ChooseEnvValue(env, is_bits);
  }

  // Returns a random pair of bits-types value from the environment. The
  // returned values will have the same type potentially by coercing a value to
  // match the type of the other by truncation or zero-extension.
  absl::StatusOr<std::pair<TypedExpr, TypedExpr>> ChooseEnvValueBitsPair(
      Env* env, std::optional<int64_t> bit_count = absl::nullopt);

  absl::StatusOr<TypedExpr> ChooseEnvValueUBits(Env* env) {
    auto is_ubits = [&](const TypedExpr& e) -> bool { return IsUBits(e.type); };
    return ChooseEnvValue(env, is_ubits);
  }

  absl::StatusOr<TypedExpr> ChooseEnvValueArray(
      Env* env, std::function<bool(ArrayTypeAnnotation*)> take =
                    [](auto _) { return true; }) {
    auto predicate = [&](const TypedExpr& e) -> bool {
      return IsArray(e.type) &&
             take(dynamic_cast<ArrayTypeAnnotation*>(e.type));
    };
    return ChooseEnvValue(env, predicate);
  }

  // Chooses a random tuple from the environment (if one exists). 'min_size',
  // if given, is the minimum number of elements in the chosen tuple.
  absl::StatusOr<TypedExpr> ChooseEnvValueTuple(Env* env,
                                                int64_t min_size = 0) {
    auto take = [&](const TypedExpr& e) -> bool {
      return IsTuple(e.type) &&
             dynamic_cast<TupleTypeAnnotation*>(e.type)->size() >= min_size;
    };
    return ChooseEnvValue(env, take);
  }
  absl::StatusOr<TypedExpr> ChooseEnvValueNotArray(Env* env) {
    auto take = [&](const TypedExpr& e) -> bool { return !IsArray(e.type); };
    return ChooseEnvValue(env, take);
  }

  absl::StatusOr<TypedExpr> ChooseEnvValueTupleWithoutToken(
      Env* env, int64_t min_size = 0);

  absl::StatusOr<TypedExpr> ChooseEnvValueNotContainingToken(Env* env);

  // Generates the body of a function AST node with the given
  // parameters. call_depth is the depth of the call stack (via map or other
  // function-calling operation) for the function being generated.
  absl::StatusOr<TypedExpr> GenerateBody(int64_t call_depth,
                                         absl::Span<Param* const> params,
                                         Context* ctx);

  absl::StatusOr<TypedExpr> GenerateUnop(Context* ctx);

  absl::StatusOr<Expr*> GenerateUmin(TypedExpr arg, int64_t other);

  // Generates a bit slice AST node.
  absl::StatusOr<TypedExpr> GenerateBitSlice(Context* ctx);

  // Generates one of the bitwise reductions as an Inovation node.
  absl::StatusOr<TypedExpr> GenerateBitwiseReduction(Context* ctx);

  // Generates a cast from bits to array type.
  absl::StatusOr<TypedExpr> GenerateCastBitsToArray(Context* ctx);

  // Generates a bit_slice_update builtin call.
  absl::StatusOr<TypedExpr> GenerateBitSliceUpdate(Context* ctx);

  // Generates a slice builtin call.
  absl::StatusOr<TypedExpr> GenerateArraySlice(Context* ctx);

  // Generate an operand count for a nary (variadic) instruction. lower_limit is
  // the inclusive lower limit of the distribution.
  absl::StatusOr<int64_t> GenerateNaryOperandCount(Context* ctx,
                                                   int64_t lower_limit = 0);

  // Generates an expression AST node and returns it. expr_size is a measure of
  // the size of the expression generated so far (used to probabilistically
  // limit the size of the generated expression). call_depth is the depth of the
  // call stack (via map or other function-calling operation) for the function
  // being generated.
  absl::StatusOr<TypedExpr> GenerateExpr(int64_t expr_size, int64_t call_depth,
                                         Context* ctx);

  // Generates an invocation of the one_hot_sel builtin.
  absl::StatusOr<TypedExpr> GenerateOneHotSelectBuiltin(Context* ctx);
  // Generates an invocation of the priority_sel builtin.
  absl::StatusOr<TypedExpr> GeneratePrioritySelectBuiltin(Context* ctx);

  // Returns a binary concatenation of two arrays from ctx.
  //
  // The two arrays to concatenate in ctx will have the same *element* type.
  //
  // Precondition: there must be an array value present in ctx or an error
  // status will be returned.
  absl::StatusOr<TypedExpr> GenerateArrayConcat(Context* ctx);

  // Generates an array operation using values in "ctx".
  absl::StatusOr<TypedExpr> GenerateArray(Context* ctx);

  // Returns an array index operation using values in "ctx".
  absl::StatusOr<TypedExpr> GenerateArrayIndex(Context* ctx);

  // Returns an array update operation using values in "ctx".
  absl::StatusOr<TypedExpr> GenerateArrayUpdate(Context* ctx);

  // Return a `gate!()` invocation using values in "ctx".
  absl::StatusOr<TypedExpr> GenerateGate(Context* ctx);

  // Returns a (potentially vacuous) concatenate operation of values in "ctx".
  absl::StatusOr<TypedExpr> GenerateConcat(Context* ctx);

  // Generates a return-value positioned expression.
  absl::StatusOr<TypedExpr> GenerateRetval(Context* ctx);

  // Generates a return-value positioned expression for a proc next function.
  absl::StatusOr<TypedExpr> GenerateProcNextFunctionRetval(Context* ctx);

  // Generates a counted for loop.
  absl::StatusOr<TypedExpr> GenerateCountedFor(Context* ctx);

  // Generates either a tupling operation or an index-a-tuple operation.
  absl::StatusOr<TypedExpr> GenerateTupleOrIndex(Context* ctx);

  // Generates a primitive type token for use in building a type.
  absl::StatusOr<Token*> GenerateTypePrimitive();

  // Generates a random-width Bits type. When set, the max_width_bits_types
  // overwrites the value of the max_width_bits_types parameter from the
  // options.
  TypeAnnotation* GenerateBitsType(
      std::optional<int64_t> max_width_bits_types = std::nullopt);

  // Generates a random type (bits, array, or tuple). Nesting is the amount of
  // nesting within the currently generated type (e.g., element type of a
  // tuple). Used to limit the depth of compound types. When set, the
  // max_width_bits_types overwrites the value of the max_width_bits_types
  // parameter from the options. When set, the max_width_aggregate_types
  // overwrites the value of the max_width_aggregate_types parameter from the
  // options.
  TypeAnnotation* GenerateType(
      int64_t nesting = 0,
      std::optional<int64_t> max_width_bits_types = std::nullopt,
      std::optional<int64_t> max_width_aggregate_types = std::nullopt);

  // Generates a primitive builtin type. When set, the max_width_bits_types
  // overwrites the value of the max_width_bits_types parameter from the
  // options.
  BuiltinTypeAnnotation* GeneratePrimitiveType(
      std::optional<int64_t> max_width_bits_types = std::nullopt);

  // Generates a number AST node with its associated type.
  TypedExpr GenerateNumberWithType(
      std::optional<BitsAndSignedness> bas = absl::nullopt);

  // Generates an invocation of the map builtin.
  absl::StatusOr<TypedExpr> GenerateMap(int64_t call_depth, Context* ctx);

  // Generates an invocation of a function.
  absl::StatusOr<TypedExpr> GenerateInvoke(int64_t call_depth, Context* ctx);

  // Generates a call to a unary builtin function.
  absl::StatusOr<TypedExpr> GenerateUnopBuiltin(Context* ctx);

  // Generates a parameter of a random type (if type is null) or the given type
  // (if type is non-null).
  Param* GenerateParam(TypeAnnotation* type = nullptr);

  // Generates the given number of parameters of random types.
  std::vector<Param*> GenerateParams(int64_t count);

  // Generates "count" ParametricBinding nodes for use in a function definition.
  // Currently, the all bindings have a number expression as their default.
  std::vector<ParametricBinding*> GenerateParametricBindings(int64_t count);

  // Return a token builtin type.
  BuiltinTypeAnnotation* MakeTokenType();

  TypeAnnotation* MakeTypeAnnotation(bool is_signed, int64_t width);

  // Generates a binary operation AST node in which operands and results type
  // are all the same (excluding shifts), such as: add, and, mul, etc.
  absl::StatusOr<TypedExpr> GenerateBinop(Context* ctx);

  // Generates a comparison operation AST node.
  absl::StatusOr<TypedExpr> GenerateCompare(Context* ctx);

  // Generates a channel operation AST node. The operations include recv,
  // recv_nonblocking, recv_if, send, send_if.
  absl::StatusOr<TypedExpr> GenerateChannelOp(Context* ctx);

  // Generates an Eq or Neq comparison on arrays.
  absl::StatusOr<TypedExpr> GenerateCompareArray(Context* ctx);

  // Generates an Eq or Neq comparison on tuples.
  absl::StatusOr<TypedExpr> GenerateCompareTuple(Context* ctx);

  // Generates a join operation AST node.
  absl::StatusOr<TypedExpr> GenerateJoinOp(Context* ctx);

  // Generates an operation AST node supported by the proc's next function.
  absl::StatusOr<TypedExpr> GenerateProcNextFunctionOp(Context* ctx);

  // Generates a shift operation AST node.
  absl::StatusOr<TypedExpr> GenerateShift(Context* ctx);

  absl::StatusOr<TypedExpr> GenerateSynthesizableDiv(Context* ctx);

  // Generates a group of operations containing PartialProduct ops. The output
  // of the group will be deterministic (e.g. a smulp followed by an add).
  absl::StatusOr<TypedExpr> GeneratePartialProductDeterministicGroup(
      Context* ctx);

  // Creates and returns a type ref for the given type.
  //
  // As part of this process, an ast.TypeDef is created and added to the set of
  // currently active set.
  TypeRefTypeAnnotation* MakeTypeRefTypeAnnotation(TypeAnnotation* type) {
    std::string type_name = GenSym();
    NameDef* name_def = MakeNameDef(type_name);
    auto* type_def =
        module_->Make<TypeDef>(fake_span_, name_def, type, /*is_public=*/false);
    auto* type_ref = module_->Make<TypeRef>(fake_span_, type_name, type_def);
    type_defs_.push_back(type_def);
    type_bit_counts_[type_ref->ToString()] = GetTypeBitCount(type);
    return module_->Make<TypeRefTypeAnnotation>(
        fake_span_, type_ref, /*parametrics=*/std::vector<Expr*>{});
  }

  // Generates a logical binary operation (e.g. and, xor, or).
  absl::StatusOr<TypedExpr> GenerateLogicalOp(Context* ctx);

  Expr* MakeSel(Expr* test, Expr* lhs, Expr* rhs) {
    return module_->Make<Ternary>(fake_span_, test, lhs, rhs);
  }
  Expr* MakeGe(Expr* lhs, Expr* rhs) {
    return module_->Make<Binop>(fake_span_, BinopKind::kGe, lhs, rhs);
  }

  // Creates a number AST node with value 'value' represented in a decimal
  // format.
  Number* MakeNumber(int64_t value);

  // Creates a number AST node with value 'value' of type 'type' represented in
  // the format specified.
  Number* MakeNumberFromBits(const Bits& value, TypeAnnotation* type,
                             FormatPreference format_preference);

  // Creates a number AST node with value 'value' of boolean type represented in
  // the format specified.
  Number* MakeBool(
      bool value, FormatPreference format_preference = FormatPreference::kHex) {
    return MakeNumberFromBits(UBits(value ? 1 : 0, 1),
                              MakeTypeAnnotation(false, 1), format_preference);
  }

  // Creates a number AST node with value 'value' of type 'type' represented in
  // a randomly choosen format between binary, decimal and hex. Note that these
  // function expect a builtin type or a one-dimensional array of builtin types.
  Number* GenerateNumber(int64_t value, TypeAnnotation* type);
  Number* GenerateNumberFromBits(const Bits& value, TypeAnnotation* type);

  // Creates an array type with the given size and element type.
  ArrayTypeAnnotation* MakeArrayType(TypeAnnotation* element_type,
                                     int64_t array_size);
  TupleTypeAnnotation* MakeTupleType(
      absl::Span<TypeAnnotation* const> members) {
    return module_->Make<TupleTypeAnnotation>(
        fake_span_,
        std::vector<TypeAnnotation*>(members.begin(), members.end()));
  }

  // Creates a `Range` AST node.
  Range* MakeRange(Expr* zero, Expr* arg);

  // Various convenience wrappers that use fake spans.
  NameRef* MakeNameRef(NameDef* name_def) {
    return module_->Make<NameRef>(fake_span_, name_def->identifier(), name_def);
  }
  NameDef* MakeNameDef(std::string identifier) {
    return module_->Make<NameDef>(fake_span_, std::move(identifier),
                                  /*definer=*/nullptr);
  }
  NameRef* MakeBuiltinNameRef(std::string identifier) {
    return module_->Make<NameRef>(
        fake_span_, identifier, module_->GetOrCreateBuiltinNameDef(identifier));
  }

  // Generates a unique symbol identifier.
  std::string GenSym() { return absl::StrCat("x", next_name_index_++); }

  // Returns true if the expression should continue to be extended. expr_depth
  // is a measure of the current size of the expression (number of times
  // GenerateExpr has been invoked). call_depth is the current depth of the call
  // stack (if any) calling this function to be generated.
  bool ShouldNest(int64_t expr_size, int64_t call_depth) {
    // Make non-top level functions smaller.
    double alpha = (call_depth > 0) ? 1.0 : 7.0;
    std::gamma_distribution<float> g(alpha, 5.0);
    return g(value_gen_->rng()) >= expr_size;
  }

  // Returns the (flattened) bit count of the given type.
  int64_t GetTypeBitCount(TypeAnnotation* type);

  // Returns the (constant) size of an array type.
  //
  // Since those are the only array types the generator currently produces, this
  // can be used to determine the length of array types in the environment.
  //
  // Note: uN[42] is an ArrayTypeAnnotation, and this function will return 42
  // for it, since it's just evaluating-to-int the value in the "dim" field of
  // the ArrayTypeAnnotation.
  int64_t GetArraySize(const ArrayTypeAnnotation* type);

  // Gets-or-creates a top level constant with the given value, using the
  // minimum number of bits required to make that constant.
  ConstRef* GetOrCreateConstRef(
      int64_t value, std::optional<int64_t> want_width = absl::nullopt);

  template <typename T>
  T RandomSetChoice(const absl::btree_set<T>& choices) {
    int64_t index = RandRange(choices.size());
    auto it = choices.begin();
    std::advance(it, index);
    return *it;
  }
  template <typename T>
  T RandomChoice(absl::Span<const T> choices) {
    XLS_CHECK(!choices.empty());
    int64_t index = RandRange(choices.size());
    return choices[index];
  }

  // Returns a recoverable status error with the given message. A recoverable
  // error may be raised to indicate that the generation of an individual
  // expression failed but that generation should continue.
  absl::Status RecoverableError(std::string_view message) {
    return absl::AbortedError(absl::StrFormat("RecoverableError: %s", message));
  }
  bool IsRecoverableError(const absl::Status& status) {
    return !status.ok() && status.code() == absl::StatusCode::kAborted &&
           absl::StartsWith(status.message(), "RecoverableError");
  }

  // Returns a recoverable error if the given width is too large for the limits
  // set in the generator options.
  absl::Status VerifyBitsWidth(int64_t width) {
    if (width > options_.max_width_bits_types) {
      return RecoverableError("bits type too wide.");
    }
    return absl::OkStatus();
  }
  absl::Status VerifyAggregateWidth(int64_t width) {
    if (width > options_.max_width_aggregate_types) {
      return RecoverableError("aggregate type too wide.");
    }
    return absl::OkStatus();
  }

  ValueGenerator* value_gen_;

  const AstGeneratorOptions options_;

  const Pos fake_pos_;
  const Span fake_span_;

  absl::btree_set<BinopKind> binops_;

  std::unique_ptr<Module> module_;

  int64_t next_name_index_ = 0;

  // Functions created during generation.
  std::vector<Function*> functions_;

  // Types defined during module generation.
  std::vector<TypeDef*> type_defs_;

  // Widths of the aggregate types, indexed by TypeAnnotation::ToString().
  absl::flat_hash_map<std::string, int64_t> type_bit_counts_;

  // Set of constants defined during module generation.
  absl::btree_map<std::string, ConstantDef*> constants_;

  // Contains properties of the generated proc.
  ProcProperties proc_properties_;
};

}  // namespace xls::dslx

#endif  // XLS_FUZZER_CPP_AST_GENERATOR_H_
