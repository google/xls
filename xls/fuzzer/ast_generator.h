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

#include <random>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/scanner.h"

namespace xls::dslx {

struct TypedExpr {
  Expr* expr;
  TypeAnnotation* type;
};

struct BitsAndSignedness {
  int64_t bits;
  bool signedness;
};

// Options that are used to configure the AST generator.
struct AstGeneratorOptions {
  // Emit signed types (currently not connected, signed types are always
  // generated).
  bool emit_signed_types = true;

  // Don't emit divide operations (currently not connected, divides are never
  // generated).
  bool disallow_divide = false;

  // The maximum (inclusive) width for bits types.
  int64_t max_width_bits_types = 64;

  // The maximum (inclusive) width for aggregate types; e.g. tuples.
  int64_t max_width_aggregate_types = 1024;

  // Emit loops (currently not connected, loops are always generated).
  bool emit_loops = true;

  // The set of binary ops (arithmetic and bitwise ops excepting shifts) to
  // generate. For example: add, and, mul, etc.
  absl::optional<absl::btree_set<BinopKind>> binop_allowlist = absl::nullopt;

  // If true, then generated samples that have fewer operations.
  bool short_samples = false;

  // If true, generate empty tuples potentially as the return value, parameters,
  // or intermediate values.
  // TODO(https://github.com/google/xls/issues/346): 2021-03-19 Remove this
  // option when pipeline generator handles empty tuples properly.
  bool generate_empty_tuples = true;
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
  // Note: we use a btree for a stable iteration order; i.e. so we can stably
  // select a random value from the environment across something like different
  // hash function seed states. That is, ideally different process invocation
  // would all produce identical generated functions for the same seed.
  using Env = absl::btree_map<std::string, TypedExpr>;

  AstGenerator(AstGeneratorOptions options, std::mt19937* rng);

  // Generates a function with name "fn_name" in a module named "module_name".
  absl::StatusOr<std::pair<Function*, std::unique_ptr<Module>>>
  GenerateFunctionInModule(std::string fn_name, std::string module_name);

  // Chooses a random "interesting" bit pattern with the given bit count.
  Bits ChooseBitPattern(int64_t bit_count);

  bool RandomBool() {
    std::bernoulli_distribution d(0.5);
    return d(rng_);
  }

  // Returns a random float uniformly distributed over [0, 1).
  float RandomFloat() {
    std::uniform_real_distribution<float> g(0.0f, 1.0f);
    return g(rng_);
  }

  // Returns a random integer over the range [0, limit).
  int64_t RandRange(int64_t limit) { return RandRange(0, limit); }

  // Returns a random integer over the range [start, limit).
  int64_t RandRange(int64_t start, int64_t limit) {
    XLS_CHECK_GT(limit, start);
    std::uniform_int_distribution<int64_t> g(start, limit - 1);
    int64_t value = g(rng_);
    XLS_CHECK_LT(value, limit);
    XLS_CHECK_GE(value, start);
    return value;
  }

  // Returns a random integer with the given expected value from a distribution
  // which tails off exponentially in both directions over the range
  // [lower_limit, inf). Useful for picking a number around some value with
  // decreasing likelihood of picking something far away from the expected
  // value.  The underlying distribution is a Poisson distribution. See:
  // https://en.wikipedia.org/wiki/Poisson_distribution
  int64_t RandomIntWithExpectedValue(float expected_value,
                                     int64_t lower_limit = 0) {
    XLS_CHECK_GE(expected_value, lower_limit);
    std::poisson_distribution<int64_t> distribution(expected_value -
                                                    lower_limit);
    return distribution(rng_) + lower_limit;
  }

 private:
  static bool IsBits(TypeAnnotation* t);
  static bool IsUBits(TypeAnnotation* t);
  static bool IsArray(TypeAnnotation* t);
  static bool IsTuple(TypeAnnotation* t);
  static bool IsNil(TypeAnnotation* t);
  static std::pair<std::vector<Expr*>, std::vector<TypeAnnotation*>> Unzip(
      absl::Span<const TypedExpr> typed_exprs);

  static bool EnvContainsArray(const Env& e);
  static bool EnvContainsTuple(const Env& e);

  // Generate an DSLX function with the given name. call_depth is the current
  // depth of the call stack (if any) calling this function to be
  // generated. param_types, if given, defines the number and types of the
  // parameters.
  absl::StatusOr<Function*> GenerateFunction(
      std::string name, int64_t call_depth = 0,
      absl::optional<absl::Span<TypeAnnotation* const>> param_types =
          absl::nullopt);

  // Chooses a value from the environment that satisfies the predicate "take",
  // or returns nullopt if none exists.
  absl::optional<TypedExpr> ChooseEnvValueOptional(
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
      Env* env, absl::optional<int64_t> bit_count = absl::nullopt) {
    auto is_bits = [&](const TypedExpr& e) -> bool {
      return IsBits(e.type) &&
             !(bit_count.has_value() || GetTypeBitCount(e.type) == bit_count);
    };
    return ChooseEnvValue(env, is_bits);
  }

  // Returns a random pair of bits-types value from the environment. The
  // returned values will have the same type potentially by coercing a value to
  // match the type of the other by truncation or zero-extension.
  absl::StatusOr<std::pair<TypedExpr, TypedExpr>> ChooseEnvValueBitsPair(
      Env* env, absl::optional<int64_t> bit_count = absl::nullopt);

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

  // Generates the body of a function AST node with the given
  // parameters. call_depth is the depth of the call stack (via map or other
  // function-calling operation) for the function being generated.
  absl::StatusOr<TypedExpr> GenerateBody(int64_t call_depth,
                                         absl::Span<Param* const> params);

  absl::StatusOr<TypedExpr> GenerateUnop(Env* env);

  absl::StatusOr<Expr*> GenerateUmin(TypedExpr arg, int64_t other);

  // Generates a bit slice AST node.
  absl::StatusOr<TypedExpr> GenerateBitSlice(Env* env);

  // Generates one of the bitwise reductions as an Inovation node.
  absl::StatusOr<TypedExpr> GenerateBitwiseReduction(Env* env);

  // Generates a cast from bits to array type.
  absl::StatusOr<TypedExpr> GenerateCastBitsToArray(Env* env);

  // Generates a bit_slice_update builtin call.
  absl::StatusOr<TypedExpr> GenerateBitSliceUpdate(Env* env);

  // Generates a slice builtin call.
  absl::StatusOr<TypedExpr> GenerateArraySlice(Env* env);

  // Generate an operand count for a nary (variadic) instruction. lower_limit is
  // the inclusive lower limit of the distribution.
  int64_t GenerateNaryOperandCount(Env* env, int64_t lower_limit = 0) {
    XLS_CHECK(!env->empty());
    int64_t result = std::min(RandomIntWithExpectedValue(4, lower_limit),
                              static_cast<int64_t>(env->size()));
    XLS_CHECK_GE(result, lower_limit);
    return result;
  }

  // Generates an expression AST node and returns it. expr_size is a measure of
  // the size of the expression generated so far (used to probabilistically
  // limit the size of the generated expression). call_depth is the depth of the
  // call stack (via map or other function-calling operation) for the function
  // being generated.
  absl::StatusOr<TypedExpr> GenerateExpr(int64_t expr_size, int64_t call_depth,
                                         Env* env);

  // Generates an invocation of the one_hot_sel builtin.
  absl::StatusOr<TypedExpr> GenerateOneHotSelectBuiltin(Env* env);

  // Returns a binary concatenation of two arrays from env.
  //
  // The two arrays to concatenate in env will have the same *element* type.
  //
  // Precondition: there must be an array value present in env or an error
  // status will be returned.
  absl::StatusOr<TypedExpr> GenerateArrayConcat(Env* env);

  // Generates an array operation using values in "env".
  absl::StatusOr<TypedExpr> GenerateArray(Env* env);

  // Returns an array index operation using values in "env".
  absl::StatusOr<TypedExpr> GenerateArrayIndex(Env* env);

  // Returns an array update operation using values in "env".
  absl::StatusOr<TypedExpr> GenerateArrayUpdate(Env* env);

  // Returns a (potentially vacuous) concatenate operation of values in "env".
  absl::StatusOr<TypedExpr> GenerateConcat(Env* env);

  // Generates a return-value positioned expression.
  absl::StatusOr<TypedExpr> GenerateRetval(Env* env);

  // Generates a counted for loop.
  absl::StatusOr<TypedExpr> GenerateCountedFor(Env* env);

  // Generates either a tupling operation or an index-a-tuple operation.
  absl::StatusOr<TypedExpr> GenerateTupleOrIndex(Env* env);

  // Generates a primitive type token for use in building a type.
  absl::StatusOr<Token*> GenerateTypePrimitive();

  // Generates a random-width Bits type.
  TypeAnnotation* GenerateBitsType();

  // Generates a random type (bits, array, or tuple). Nesting is the amount of
  // nesting within the currently generated type (e.g., element type of a
  // tuple). Used to limit the depth of compound types.
  TypeAnnotation* GenerateType(int64_t nesting = 0);

  BuiltinTypeAnnotation* GeneratePrimitiveType();

  // Generates a number AST node with its associated type.
  absl::StatusOr<TypedExpr> GenerateNumber(
      Env* env, absl::optional<BitsAndSignedness> bas = absl::nullopt);

  // Generates an invocation of the map builtin.
  absl::StatusOr<TypedExpr> GenerateMap(int64_t call_depth, Env* env);

  // Generates a call to a unary builtin function.
  absl::StatusOr<TypedExpr> GenerateUnopBuiltin(Env* env);

  // Generates a parameter of a random type (if type is null) or the given type
  // (if type is non-null).
  Param* GenerateParam(TypeAnnotation* type = nullptr);

  // Generates the given number of parameters of random types.
  std::vector<Param*> GenerateParams(int64_t count);

  TypeAnnotation* MakeTypeAnnotation(bool is_signed, int64_t width);

  // Generates a binary operation AST node in which operands and results type
  // are all the same (excluding shifts), such as: add, and, mul, etc.
  absl::StatusOr<TypedExpr> GenerateBinop(Env* env);

  // Generates a comparison operation AST node.
  absl::StatusOr<TypedExpr> GenerateCompare(Env* env);

  // Generates an Eq or Neq comparison on arrays.
  absl::StatusOr<TypedExpr> GenerateCompareArray(Env* env);

  // Generates an Eq or Neq comparison on tuples.
  absl::StatusOr<TypedExpr> GenerateCompareTuple(Env* env);

  // Generates a shift operation AST node.
  absl::StatusOr<TypedExpr> GenerateShift(Env* env);

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
  absl::StatusOr<TypedExpr> GenerateLogicalOp(Env* env);

  Expr* MakeSel(Expr* test, Expr* lhs, Expr* rhs) {
    return module_->Make<Ternary>(fake_span_, test, lhs, rhs);
  }
  Expr* MakeGe(Expr* lhs, Expr* rhs) {
    return module_->Make<Binop>(fake_span_, BinopKind::kGe, lhs, rhs);
  }

  // Creates a number AsT node with value 'value' of type 'type'.
  Number* MakeNumber(int64_t value, TypeAnnotation* type = nullptr);
  Number* MakeNumberFromBits(const Bits& value, TypeAnnotation* type);
  Number* MakeBool(bool value) {
    return MakeNumber(value, MakeTypeAnnotation(false, 1));
  }

  // Creates an array type with the given size and element type.
  ArrayTypeAnnotation* MakeArrayType(TypeAnnotation* element_type,
                                     int64_t array_size);
  TupleTypeAnnotation* MakeTupleType(
      absl::Span<TypeAnnotation* const> members) {
    return module_->Make<TupleTypeAnnotation>(
        fake_span_,
        std::vector<TypeAnnotation*>(members.begin(), members.end()));
  }

  static bool IsBuiltinBool(TypeAnnotation* type) {
    if (auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type)) {
      return builtin_type->GetBitCount() == 1;
    }
    return false;
  }

  // Makes an invocation of the range builtin.
  Invocation* MakeRange(Expr* zero, Expr* arg);

  // Various convenience wrappers that use fake spans.
  NameRef* MakeNameRef(NameDef* name_def) {
    return module_->Make<NameRef>(fake_span_, name_def->identifier(), name_def);
  }
  NameDef* MakeNameDef(std::string identifier) {
    return module_->Make<NameDef>(fake_span_, std::move(identifier),
                                  /*definer=*/nullptr);
  }
  NameRef* MakeBuiltinNameRef(std::string identifier) {
    return module_->Make<NameRef>(fake_span_, identifier,
                                  module_->Make<BuiltinNameDef>(identifier));
  }

  // Generates a unique symbol identifier.
  std::string GenSym() { return absl::StrCat("x", next_name_index_++); }

  // Returns true if the expression should continue to be extended. expr_depth
  // is a measure of the current size of the expression (number of times
  // GenerateExpr has been invoked). call_depth is the current depth of the call
  // stack (if any) calling this function to be generated.
  bool ShouldNest(int64_t expr_size, int64_t call_depth) {
    // Make non-top level functions smaller.
    double alpha = (options_.short_samples || call_depth > 0) ? 1.0 : 7.0;
    std::gamma_distribution<float> g(alpha, 5.0);
    return g(rng_) >= expr_size;
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
      int64_t value, absl::optional<int64_t> want_width = absl::nullopt);

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

  std::mt19937& rng_;

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
};

}  // namespace xls::dslx

#endif  // XLS_FUZZER_CPP_AST_GENERATOR_H_
