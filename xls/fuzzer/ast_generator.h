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
  int64 bits;
  bool signedness;
};

// Options that are used to configure the AST generator.
struct AstGeneratorOptions {
  bool emit_signed_types = true;
  bool disallow_divide = false;
  // The maximum (inclusive) width for bits types.
  int64 max_width_bits_types = 64;
  // The maximum (inclusive) width for aggregate types; e.g. tuples.
  int64 max_width_aggregate_types = 1024;
  bool emit_loops = true;
  absl::optional<absl::btree_set<BinopKind>> binop_allowlist = absl::nullopt;
  bool short_samples = false;
  bool codegen_ops_only = false;
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
  Bits ChooseBitPattern(int64 bit_count);

 private:
  static bool IsBits(TypeAnnotation* t);
  static bool IsUBits(TypeAnnotation* t);
  static bool IsArray(TypeAnnotation* t);
  static bool IsTuple(TypeAnnotation* t);
  static std::pair<std::vector<Expr*>, std::vector<TypeAnnotation*>> Unzip(
      absl::Span<const TypedExpr> typed_exprs);

  static bool EnvContainsArray(const Env& e);
  static bool EnvContainsTuple(const Env& e);

  absl::StatusOr<Function*> GenerateFunction(
      std::string name, int64 level = 0,
      absl::optional<int64> param_count = absl::nullopt);

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

  absl::StatusOr<TypedExpr> ChooseEnvValueBits(Env* env) {
    auto is_bits = [&](const TypedExpr& e) -> bool { return IsBits(e.type); };
    return ChooseEnvValue(env, is_bits);
  }
  absl::StatusOr<TypedExpr> ChooseEnvValueUBits(Env* env) {
    auto is_ubits = [&](const TypedExpr& e) -> bool { return IsUBits(e.type); };
    return ChooseEnvValue(env, is_ubits);
  }
  absl::StatusOr<TypedExpr> ChooseEnvValueArray(Env* env) {
    auto is_array = [&](const TypedExpr& e) -> bool { return IsArray(e.type); };
    return ChooseEnvValue(env, is_array);
  }
  absl::StatusOr<TypedExpr> ChooseEnvValueTuple(Env* env) {
    auto take = [&](const TypedExpr& e) -> bool { return IsTuple(e.type); };
    return ChooseEnvValue(env, take);
  }
  absl::StatusOr<TypedExpr> ChooseEnvValueNotArray(Env* env) {
    auto take = [&](const TypedExpr& e) -> bool { return !IsArray(e.type); };
    return ChooseEnvValue(env, take);
  }

  // Generates the body of a function AST node.
  absl::StatusOr<TypedExpr> GenerateBody(int64 level,
                                         absl::Span<Param* const> params);

  absl::StatusOr<TypedExpr> GenerateUnop(Env* env);

  absl::StatusOr<Expr*> GenerateUmin(TypedExpr arg, int64 other);

  // Generates a bit slice AST node.
  absl::StatusOr<TypedExpr> GenerateBitSlice(Env* env);

  // Generates one of the bitwise reductions as an Inovation node.
  absl::StatusOr<TypedExpr> GenerateBitwiseReduction(Env* env);

  // Generates a cast from bits to array type.
  absl::StatusOr<TypedExpr> GenerateCastBitsToArray(Env* env);

  // Generates a bit_slice_update builtin call.
  absl::StatusOr<TypedExpr> GenerateBitSliceUpdate(Env* env);

  int64 GenerateNaryOperandCount(Env* env) {
    XLS_CHECK(!env->empty());
    std::weibull_distribution<float> d(1.0, 0.5);
    int64 count = static_cast<int64>(std::ceil(d(rng_) * 4));
    XLS_CHECK_GT(count, 0);
    return std::min(count, static_cast<int64>(env->size()));
  }

  // Generates an expression AST node and returns it.
  absl::StatusOr<TypedExpr> GenerateExpr(int64 level, Env* env);

  // Generates an invocation of the one_hot_sel builtin.
  absl::StatusOr<TypedExpr> GenerateOneHotSelectBuiltin(Env* env);

  // Returns a binary concatenation of two arrays from env.
  //
  // The two arrays to concatenate in env will have the same *element* type.
  //
  // Precondition: there must be an array value present in env or an error
  // status will be returned.
  absl::StatusOr<TypedExpr> GenerateArrayConcat(Env* env);

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

  absl::StatusOr<TypeAnnotation*> GenerateBitsType();

  BuiltinTypeAnnotation* GeneratePrimitiveType();

  // Generates a number AST node with its associated type.
  absl::StatusOr<TypedExpr> GenerateNumber(
      Env* env, absl::optional<BitsAndSignedness> bas = absl::nullopt);

  // Generates an invocation of the map builtin.
  absl::StatusOr<TypedExpr> GenerateMap(int64 level, Env* env);

  // Generates a call to a unary builtin function.
  absl::StatusOr<TypedExpr> GenerateUnopBuiltin(Env* env);

  absl::StatusOr<Param*> GenerateParam();
  absl::StatusOr<std::vector<Param*>> GenerateParams(int64 count);

  TypeAnnotation* MakeTypeAnnotation(bool is_signed, int64 width);

  // Generates a binary operation AST node.
  absl::StatusOr<TypedExpr> GenerateBinop(Env* env);

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

  // Generates a binary operator on lhs/rhs which have the same input type.
  absl::StatusOr<TypedExpr> GenerateBinopSameInputType(
      Env* env, Expr* lhs, Expr* rhs, TypeAnnotation* input_type);

  // Generates a logical binary operation (e.g. and, xor, or).
  absl::StatusOr<TypedExpr> GenerateLogicalBinop(Env* env);

  Expr* MakeSel(Expr* test, Expr* lhs, Expr* rhs) {
    return module_->Make<Ternary>(fake_span_, test, lhs, rhs);
  }
  Expr* MakeGe(Expr* lhs, Expr* rhs) {
    return module_->Make<Binop>(fake_span_, BinopKind::kGe, lhs, rhs);
  }

  // Creates a number AsT node with value 'value' of type 'type'.
  Number* MakeNumber(int64 value, TypeAnnotation* type = nullptr);
  Number* MakeNumberFromBits(const Bits& value, TypeAnnotation* type);
  Number* MakeBool(bool value) {
    return MakeNumber(value, MakeTypeAnnotation(false, 1));
  }

  // Creates an array type with the given size and element type.
  ArrayTypeAnnotation* MakeArrayType(TypeAnnotation* element_type,
                                     int64 array_size);
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

  bool ShouldNest(int64 level) {
    std::gamma_distribution<float> g(options_.short_samples ? 1.0 : 7.0, 5.0);
    return g(rng_) >= level;
  }

  // Returns the (flattened) bit count of the given type.
  int64 GetTypeBitCount(TypeAnnotation* type);

  // Returns the (constant) size of an array type.
  //
  // Since those are the only array types the generator currently produces, this
  // can be used to determine the length of array types in the environment.
  //
  // Note: uN[42] is an ArrayTypeAnnotation, and this function will return 42
  // for it, since it's just evaluating-to-int the value in the "dim" field of
  // the ArrayTypeAnnotation.
  int64 GetArraySize(ArrayTypeAnnotation* type);

  bool RandomBool() {
    std::bernoulli_distribution d(0.5);
    return d(rng_);
  }
  float RandomFloat() {
    std::uniform_real_distribution<float> g;
    return g(rng_);
  }
  int64 RandRange(int64 limit) { return RandRange(0, limit); }
  int64 RandRange(int64 start, int64 limit) {
    XLS_CHECK_GT(limit, start);
    std::uniform_int_distribution<int64> g(start, limit - 1);
    int64 value = g(rng_);
    XLS_CHECK_LT(value, limit);
    XLS_CHECK_GE(value, start);
    return value;
  }
  float RandomWeibull(float a, float b) {
    std::weibull_distribution<float> distribution(a, b);
    return distribution(rng_);
  }

  template <typename T>
  T RandomSetChoice(const absl::btree_set<T>& choices) {
    int64 index = RandRange(choices.size());
    auto it = choices.begin();
    std::advance(it, index);
    return *it;
  }
  template <typename T>
  T RandomChoice(absl::Span<const T> choices) {
    XLS_CHECK(!choices.empty());
    int64 index = RandRange(choices.size());
    return choices[index];
  }

  std::mt19937& rng_;

  const AstGeneratorOptions options_;

  const Pos fake_pos_;
  const Span fake_span_;

  absl::btree_set<BinopKind> binops_;

  std::unique_ptr<Module> module_;

  int64 next_name_index_ = 0;

  // Functions created during generation.
  std::vector<Function*> functions_;

  // Types defined during module generation.
  std::vector<TypeDef*> type_defs_;

  // Widths of the aggregate types, indexed by TypeAnnotation::ToString().
  absl::flat_hash_map<std::string, int64> type_bit_counts_;

  // Set of constants defined during module generation.
  absl::btree_map<std::string, ConstantDef*> constants_;
};

}  // namespace xls::dslx

#endif  // XLS_FUZZER_CPP_AST_GENERATOR_H_
