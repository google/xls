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

#ifndef XLS_FUZZER_AST_GENERATOR_H_
#define XLS_FUZZER_AST_GENERATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/test_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"
#include "xls/fuzzer/ast_generator_options.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {

// While building our AST, our choices can affect the legal scheduling options;
// in particular, some ASTs cannot be scheduled in anything under N cycles (for
// some N > 1).
//
// We track the impact of our choices through every value we construct using two
// descriptors:
// - the last operation applied to this value (or its dependencies) that could
//   affect scheduling, here called its "last delaying operation", and
// - the earliest stage the operation that produced this value could have
//   legally been scheduled into, here called its "min stage".
enum class LastDelayingOp : uint8_t { kNone, kSend, kRecv };

struct TypedExpr {
  Expr* expr;
  TypeAnnotation* type;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 1;
};

struct AnnotatedType {
  TypeAnnotation* type;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 1;
};

struct AnnotatedParam {
  Param* param;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 1;
};

// These tags also need to be tracked for functions, in order to model the
// effect on their returned values.
struct AnnotatedFunction {
  Function* function;
  LastDelayingOp last_delaying_op = LastDelayingOp::kNone;
  int64_t min_stage = 1;
};

// Procs don't have return values - but we still need to track the smallest
// number of stages the procs we generate can be legally scheduled into.
struct AnnotatedProc {
  Proc* proc;
  int64_t min_stages = 1;
};

// When we return a module, we also need to track the smallest number of stages
// it can be scheduled into as a pipeline.
struct AnnotatedModule {
  std::unique_ptr<Module> module;
  int64_t min_stages = 1;
};

struct BitsAndSignedness {
  int64_t bits;
  bool signedness;
};

// Options that are used to configure the AST generator.
//
// See ast_generator_options.proto for field descriptions.
struct AstGeneratorOptions {
  bool emit_signed_types = true;
  int64_t max_width_bits_types = 64;
  int64_t max_width_aggregate_types = 1024;
  bool emit_loops = true;
  bool emit_gate = true;
  bool generate_proc = false;
  bool emit_stateless_proc = false;
  // TODO(https://github.com/google/xls/issues/1138): Switch this to default
  // true.
  bool emit_zero_width_bits_types = false;

  static absl::StatusOr<AstGeneratorOptions> FromProto(
      const AstGeneratorOptionsProto& proto);
  AstGeneratorOptionsProto ToProto() const;

  static AstGeneratorOptionsProto DefaultOptionsProto() {
    return AstGeneratorOptions{}.ToProto();
  }
};
bool AbslParseFlag(std::string_view text,
                   AstGeneratorOptions* ast_generator_options,
                   std::string* error);
std::string AbslUnparseFlag(const AstGeneratorOptions& ast_generator_options);

// Type that generates a random module for use in fuzz testing; i.e.
//
//    std::mt19937_64 rng;
//    AstGenerator g(AstGeneratorOptions(), rng);
//    auto [f, module] = g.GenerateFunctionInModule().value();
//
// Where if is the main entry point inside of the returned module.
class AstGenerator {
 public:
  // The random generator must be alive for the lifetime of the object.
  AstGenerator(AstGeneratorOptions options, absl::BitGenRef bit_gen);

  // Generates the entity with name "name" in a module named "module_name".
  absl::StatusOr<AnnotatedModule> Generate(const std::string& top_entity_name,
                                           const std::string& module_name);

 private:
  // We include RNG helper functions to help simplify circumstances where the
  // Abseil Random library does not provide native interfaces (e.g., choosing a
  // random item from a list of choices, or picking a random integer from a
  // shifted Poisson distribution), or where it uses terminology that is not
  // widely known (e.g., Bernoulli distribution for a weighted-coin-flip).

  // Returns a random boolean, true with the given probability.
  //
  // NOTE: This is significantly more efficient (in entropy, and possibly
  //       in runtime) than generating a random float and comparing it to a
  //       cutoff probability.
  bool RandomBool(double true_probability = 0.5) {
    return absl::Bernoulli(bit_gen_, true_probability);
  }

  // Returns a random integer with the given expected value from a distribution
  // which tails off exponentially in both directions over the range
  // [lower_limit, inf). Useful for picking a number around some value with
  // decreasing likelihood of picking something far away from the expected
  // value.  The underlying distribution is a shifted Poisson distribution. See:
  // https://en.wikipedia.org/wiki/Poisson_distribution.
  int64_t RandomIntWithExpectedValue(double expected_value,
                                     int64_t lower_limit = 0) {
    const double mean = expected_value - static_cast<double>(lower_limit);
    return lower_limit + absl::Poisson<int64_t>(bit_gen_, mean);
  }

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

  static bool IsTypeRef(const TypeAnnotation* t);
  static bool IsBits(const TypeAnnotation* t);
  static bool IsUBits(const TypeAnnotation* t);
  static bool IsArray(const TypeAnnotation* t);
  static bool IsTuple(const TypeAnnotation* t);
  static bool IsToken(const TypeAnnotation* t);
  static bool IsChannel(const TypeAnnotation* t);
  static bool IsNil(const TypeAnnotation* t);
  static bool IsBuiltinBool(const TypeAnnotation* type) {
    if (auto* builtin_type = dynamic_cast<const BuiltinTypeAnnotation*>(type)) {
      return !builtin_type->GetSignedness() && builtin_type->GetBitCount() == 1;
    }
    return false;
  }
  // Helper that returns the signedness of type when converted to a builtin bits
  // type. Returns an error status if the type is not a builtin bits type.
  static absl::StatusOr<bool> BitsTypeIsSigned(const TypeAnnotation* type);

  // Helper that returns the bit count of type when converted to a builtin bits
  // type. Returns an error status if the type is not a builtin bits type.
  absl::StatusOr<int64_t> BitsTypeGetBitCount(TypeAnnotation* type);

  static std::tuple<std::vector<Expr*>, std::vector<TypeAnnotation*>,
                    std::vector<LastDelayingOp>>
  Unzip(absl::Span<const TypedExpr> typed_exprs);

  static bool EnvContainsArray(const Env& e);
  static bool EnvContainsTuple(const Env& e);
  static bool EnvContainsToken(const Env& e);
  static bool EnvContainsChannel(const Env& e);
  static absl::StatusOr<bool> ContainsToken(const TypeAnnotation* type);
  static bool ContainsTypeRef(const TypeAnnotation* type);

  // Generates a function with name "name", returning the minimum number of
  // stages the function can be scheduled in.
  absl::StatusOr<int64_t> GenerateFunctionInModule(const std::string& name);

  // Generates a proc with name "name", returning the minimum number of stages
  // the proc can be scheduled in.
  absl::StatusOr<int64_t> GenerateProcInModule(const std::string& name);

  // Generate a DSLX function with the given name. call_depth is the current
  // depth of the call stack (if any) calling this function to be
  // generated. param_types, if given, defines the number and types of the
  // parameters.
  absl::StatusOr<AnnotatedFunction> GenerateFunction(
      const std::string& name, int64_t call_depth,
      absl::Span<const AnnotatedType> param_types);

  // Generate the proc's config function with the given name and proc parameters
  // (proc members).
  absl::StatusOr<Function*> GenerateProcConfigFunction(
      std::string name, absl::Span<Param* const> proc_params);

  // Generate the proc's next function with the given name.
  absl::StatusOr<AnnotatedFunction> GenerateProcNextFunction(std::string name);

  // Generate a function to return a constant with the given TypeAnnotation to
  // serve as a Proc's [required] init function.
  absl::StatusOr<Function*> GenerateProcInitFunction(
      std::string_view name, TypeAnnotation* return_type);

  // Generate a DSLX proc with the given name.
  absl::StatusOr<AnnotatedProc> GenerateProc(const std::string& name);

  // Chooses a value from the environment that satisfies the predicate "take",
  // or returns nullopt if none exists.
  std::optional<TypedExpr> ChooseEnvValueOptional(
      Env* env, const std::function<bool(const TypedExpr&)>& take = nullptr);

  // As above, but takes a type to compare for equality.
  std::optional<TypedExpr> ChooseEnvValueOptional(Env* env,
                                                  const TypeAnnotation* type) {
    return ChooseEnvValueOptional(env, [type](const TypedExpr& e) {
      return e.type->ToString() == type->ToString();
    });
  }

  absl::StatusOr<TypedExpr> ChooseEnvValue(
      Env* env, const std::function<bool(const TypedExpr&)>& take = nullptr);

  // As above, but takes a type to compare for equality.
  absl::StatusOr<TypedExpr> ChooseEnvValue(Env* env,
                                           const TypeAnnotation* type) {
    return ChooseEnvValue(env, [type](const TypedExpr& e) {
      return e.type->ToString() == type->ToString();
    });
  }

  // Return all values from the environment which satisty the given predicate.
  std::vector<TypedExpr> GatherAllValues(
      Env* env, const std::function<bool(const TypedExpr&)>& take);

  // As above, but takes a type to compare for equality.
  std::vector<TypedExpr> GatherAllValues(Env* env, const TypeAnnotation* type) {
    return GatherAllValues(env, [type](const TypedExpr& e) {
      return e.type->ToString() == type->ToString();
    });
  }

  // Returns a random bits-types value from the environment.
  absl::StatusOr<TypedExpr> ChooseEnvValueBits(
      Env* env, std::optional<int64_t> bit_count = std::nullopt) {
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
      Env* env, std::optional<int64_t> bit_count = std::nullopt);

  absl::StatusOr<TypedExpr> ChooseEnvValueUBits(
      Env* env, std::optional<int64_t> bit_count = std::nullopt) {
    auto want_expr = [&](const TypedExpr& e) -> bool {
      if (!IsUBits(e.type)) {
        return false;
      }
      if (bit_count.has_value() &&
          GetTypeBitCount(e.type) != bit_count.value()) {
        return false;
      }
      return true;
    };
    return ChooseEnvValue(env, want_expr);
  }

  absl::StatusOr<TypedExpr> ChooseEnvValueArray(
      Env* env, std::function<bool(ArrayTypeAnnotation*)> take = [](auto _) {
        return true;
      }) {
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

  // Generates the body of a function AST node. call_depth is the depth of the
  // call stack (via map or other function-calling operation) for the function
  // being generated, and `ctx.env` contains the parameters as NameRefs.
  absl::StatusOr<TypedExpr> GenerateBody(int64_t call_depth, Context* ctx);

  absl::StatusOr<TypedExpr> GenerateUnop(Context* ctx);

  absl::StatusOr<Expr*> GenerateUmin(TypedExpr arg, int64_t other);

  // Generates a bit slice AST node.
  absl::StatusOr<TypedExpr> GenerateBitSlice(Context* ctx);

  // Generates one of the bitwise reductions as an Invocation node.
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

  // Generates an expression AST node and returns it. call_depth is the depth of
  // the call stack (via map or other function-calling operation) for the
  // function being generated.
  absl::StatusOr<TypedExpr> GenerateExpr(int64_t call_depth, Context* ctx);

  // Generates an invocation of the one_hot_sel builtin.
  absl::StatusOr<TypedExpr> GenerateOneHotSelectBuiltin(Context* ctx);

  // Generates an invocation of the priority_sel builtin.
  absl::StatusOr<TypedExpr> GeneratePrioritySelectBuiltin(Context* ctx);

  // Generates an invocation of the signex builtin.
  absl::StatusOr<TypedExpr> GenerateSignExtendBuiltin(Context* ctx);

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
  //
  // The caller can optionally capture the bits value created in the AST via the
  // "out" param.
  TypedExpr GenerateNumberWithType(
      std::optional<BitsAndSignedness> bas = std::nullopt, Bits* out = nullptr);

  // Generates String AST node with a string literal of 'char_count'.
  String* GenerateString(int64_t char_count);

  // Generates an invocation of the map builtin.
  absl::StatusOr<TypedExpr> GenerateMap(int64_t call_depth, Context* ctx);

  // Generates an invocation of a function.
  absl::StatusOr<TypedExpr> GenerateInvoke(int64_t call_depth, Context* ctx);

  // Generates a call to a unary builtin function.
  absl::StatusOr<TypedExpr> GenerateUnopBuiltin(Context* ctx);

  // Generates a parameter of the given type, with `last_delaying_op` and
  // `min_stage` matching the provided specification.
  AnnotatedParam GenerateParam(AnnotatedType type);

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

  // Generates an expression with type 'type'.
  absl::StatusOr<TypedExpr> GenerateExprOfType(Context* ctx,
                                               TypeAnnotation* type);

  // Generate a MatchArmPattern with type 'type'. The pattern is represented as
  // an xls::dslx::NameDefTree.
  absl::StatusOr<NameDefTree*> GenerateMatchArmPattern(
      Context* ctx, const TypeAnnotation* type);

  // Generate a Match expression.
  absl::StatusOr<TypedExpr> GenerateMatch(Context* ctx);

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
  // As part of this process, a TypeAlias is created and added to the
  // currently-active set.
  TypeRefTypeAnnotation* MakeTypeRefTypeAnnotation(TypeAnnotation* type) {
    CHECK(type != nullptr);
    std::string type_name = GenSym();
    NameDef* name_def = MakeNameDef(type_name);
    auto* type_alias = module_->Make<TypeAlias>(fake_span_, *name_def, *type,
                                                /*is_public=*/false);
    auto* type_ref = module_->Make<TypeRef>(fake_span_, type_alias);
    type_aliases_.push_back(type_alias);
    type_bit_counts_[type_ref->ToString()] = GetTypeBitCount(type);
    return module_->Make<TypeRefTypeAnnotation>(
        fake_span_, type_ref, /*parametrics=*/std::vector<ExprOrType>{});
  }

  // Generates a logical binary operation (e.g. and, xor, or).
  absl::StatusOr<TypedExpr> GenerateLogicalOp(Context* ctx);

  Expr* MakeSel(Expr* test, Expr* lhs, Expr* rhs) {
    StatementBlock* consequent = module_->Make<StatementBlock>(
        fake_span_, std::vector<Statement*>{module_->Make<Statement>(lhs)},
        /*trailing_semi=*/false);
    StatementBlock* alternate = module_->Make<StatementBlock>(
        fake_span_, std::vector<Statement*>{module_->Make<Statement>(rhs)},
        /*trailing_semi=*/false);
    return module_->Make<Conditional>(fake_span_, test, consequent, alternate);
  }
  Expr* MakeGe(Expr* lhs, Expr* rhs) {
    return module_->Make<Binop>(fake_span_, BinopKind::kGe, lhs, rhs);
  }

  // Creates a number AST node with value 'value' represented in a decimal
  // format.
  Number* MakeNumber(int64_t value, TypeAnnotation* type = nullptr);

  // Creates a number AST node with value 'value' of type 'type' represented in
  // the format specified.
  Number* MakeNumberFromBits(const Bits& value, TypeAnnotation* type,
                             FormatPreference format_preference);

  // Creates a number AST node with value 'value' of boolean type represented in
  // the format specified.
  Number* MakeBool(
      bool value, FormatPreference format_preference = FormatPreference::kHex) {
    return MakeNumberFromBits(
        UBits(value ? 1 : 0, 1),
        MakeTypeAnnotation(/*is_signed=*/false, /*width=*/1),
        format_preference);
  }

  // Creates a number AST node with value 'value' of type 'type' represented in
  // a randomly chosen format between binary, decimal, hex and, when possible,
  // a "character" number. Note that these function expect a builtin type or a
  // one-dimensional array of builtin types.
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
  std::string GenSym();

  // Returns the (flattened) bit count of the given type.
  int64_t GetTypeBitCount(const TypeAnnotation* type);

  // Returns the expression as a concrete value. Returns an error if `expr` is
  // not a number or a known constant, or if `expr` does not fit in 64-bits.
  absl::StatusOr<uint64_t> GetExprAsUint64(Expr* expr);

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
      int64_t value, std::optional<int64_t> want_width = std::nullopt);

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

  struct ProcProperties {
    // A list of the state types in the proc's next function. Currently, at most
    // a single state is supported. The order of types as they appear in the
    // container mirrors the order present in the proc's next function.
    std::vector<TypeAnnotation*> state_types;

    // Parameters of the proc config function.
    std::vector<Param*> config_params;

    // Members of the proc.
    std::vector<ProcMember*> members;
  };

  absl::BitGenRef bit_gen_;

  const AstGeneratorOptions options_;

  const Pos fake_pos_;
  const Span fake_span_;

  absl::btree_set<BinopKind> binops_;

  std::unique_ptr<Module> module_;

  int64_t next_name_index_ = 0;

  // Functions created during generation.
  std::vector<Function*> functions_;

  // Types defined during module generation.
  std::vector<TypeAlias*> type_aliases_;

  // Widths of the aggregate types, indexed by TypeAnnotation::ToString().
  absl::flat_hash_map<std::string, int64_t> type_bit_counts_;

  // Set of constants defined during module generation.
  absl::btree_map<std::string, ConstantDef*> constants_;

  // Contains properties of the generated proc.
  ProcProperties proc_properties_;
};

}  // namespace xls::dslx

#endif  // XLS_FUZZER_AST_GENERATOR_H_
