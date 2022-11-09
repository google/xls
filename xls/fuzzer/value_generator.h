// Copyright 2022 The XLS Authors
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
#ifndef XLS_FUZZER_VALUE_GENERATOR_H_
#define XLS_FUZZER_VALUE_GENERATOR_H_

#include <random>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/interp_value.h"

namespace xls {

// Contains logic to generate random values for use by fuzzer components.
class ValueGenerator {
 public:
  explicit ValueGenerator(std::mt19937 rng) : rng_(std::move(rng)) {}

  // Returns a random boolean value.
  bool RandomBool();

  // Returns a random integer with the given expected value from a distribution
  // which tails off exponentially in both directions over the range
  // [lower_limit, inf). Useful for picking a number around some value with
  // decreasing likelihood of picking something far away from the expected
  // value.  The underlying distribution is a Poisson distribution. See:
  // https://en.wikipedia.org/wiki/Poisson_distribution
  int64_t RandomIntWithExpectedValue(float expected_value,
                                     int64_t lower_limit = 0);

  // Returns a random float value in the range [0.0, 1.0).
  float RandomFloat();

  // Returns a random double value in the range [0.0, 1.0).
  double RandomDouble();

  // Returns a random integer in the range [0, limit).
  int64_t RandRange(int64_t limit);

  // Returns a random integer in the range [start, limit).
  int64_t RandRange(int64_t start, int64_t limit);

  // Returns a triangular distribution biased towards the zero side, with limit
  // as the exclusive limit.
  int64_t RandRangeBiasedTowardsZero(int64_t limit);

  // Generates a Bits object with the given size.
  Bits GenerateBits(int64_t bit_count);

  // Generates a BitsTyped InterpValue with the given attributes.
  absl::StatusOr<dslx::InterpValue> GenerateBitValue(int64_t bit_count,
                                                     bool is_signed);

  // Returns a single value of the given type.
  absl::StatusOr<dslx::InterpValue> GenerateInterpValue(
      const dslx::ConcreteType& arg_type,
      absl::Span<const dslx::InterpValue> prior);

  // Returns randomly generated values of the given types.
  absl::StatusOr<std::vector<dslx::InterpValue>> GenerateInterpValues(
      absl::Span<const dslx::ConcreteType* const> arg_types);

  // Randomly generates an Expr* holding a value of the given type.
  // Note: Currently, AstGenerator only produces single-dimensional arrays with
  // [AST] Number-typed or ConstantDef-defined sizes. If that changes, then this
  // function will need to be modified.
  absl::StatusOr<dslx::Expr*> GenerateDslxConstant(dslx::Module* module,
                                                   dslx::TypeAnnotation* type);

  std::mt19937& rng() { return rng_; }

 private:
  absl::StatusOr<dslx::InterpValue> GenerateUnbiasedValue(
      const dslx::BitsType& bits_type);

  // Evaluates the given Expr* (holding the declaration of an
  // ArrayTypeAnnotation's size) and returns its resolved integer value.
  // This relies on current behavior of AstGenerator, namely that array dims are
  // pure Number nodes or are references to ConstantDefs (potentially via a
  // series of NameRefs) whose values are Numbers.
  absl::StatusOr<int64_t> GetArraySize(const dslx::Expr* dim);

  std::mt19937 rng_;
};

}  // namespace xls

#endif  // XLS_FUZZER_VALUE_GENERATOR_H_
