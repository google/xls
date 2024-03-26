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

#ifndef XLS_DATA_STRUCTURES_SUBMODULAR_H_
#define XLS_DATA_STRUCTURES_SUBMODULAR_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"

namespace xls {

enum class MinimizeMode {
  MMinI,
  MMinII,
  Alternating,
};

struct MinimizeOptions {
  MinimizeMode mode;
  std::optional<int64_t> seed;
  int64_t rounds;
};

// A submodular function is a function from a set to the reals that has
// "diminishing returns" or "diminishing costs". Formally, this means that
// for any submodular `f : 2ᴱ → R` and `X ⊆ Y ⊆ E` and `x ∈ E \ Y`,
// `f(X ∪ {x}) - f(X) ≥ f(Y ∪ {x}) - f(Y)`.
//
// - Submodular functions may be monotone, but do not have to be.
// - A nonnegative linear combination of submodular functions is submodular.
// - Replacing the ≥ by ≤ in the definition of submodular gives the definition
//   of supermodularity.
// - A function that is both submodular and supermodular is called modular.
// - Summing a function `f : E → R` over `2ᴱ` (yielding `g : 2ᴱ → R`) gives a
//   modular function.
//
// Examples of submodular functions:
//
// - Budget-additive functions: min(b, f(S)) where b ∈ R⁺ and f is modular.
// - For a set C of pieces of code, take any subset, run common subexpression
//   elimination on it, and return the resulting number of nodes.
// - The amount of area covered by a set of discs confined to some region.
// - The rank of a matrix formed by taking a subset of the columns of a matrix.
// - The entropy of a subset of a set of random variables.
// - The mutual information of a subset of a set of random variables.
//
// The `MMinI`, `MMinII`, and `AlternatingMin` algorithms are all based on this
// paper: http://proceedings.mlr.press/v28/iyer13-supp.pdf
// Fast Semidifferential-based Submodular Function Optimization: Extended Ver.
template <typename T, typename C = std::less<T>>
class SubmodularFunction {
 public:
  SubmodularFunction(
      const absl::btree_set<T, C>& universe,
      const std::function<double(const absl::btree_set<T, C>&)>& function)
      : universe_(universe), function_(function) {}

  // Returns the set of all valid elements to pass into this function.
  const absl::btree_set<T, C>& Universe() const { return universe_; }

  // Call the function on a given subset of the universe.
  double Call(const absl::btree_set<T, C>& set) const { return function_(set); }

  // Computes `f(X ∪ {x}) - f(X)`, which is called the incremental value of `f`.
  double IncrementalValue(const absl::btree_set<T, C>& set, const T& element) {
    absl::btree_set<T> incremented = set;
    incremented.insert(element);
    return function_(incremented) - function_(set);
  }

  // Computes `f(X) - f(X \ {x})`, which is called the decremental value of `f`.
  double DecrementalValue(const absl::btree_set<T, C>& set, const T& element) {
    absl::btree_set<T> decremented = set;
    decremented.erase(element);
    return function_(set) - function_(decremented);
  }

  // Approximate minimization of a submodular function.
  absl::btree_set<T, C> ApproxMinimize(const MinimizeOptions& options) {
    CHECK_GT(options.rounds, 0);
    double best_cost = std::numeric_limits<double>::max();
    absl::btree_set<T, C> best;

    int64_t seed;

    if (options.seed.has_value()) {
      seed = options.seed.value();
    } else {
      seed = absl::Uniform<int64_t>(absl::IntervalClosed, absl::BitGen(),
                                    std::numeric_limits<int64_t>::min(),
                                    std::numeric_limits<int64_t>::max());
    }

    std::mt19937_64 bit_gen(seed);

    for (int64_t i = 0; i < options.rounds + 2; ++i) {
      absl::btree_set<T, C> random;

      if (i == 0) {
        // Keep random empty
      } else if (i == 1) {
        random = universe_;
      } else {
        for (const T& element : universe_) {
          if (absl::Bernoulli(bit_gen, 0.5)) {
            random.insert(element);
          }
        }
      }

      absl::btree_set<T, C> newest;

      switch (options.mode) {
        case MinimizeMode::MMinI:
          newest = MMinI(random);
          break;
        case MinimizeMode::MMinII:
          newest = MMinII(random);
          break;
        case MinimizeMode::Alternating:
          newest = AlternatingMin(random);
          break;
      }

      double newest_cost = function_(newest);
      if (newest_cost < best_cost) {
        best = newest;
        best_cost = newest_cost;
      }
    }
    return best;
  }

  // Uses the alternating algorithm for submodular minimization.
  //
  // Should be used with random initialization.
  absl::btree_set<T, C> AlternatingMin(const absl::btree_set<T, C>& initial) {
    absl::btree_set<T, C> x = initial;
    std::optional<absl::btree_set<T>> x_old;
    do {
      x_old = x;
      x = MMinII(MMinI(x));
    } while (x_old.value() != x);
    return x;
  }

  // Uses the MMin-I algorithm for submodular minimization.
  absl::btree_set<T, C> MMinI(const absl::btree_set<T, C>& initial) {
    absl::btree_set<T, C> x = initial;
    std::optional<absl::btree_set<T, C>> x_old;
    do {
      x_old = x;
      // By Lemma 5.4, the body of the loop is equivalent to
      // `X_{t + 1} = X_t ∪ {j | f(j | X_t) < 0}`
      absl::btree_set<T, C> x_new = x;
      for (const T& j : universe_) {
        if (IncrementalValue(x, j) < 0) {
          x_new.insert(j);
        }
      }
      x = x_new;
    } while (x_old.value() != x);
    return x;
  }

  // Uses the MMin-II algorithm for submodular minimization.
  absl::btree_set<T, C> MMinII(const absl::btree_set<T, C>& initial) {
    absl::btree_set<T, C> x = initial;
    std::optional<absl::btree_set<T, C>> x_old;
    do {
      x_old = x;
      // By Lemma 5.4, the body of the loop is equivalent to
      // `X_{t + 1} = X_t \ {j | f(j | X_t \ {j}) > 0}`
      absl::btree_set<T, C> x_new = x;
      for (const T& j : universe_) {
        if (DecrementalValue(x, j) > 0) {
          x_new.erase(j);
        }
      }
      x = x_new;
    } while (x_old.value() != x);
    return x;
  }

 private:
  absl::btree_set<T, C> universe_;
  std::function<double(const absl::btree_set<T, C>&)> function_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_SUBMODULAR_H_
