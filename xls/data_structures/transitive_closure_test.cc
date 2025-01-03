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

#include "xls/data_structures/transitive_closure.h"

#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/strings/str_cat.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

using V = std::string;

TEST(TransitiveClosureTest, Simple) {
  HashRelation<V> rel;
  rel["foo"].insert("bar");
  rel["bar"].insert("baz");
  rel["bar"].insert("qux");
  rel["baz"].insert("qux");
  rel["foo2"].insert("baz");
  HashRelation<V> tc = TransitiveClosure<V>(rel);
  EXPECT_THAT(tc.at("foo"), UnorderedElementsAre("bar", "baz", "qux"));
  EXPECT_THAT(tc.at("foo2"), UnorderedElementsAre("baz", "qux"));
  EXPECT_THAT(tc.at("bar"), UnorderedElementsAre("baz", "qux"));
  EXPECT_THAT(tc.at("baz"), UnorderedElementsAre("qux"));
  EXPECT_FALSE(tc.contains("qux"));
}

HashRelation<V> RandomRelation(std::vector<V> nodes, double p,
                               absl::BitGenRef rng) {
  HashRelation<V> rel;
  for (const V& node : nodes) {
    rel[node] = {};
    for (const V& neighbor : nodes) {
      if (absl::Bernoulli(rng, p)) {
        rel[node].insert(neighbor);
      }
    }
  }
  return rel;
}

void BM_RandomRelation(benchmark::State& state) {
  std::seed_seq seq = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  std::mt19937_64 rng(seq);
  std::vector<V> nodes;
  nodes.reserve(state.range(0));
  for (int i = 0; i < state.range(0); ++i) {
    nodes.push_back(absl::StrCat(i));
  }
  HashRelation<V> rel = RandomRelation(nodes, 2.0 / state.range(0), rng);
  for (auto _ : state) {
    HashRelation<V> tc = TransitiveClosure(rel);
    benchmark::DoNotOptimize(tc);
  }
}
BENCHMARK(BM_RandomRelation)->Range(5, 500);

}  // namespace
}  // namespace xls
