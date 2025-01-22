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

#include <cstdint>
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
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {
namespace {

using ::testing::ElementsAre;
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

TEST(TransitiveClosureTest, SimpleDense) {
  std::vector<InlineBitmap> rel{
      InlineBitmap::FromBitsLsbIs0({false, false, true, false, false}),
      InlineBitmap::FromBitsLsbIs0({false, false, false, false, false}),
      InlineBitmap::FromBitsLsbIs0({false, true, false, false, false}),
      InlineBitmap::FromBitsLsbIs0({true, false, false, false, false}),
      InlineBitmap::FromBitsLsbIs0({false, true, false, false, true}),
  };
  std::vector<InlineBitmap> tc = TransitiveClosure(rel);
  EXPECT_THAT(
      tc, ElementsAre(
              InlineBitmap::FromBitsLsbIs0({false, true, true, false, false}),
              InlineBitmap::FromBitsLsbIs0({false, false, false, false, false}),
              InlineBitmap::FromBitsLsbIs0({false, true, false, false, false}),
              InlineBitmap::FromBitsLsbIs0({true, true, true, false, false}),
              InlineBitmap::FromBitsLsbIs0({false, true, false, false, true})));
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

std::vector<InlineBitmap> RandomDenseRelation(int64_t node_cnt,
                                              absl::BitGenRef rng) {
  int64_t byte_cnt = CeilOfRatio(node_cnt, int64_t{8});
  std::vector<InlineBitmap> res;
  res.reserve(node_cnt);
  for (int64_t i = 0; i < node_cnt; ++i) {
    std::vector<uint8_t> bytes;
    bytes.reserve(byte_cnt);
    for (int64_t j = 0; j < byte_cnt; ++j) {
      bytes.push_back(absl::Uniform<uint8_t>(rng));
    }
    res.push_back(InlineBitmap::FromBytes(node_cnt, bytes));
  }
  return res;
}

void BM_RandomDenseRelation(benchmark::State& state) {
  absl::BitGen rng;
  for (auto _ : state) {
    state.PauseTiming();
    // Avoid the copy being counted, since it is a non-trivial part of the
    // timing.
    std::vector<InlineBitmap> rel = RandomDenseRelation(state.range(0), rng);
    absl::Span<InlineBitmap> span = absl::MakeSpan(rel);
    state.ResumeTiming();
    span = TransitiveClosure(span);
    benchmark::DoNotOptimize(span);
    benchmark::DoNotOptimize(rel);
  }
}
BENCHMARK(BM_RandomDenseRelation)->Range(500, 10000);
}  // namespace
}  // namespace xls
