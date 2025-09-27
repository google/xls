// Copyright 2025 The XLS Authors
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

#include "xls/ir/node_map.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/benchmark_support.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using testing::_;
using testing::Eq;
using testing::Pair;
using testing::UnorderedElementsAre;

template <typename T>
using TestNodeMap = NodeMap<T, ForceAllowNodeMap>;

struct EmplaceOnly {
  inline static int constructions = 0;
  int x;
  explicit EmplaceOnly(int i) : x(i) { ++constructions; }
  EmplaceOnly(const EmplaceOnly&) = delete;
  EmplaceOnly& operator=(const EmplaceOnly&) = delete;
  EmplaceOnly(EmplaceOnly&&) = default;
  EmplaceOnly& operator=(EmplaceOnly&&) = default;
  bool operator==(const EmplaceOnly& other) const { return x == other.x; }
};

class NodeMapTest : public IrTestBase {};

TEST_F(NodeMapTest, Basic) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  auto c = fb.Add(a, b);
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<int64_t> map;
  // Set.
  map[a.node()] = 1;
  map[b.node()] = 2;

  EXPECT_THAT(map.at(a.node()), Eq(1));
  EXPECT_THAT(map.at(b.node()), Eq(2));
  EXPECT_TRUE(map.contains(a.node()));
  EXPECT_TRUE(map.contains(b.node()));
  EXPECT_FALSE(map.contains(c.node()));
  EXPECT_EQ(map.count(a.node()), 1);
  EXPECT_EQ(map.count(b.node()), 1);
  EXPECT_EQ(map.count(c.node()), 0);
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2))));

  // Update.
  map[a.node()] += 5;

  EXPECT_THAT(map.at(a.node()), Eq(6));
  EXPECT_THAT(map.at(b.node()), Eq(2));
  EXPECT_TRUE(map.contains(a.node()));
  EXPECT_TRUE(map.contains(b.node()));
  EXPECT_FALSE(map.contains(c.node()));
  EXPECT_EQ(map.count(a.node()), 1);
  EXPECT_EQ(map.count(b.node()), 1);
  EXPECT_EQ(map.count(c.node()), 0);
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(6)),
                                        Pair(m::Param("bar"), Eq(2))));
  // erase.
  map.erase(a.node());

  EXPECT_THAT(map.at(b.node()), Eq(2));
  EXPECT_TRUE(map.contains(b.node()));
  EXPECT_FALSE(map.contains(c.node()));
  EXPECT_FALSE(map.contains(a.node()));
  EXPECT_EQ(map.count(a.node()), 0);
  EXPECT_EQ(map.count(b.node()), 1);
  EXPECT_EQ(map.count(c.node()), 0);
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("bar"), Eq(2))));
}

TEST_F(NodeMapTest, Find) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  auto c = fb.Add(a, b);
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<int64_t> map;
  // Set.
  map[a.node()] = 1;
  map[b.node()] = 2;

  EXPECT_THAT(map.find(c.node()), Eq(map.end()));
  EXPECT_THAT(map.find(a.node()),
              testing::Pointee(Pair(m::Param("foo"), Eq(1))));
  EXPECT_THAT(map.find(b.node()),
              testing::Pointee(Pair(m::Param("bar"), Eq(2))));
  // Update with iterator.
  map.find(a.node())->second = 33;
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(33)),
                                        Pair(m::Param("bar"), Eq(2))));
}

TEST_F(NodeMapTest, Copy) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  fb.Add(a, b);
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<int64_t> map;
  {
    TestNodeMap<int64_t> map1;
    // Set.
    map1[a.node()] = 1;
    map1[b.node()] = 2;
    map = map1;
    map[a.node()] = 4;
    map1[b.node()] = 6;
    EXPECT_THAT(map1, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                           Pair(m::Param("bar"), Eq(6))));
    EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(4)),
                                          Pair(m::Param("bar"), Eq(2))));
  }
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(4)),
                                        Pair(m::Param("bar"), Eq(2))));
}

TEST_F(NodeMapTest, IterConstructor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  fb.Add(a, b);
  XLS_ASSERT_OK(fb.Build().status());
  std::vector<std::pair<Node*, int64_t>> v = {{a.node(), 1}, {b.node(), 2}};
  TestNodeMap<int64_t> map(v.begin(), v.end());
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2))));
}

TEST_F(NodeMapTest, Move) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  auto c = fb.Add(a, b);
  XLS_ASSERT_OK(fb.Build().status());

  std::optional<TestNodeMap<int64_t>> opt_map;
  {
    TestNodeMap<int64_t> map1;
    // Set.
    map1[a.node()] = 1;
    map1[b.node()] = 2;
    opt_map.emplace(std::move(map1));
    EXPECT_FALSE(map1.HasPackage());
  }

  TestNodeMap<int64_t> map(*std::move(opt_map));

  EXPECT_THAT(map.find(c.node()), Eq(map.end()));
  EXPECT_THAT(map.find(a.node()),
              testing::Pointee(Pair(m::Param("foo"), Eq(1))));
  EXPECT_THAT(map.find(b.node()),
              testing::Pointee(Pair(m::Param("bar"), Eq(2))));
}

TEST_F(NodeMapTest, Insert) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  auto c = fb.Add(a, b);
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<int64_t> map;
  // Set.
  map[a.node()] = 1;
  map[b.node()] = 2;

  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2))));
  EXPECT_THAT(map.insert(c.node(), 3), Pair(_, true));
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2)),
                                        Pair(m::Add(), Eq(3))));

  EXPECT_THAT(map.insert(a.node(), 3), Pair(map.find(a.node()), false));
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2)),
                                        Pair(m::Add(), Eq(3))));

  EXPECT_THAT(map.insert_or_assign(a.node(), 3),
              Pair(map.find(a.node()), false));
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(3)),
                                        Pair(m::Param("bar"), Eq(2)),
                                        Pair(m::Add(), Eq(3))));

  map.erase(c.node());
  EXPECT_THAT(map.insert_or_assign(c.node(), 7), Pair(_, true));
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(3)),
                                        Pair(m::Param("bar"), Eq(2)),
                                        Pair(m::Add(), Eq(7))));
}

TEST_F(NodeMapTest, Emplace) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<EmplaceOnly> map;
  EmplaceOnly::constructions = 0;
  auto [it, inserted] = map.emplace(a.node(), 42);
  EXPECT_TRUE(inserted);
  EXPECT_TRUE(map.contains(a.node()));
  EXPECT_EQ(map.at(a.node()).x, 42);
  EXPECT_EQ(it->second.x, 42);
  EXPECT_EQ(EmplaceOnly::constructions, 1);
  // Emplace on existing fails but constructs argument.
  auto [it2, inserted2] = map.emplace(a.node(), 44);
  EXPECT_FALSE(inserted2);
  EXPECT_EQ(map.at(a.node()).x, 42);
  EXPECT_EQ(it2->second.x, 42);
  EXPECT_EQ(it2, it);
  EXPECT_EQ(EmplaceOnly::constructions, 2);
}

TEST_F(NodeMapTest, TryEmplace) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<EmplaceOnly> map;
  EmplaceOnly::constructions = 0;
  auto [it, inserted] = map.try_emplace(a.node(), 42);
  EXPECT_TRUE(inserted);
  EXPECT_TRUE(map.contains(a.node()));
  EXPECT_EQ(map.at(a.node()).x, 42);
  EXPECT_EQ(it->second.x, 42);
  EXPECT_EQ(EmplaceOnly::constructions, 1);
  // try_emplace on existing fails and does not construct argument.
  auto [it2, inserted2] = map.try_emplace(a.node(), 44);
  EXPECT_FALSE(inserted2);
  EXPECT_EQ(map.at(a.node()).x, 42);
  EXPECT_EQ(it2->second.x, 42);
  EXPECT_EQ(it2, it);
  EXPECT_EQ(EmplaceOnly::constructions, 1);
}

TEST_F(NodeMapTest, Swap) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<int64_t> map1;
  TestNodeMap<int64_t> map2;
  map1[a.node()] = 1;
  map2[b.node()] = 2;
  map1.swap(map2);
  EXPECT_THAT(map1, UnorderedElementsAre(Pair(m::Param("bar"), Eq(2))));
  EXPECT_THAT(map2, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1))));
}

TEST_F(NodeMapTest, InitializerList) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  XLS_ASSERT_OK(fb.Build().status());

  TestNodeMap<int64_t> map{{a.node(), 1}, {b.node(), 2}};
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2))));
}

TEST_F(NodeMapTest, NodeDeletionRemovesMapElement) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("foo", p->GetBitsType(32));
  auto b = fb.Param("bar", p->GetBitsType(32));
  auto c = fb.Param("baz", p->GetBitsType(32));
  // Remove doesn't like getting rid of the last node.
  fb.Param("other", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  TestNodeMap<int64_t> map{{a.node(), 1}, {b.node(), 2}, {c.node(), 3}};
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("bar"), Eq(2)),
                                        Pair(m::Param("baz"), Eq(3))));
  XLS_ASSERT_OK(f->RemoveNode(b.node()));
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("foo"), Eq(1)),
                                        Pair(m::Param("baz"), Eq(3))));
  XLS_ASSERT_OK(f->RemoveNode(a.node()));
  EXPECT_THAT(map, UnorderedElementsAre(Pair(m::Param("baz"), Eq(3))));
  XLS_ASSERT_OK(f->RemoveNode(c.node()));
  EXPECT_THAT(map, UnorderedElementsAre());
}

absl::Status GenerateFunction(Package* p, benchmark::State& state) {
  FunctionBuilder fb("benchmark", p);
  XLS_RETURN_IF_ERROR(
      benchmark_support::GenerateChain(
          fb, state.range(0), 2, benchmark_support::strategy::BinaryAdd(),
          benchmark_support::strategy::SharedLiteral(UBits(32, 32)))
          .status());
  return fb.Build().status();
}

template <typename Map, typename Setup>
void BM_ReadSome(benchmark::State& v, Setup setup) {
  Package p("benchmark");
  XLS_ASSERT_OK(GenerateFunction(&p, v));
  Function* f = p.functions()[0].get();
  for (auto s : v) {
    Map map;
    setup(map);
    int64_t i = 0;
    for (Node* n : f->nodes()) {
      if (i++ % 4 == 0) {
        map[n].value = i;
      }
    }
    for (int64_t i = 0; i < v.range(1); ++i) {
      for (Node* n : f->nodes()) {
        auto v = map.find(n);
        if (v != map.end()) {
          benchmark::DoNotOptimize(v->second.value);
        }
        benchmark::DoNotOptimize(v);
      }
    }
    for (Node* n : f->nodes()) {
      if (i++ % 3 == 0) {
        map.erase(n);
      }
      if (i++ % 7 == 0) {
        map[n].value = i;
      }
    }
    for (int64_t i = 0; i < v.range(1); ++i) {
      for (Node* n : f->nodes()) {
        auto v = map.find(n);
        if (v != map.end()) {
          benchmark::DoNotOptimize(v->second.value);
        }
        benchmark::DoNotOptimize(v);
      }
    }
  }
}

// Simulate a typical xls map value which has real destructors etc.
struct TestValue {
  int64_t value;
  TestValue() ABSL_ATTRIBUTE_NOINLINE : value(12) {}
  explicit TestValue(int64_t v) ABSL_ATTRIBUTE_NOINLINE : value(v) {
    benchmark::DoNotOptimize(v);
  }
  TestValue(const TestValue& v) ABSL_ATTRIBUTE_NOINLINE : value(v.value) {
    benchmark::DoNotOptimize(v);
  }
  ~TestValue() ABSL_ATTRIBUTE_NOINLINE { benchmark::DoNotOptimize(value); }
  TestValue(TestValue&& v) ABSL_ATTRIBUTE_NOINLINE : value(v.value) {
    benchmark::DoNotOptimize(v);
  }
};
void BM_ReadSomeNodeMap(benchmark::State& v) {
  BM_ReadSome<NodeMap<TestValue>>(v, [](auto& a) {});
}
void BM_ReadSomeFlatMap(benchmark::State& v) {
  BM_ReadSome<absl::flat_hash_map<Node*, TestValue>>(v, [](auto& a) {});
}
void BM_ReadSomeFlatMapReserve(benchmark::State& v) {
  BM_ReadSome<absl::flat_hash_map<Node*, TestValue>>(
      v, [&v](auto& map) { map.reserve(v.range(0)); });
}
BENCHMARK(BM_ReadSomeNodeMap)->RangePair(100, 100000, 1, 100);
BENCHMARK(BM_ReadSomeFlatMap)->RangePair(100, 100000, 1, 100);
BENCHMARK(BM_ReadSomeFlatMapReserve)->RangePair(100, 100000, 1, 100);

}  // namespace
}  // namespace xls
