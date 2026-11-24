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

#ifndef XLS_INTERPRETER_PROC_EVALUATOR_TEST_BASE_H_
#define XLS_INTERPRETER_PROC_EVALUATOR_TEST_BASE_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_evaluator.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {

// Matches a container containing elements matching `seq` in order (not
// necessarily contiguously).
// Note: Can be replaced with `::testing::ContainsSubsequence` once
// `MODULE.bazel` upgrades to `googletest` >= 1.18.0.
MATCHER_P(ContainsSubsequence, seq,
          absl::StrCat(
              negation ? "does not contain" : "contains", " subsequence [",
              absl::StrJoin(seq, ", ",
                            [](std::string* out, const auto& elem) {
                              using ElemType = std::decay_t<decltype(elem)>;
                              if constexpr (requires(std::ostringstream os) {
                                              elem.DescribeTo(&os);
                                            }) {
                                std::ostringstream os;
                                elem.DescribeTo(&os);
                                absl::StrAppend(out, os.str());
                              } else {
                                absl::StrAppend(
                                    out,
                                    ::testing::DescribeMatcher<ElemType>(elem));
                              }
                            }),
              "]")) {
  using Element = typename std::decay_t<decltype(arg)>::value_type;
  auto it = arg.begin();
  int64_t idx = 0;
  for (const auto& elem : seq) {
    auto matcher = ::testing::SafeMatcherCast<const Element&>(elem);
    bool found = false;
    while (it != arg.end()) {
      ::testing::StringMatchResultListener dummy_listener;
      if (::testing::ExplainMatchResult(matcher, *it, &dummy_listener)) {
        found = true;
        ++it;
        break;
      }
      ++it;
    }
    if (!found) {
      *result_listener << "where element #" << idx << " ("
                       << ::testing::DescribeMatcher<const Element&>(matcher)
                       << ") was not found in remaining sequence";
      return false;
    }
    ++idx;
  }
  return true;
}

template <typename T>
inline auto ContainsSubsequence(std::initializer_list<T> seq) {
  return ContainsSubsequence(std::vector<T>(seq));
}

class ProcEvaluatorTestParam {
 public:
  ProcEvaluatorTestParam(
      std::function<std::unique_ptr<ProcEvaluator>(Proc*, ChannelQueueManager*)>
          evaluator_factory,
      std::function<std::unique_ptr<ChannelQueueManager>(Package*)>
          queue_manager_factory,
      bool supports_observers = true)
      : evaluator_factory_(std::move(evaluator_factory)),
        queue_manager_factory_(std::move(queue_manager_factory)),
        supports_observers_(supports_observers) {}
  ProcEvaluatorTestParam() = default;

  std::unique_ptr<ChannelQueueManager> CreateQueueManager(
      Package* package) const {
    return queue_manager_factory_(package);
  }

  std::unique_ptr<ProcEvaluator> CreateEvaluator(
      Proc* proc, ChannelQueueManager* queue_manager) const {
    return evaluator_factory_(proc, queue_manager);
  }

  bool supports_observers() const { return supports_observers_; }

 private:
  std::function<std::unique_ptr<ProcEvaluator>(Proc*, ChannelQueueManager*)>
      evaluator_factory_;
  std::function<std::unique_ptr<ChannelQueueManager>(Package*)>
      queue_manager_factory_;
  bool supports_observers_;
};

// A suite of test which can be run against arbitrary ProcEvaluator
// implementations. Users should instantiate with an INSTANTIATE_TEST_SUITE_P
// macro.
class ProcEvaluatorTestBase
    : public IrTestBase,
      public testing::WithParamInterface<ProcEvaluatorTestParam> {};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_EVALUATOR_TEST_BASE_H_
