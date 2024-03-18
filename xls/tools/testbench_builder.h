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

#ifndef XLS_TOOLS_TESTBENCH_BUILDER_H_
#define XLS_TOOLS_TESTBENCH_BUILDER_H_

// Builder classes for XLS Testbench objects.
// Often, much of the functionality of a Testbench doesn't need to be
// specialized per-creation: common code can be used to generate random floats
// or ints, or to print the same. Even composite values, such as std::tuples of
// known types, can be manipulated in some ways without needing custom code.
// These Builder objects enable such usage: a user need only specify what's
// required: "compute expected" and "compute actual" functions and optionally a
// "create shard data" function (for cases needing per-shard (or thread) state).
// For built-in types or STL containers, this may be enough. For other types, or
// cases where, e.g., custom sample generation is desired, the user may specify
// additional functionality.
//
// Default functionality is mainly specified inside testbench_builder_utils.h,
// for example internal::DefaultIndexToInput(). When possible, support for
// common (but missing) types should be added there, instead of in a private
// implementation.
//
// TODO(rspringer): 2021-05-12: Add support for "styles" of default functions,
// e.g., IndexToInputRandom(), SerialIndexToInputRandom(), etc., or
// PrintInputDecimal(), PrintInputHex(), as well as selectors for each.

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>

#include "absl/strings/str_format.h"
#include "xls/tools/testbench.h"
#include "xls/tools/testbench_builder_utils.h"

namespace xls {

// Builder for Testbenches containing ShardData.
template <typename InputT, typename ResultT, typename ShardDataT = void,
          typename EnableT = void>
class TestbenchBuilder {
 public:
  using CompareResultsFnT = std::function<bool(const ResultT&, const ResultT&)>;
  using ComputeFnT = std::function<ResultT(ShardDataT*, InputT)>;
  using CreateShardDataFnT = std::function<std::unique_ptr<ShardDataT>()>;
  using IndexToInputFnT = std::function<InputT(int64_t)>;
  using LogErrorsFnT = std::function<void(int64_t, InputT, ResultT, ResultT)>;
  using PrintInputFnT = std::function<std::string(const InputT&)>;
  using PrintResultFnT = std::function<std::string(const ResultT&)>;

  TestbenchBuilder(ComputeFnT compute_expected, ComputeFnT compute_actual,
                   CreateShardDataFnT create_shard_data)
      : compute_expected_(compute_expected),
        compute_actual_(compute_actual),
        create_shard_data_(create_shard_data) {}

  TestbenchBuilder& SetCompareResultsFn(const CompareResultsFnT& fn) {
    compare_results_ = fn;
    return *this;
  }

  TestbenchBuilder& SetIndexToInputFn(const IndexToInputFnT& fn) {
    index_to_input_ = fn;
    return *this;
  }

  TestbenchBuilder& SetMaxFailures(int64_t max_failures) {
    max_failures_ = max_failures;
    return *this;
  }

  TestbenchBuilder& SetNumSamples(int64_t num_samples) {
    num_samples_ = num_samples;
    return *this;
  }

  TestbenchBuilder& SetNumThreads(int64_t num_threads) {
    num_threads_ = num_threads;
    return *this;
  }

  TestbenchBuilder& SetPrintInputFn(const PrintInputFnT& fn) {
    print_input_ = fn;
    return *this;
  }

  TestbenchBuilder& SetPrintResultFn(const PrintResultFnT& fn) {
    print_result_ = fn;
    return *this;
  }

  TestbenchBuilder& SetLogErrorsFn(const LogErrorsFnT& fn) {
    log_errors_ = fn;
    return *this;
  }

  Testbench<InputT, ResultT, ShardDataT, EnableT> Build();

 private:
  uint64_t num_samples_ = 16 * 1024;
  uint64_t num_threads_ = std::thread::hardware_concurrency();
  int64_t max_failures_ = 1;
  ComputeFnT compute_expected_;
  ComputeFnT compute_actual_;
  std::optional<CompareResultsFnT> compare_results_;
  CreateShardDataFnT create_shard_data_;
  std::optional<IndexToInputFnT> index_to_input_;
  std::optional<PrintInputFnT> print_input_;
  std::optional<PrintResultFnT> print_result_;
  std::optional<LogErrorsFnT> log_errors_;
};

// Builder for Testbenches without ShardData.
template <typename InputT, typename ResultT, typename ShardDataT>
class TestbenchBuilder<
    InputT, ResultT, ShardDataT,
    typename std::enable_if_t<std::is_void<ShardDataT>::value>> {
 public:
  using CompareResultsFnT = std::function<bool(const ResultT&, const ResultT&)>;
  using ComputeFnT = std::function<ResultT(InputT)>;
  using IndexToInputFnT = std::function<InputT(int64_t)>;
  using LogErrorsFnT = std::function<void(int64_t, InputT, ResultT, ResultT)>;
  using PrintInputFnT = std::function<std::string(const InputT&)>;
  using PrintResultFnT = std::function<std::string(const ResultT&)>;

  TestbenchBuilder(ComputeFnT compute_expected, ComputeFnT compute_actual)
      : compute_expected_(compute_expected), compute_actual_(compute_actual) {}

  TestbenchBuilder& SetCompareResultsFn(const CompareResultsFnT& fn) {
    compare_results_ = fn;
    return *this;
  }

  TestbenchBuilder& SetIndexToInputFn(const IndexToInputFnT& fn) {
    index_to_input_ = fn;
    return *this;
  }

  TestbenchBuilder& SetMaxFailures(int64_t max_failures) {
    max_failures_ = max_failures;
    return *this;
  }

  TestbenchBuilder& SetNumIters(int64_t num_samples) {
    num_samples_ = num_samples;
    return *this;
  }

  TestbenchBuilder& SetNumThreads(int64_t num_threads) {
    num_threads_ = num_threads;
    return *this;
  }

  TestbenchBuilder& SetPrintInputFn(const PrintInputFnT& fn) {
    print_input_ = fn;
    return *this;
  }

  TestbenchBuilder& SetPrintResultFn(const PrintResultFnT& fn) {
    print_result_ = fn;
    return *this;
  }

  TestbenchBuilder& SetLogErrorsFn(const LogErrorsFnT& fn) {
    log_errors_ = fn;
    return *this;
  }

  Testbench<InputT, ResultT> Build();

 private:
  uint64_t num_samples_ = 16 * 1024;
  uint64_t num_threads_ = std::thread::hardware_concurrency();
  int64_t max_failures_ = 1;
  ComputeFnT compute_expected_;
  ComputeFnT compute_actual_;
  std::optional<CompareResultsFnT> compare_results_;
  std::optional<IndexToInputFnT> index_to_input_;
  std::optional<PrintInputFnT> print_input_;
  std::optional<PrintResultFnT> print_result_;
  std::optional<LogErrorsFnT> log_errors_;
};

// Shard-data-containing Build() implementation.
template <typename InputT, typename ResultT, typename ShardDataT,
          typename EnableT>
Testbench<InputT, ResultT, ShardDataT, EnableT>
TestbenchBuilder<InputT, ResultT, ShardDataT, EnableT>::Build() {
  auto index_to_input = this->index_to_input_.has_value()
                            ? this->index_to_input_.value()
                            : [](int64_t index) {
                                InputT input;
                                internal::DefaultIndexToInput(index, &input);
                                return input;
                              };
  auto compare_results = this->compare_results_.has_value()
                             ? this->compare_results_.value()
                             : internal::DefaultCompareResults<ResultT>;
  auto print_input = this->print_input_.has_value()
                         ? this->print_input_.value()
                         : [](const InputT& input) {
                             return internal::DefaultPrintValue(input);
                           };
  auto print_result = this->print_result_.has_value()
                          ? this->print_result_.value()
                          : [](const ResultT& result) {
                              return internal::DefaultPrintValue(result);
                            };
  auto log_errors =
      this->log_errors_.has_value()
          ? this->log_errors_.value()
          : [print_input, print_result](int64_t index, const InputT& input,
                                        const ResultT& expected,
                                        const ResultT& actual) {
              internal::DefaultLogError<InputT, ResultT>(
                  index, input, expected, actual, print_input, print_result);
            };
  return Testbench<InputT, ResultT, ShardDataT>(
      /*start=*/0, this->num_samples_, this->num_threads_, this->max_failures_,
      index_to_input, create_shard_data_, this->compute_expected_,
      this->compute_actual_, compare_results, log_errors);
}

// Non-shard-data-containing Build() implementation.
template <typename InputT, typename ResultT, typename ShardDataT>
Testbench<InputT, ResultT> TestbenchBuilder<
    InputT, ResultT, ShardDataT,
    typename std::enable_if_t<std::is_void<ShardDataT>::value>>::Build() {
  auto index_to_input = this->index_to_input_.has_value()
                            ? this->index_to_input_.value()
                            : [](int64_t index) {
                                InputT input;
                                internal::DefaultIndexToInput(index, &input);
                                return input;
                              };
  auto compare_results = this->compare_results_.has_value()
                             ? this->compare_results_.value()
                             : internal::DefaultCompareResults<ResultT>;
  auto print_input = this->print_input_.has_value()
                         ? this->print_input_.value()
                         : [](const InputT& input) -> std::string {
    return internal::DefaultPrintValue(input);
  };
  auto print_result = this->print_result_.has_value()
                          ? this->print_result_.value()
                          : [](const ResultT& result) {
                              return internal::DefaultPrintValue(result);
                            };
  auto log_errors =
      this->log_errors_.has_value()
          ? this->log_errors_.value()
          : [print_input, print_result](int64_t index, const InputT& input,
                                        const ResultT& expected,
                                        const ResultT& actual) {
              internal::DefaultLogError<InputT, ResultT>(
                  index, input, expected, actual, print_input, print_result);
            };
  return Testbench<InputT, ResultT, ShardDataT>(
      /*start=*/0, this->num_samples_, this->num_threads_, this->max_failures_,
      index_to_input, this->compute_expected_, this->compute_actual_,
      compare_results, log_errors);
}

}  // namespace xls

#endif  // XLS_TOOLS_TESTBENCH_BUILDER_H_
