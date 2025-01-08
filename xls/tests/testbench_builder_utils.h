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

#ifndef XLS_TESTS_TESTBENCH_BUILDER_UTILS_H_
#define XLS_TESTS_TESTBENCH_BUILDER_UTILS_H_

// This file contains "helper" default implementations of the IndexToInput,
// CompareResults, and Print* routines.
// These are provided to reduce the amount of overhead needed to create a
// Testbench - for values supported here, the above functions needn't be
// specified in the builder.

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace xls {
namespace internal {

////////////////////////////////////////////////////
// IndexToInput support - creates random samples. //
////////////////////////////////////////////////////
template <typename ValueT>
void GenerateRandomValue(absl::BitGen& gen, int64_t index, ValueT* value) {
  // Fallback case - unknown ValueT. Do nothing.
}

template <>
inline void GenerateRandomValue<int32_t>(absl::BitGen& gen, int64_t index,
                                         int32_t* value) {
  *value = absl::bit_cast<int32_t>(absl::Uniform<uint32_t>(gen));
}

// We could use MORE TEMPLATE MAGIC to combine some of these cases, but I don't
// think it's worth the trouble.
template <>
inline void GenerateRandomValue<int64_t>(absl::BitGen& gen, int64_t index,
                                         int64_t* value) {
  *value = absl::bit_cast<int64_t>(absl::Uniform<uint64_t>(gen));
}

template <>
inline void GenerateRandomValue<float>(absl::BitGen& gen, int64_t index,
                                       float* value) {
  *value = absl::bit_cast<float>(absl::Uniform<uint32_t>(gen));
}

template <>
inline void GenerateRandomValue<double>(absl::BitGen& gen, int64_t index,
                                        double* value) {
  *value = absl::bit_cast<double>(absl::Uniform<uint64_t>(gen));
}

// Tuples require two levels - one base case (here) and the recursive case
// below.
template <typename TupleT, int kTupleIndex,
          std::enable_if_t<kTupleIndex == std::tuple_size<TupleT>{}>* = nullptr>
void GenerateRandomTuple(absl::BitGen& gen, int64_t index, TupleT* value) {
  // Nothing to do - we're at the end of the tuple!
}

// Tuple recursive case: Populate the kTupleIndex'th element and move on to the
// next.
template <typename TupleT, int kTupleIndex = 0,
          std::enable_if_t<kTupleIndex != std::tuple_size<TupleT>{}>* = nullptr>
void GenerateRandomTuple(absl::BitGen& gen, int64_t index, TupleT* value) {
  GenerateRandomValue<typename std::tuple_element<kTupleIndex, TupleT>::type>(
      gen, index, &std::get<kTupleIndex>(*value));
  GenerateRandomTuple<TupleT, kTupleIndex + 1>(gen, index, value);
}

// Entry point for general usage.
template <typename ValueT>
void DefaultIndexToInput(int64_t index, ValueT* value) {
  thread_local absl::BitGen gen;
  GenerateRandomValue<ValueT>(gen, index, value);
}

// std::tuple entry point.
template <typename... ElementsT>
void DefaultIndexToInput(int64_t index, std::tuple<ElementsT...>* value) {
  thread_local absl::BitGen gen;
  GenerateRandomTuple<std::tuple<ElementsT...>>(gen, index, value);
}

/////////////////////////////////////////////
// CompareResults support.                 //
/////////////////////////////////////////////
// The per-comparer evaluations were collected here in the hopes that they might
// be easier to read this way, but nothing prevents them from being
// re-associated if it's not an improvement.
template <int kComparer, typename ResultT>
struct ShouldEnableComparer {
  constexpr static bool value =
      (kComparer == 0 && (!std::is_arithmetic<ResultT>::value)) ||
      (kComparer == 1 && (std::is_integral<ResultT>::value)) ||
      (kComparer == 2 && (std::is_floating_point<ResultT>::value));
};

// Last-effort "we can't compare these" comparer.
template <typename ResultT, typename EnableT = void>
class Comparer;

template <typename ResultT>
class Comparer<ResultT, typename std::enable_if<
                            ShouldEnableComparer<0, ResultT>::value>::type> {
 public:
  static bool Compare(const ResultT& a, const ResultT& b) {
    LOG(ERROR) << absl::StrCat(
        "Returning false as the testbench doesn't know how to compare "
        "these ResultTs.");
    return false;
  }
};

// Integral-type comparer.
template <typename ResultT>
class Comparer<ResultT, typename std::enable_if<
                            ShouldEnableComparer<1, ResultT>::value>::type> {
 public:
  static bool Compare(const ResultT& a, const ResultT& b) { return a == b; }
};

// Floating-point comparer.
template <typename ResultT>
class Comparer<ResultT, typename std::enable_if<
                            ShouldEnableComparer<2, ResultT>::value>::type> {
 public:
  static bool Compare(const ResultT& a, const ResultT& b) {
    return (std::isnan(a) && std::isnan(b)) || (a == b);
  }
};

template <typename ResultT>
bool DefaultCompareResults(const ResultT& a, const ResultT& b) {
  return Comparer<ResultT>::Compare(a, b);
}

/////////////////////////////////////////////
// Printing support.                       //
/////////////////////////////////////////////
template <typename... ElementsT>
std::string DefaultPrintValue(const std::tuple<ElementsT...>& value);

template <typename ValueT>
std::string DefaultPrintValue(
    const ValueT& value,
    typename std::enable_if<std::is_arithmetic<ValueT>::value>::type* = 0) {
  return absl::StrCat(value);
}

inline std::string DefaultPrintValue(double value) {
  return absl::StrFormat("%.16e (%a)", value, value);
}

inline std::string DefaultPrintValue(float value) {
  return absl::StrFormat("%.9e (%a)", value, value);
}

template <typename ValueT>
std::string DefaultPrintValue(
    const ValueT& value,
    typename std::enable_if<!std::is_arithmetic<ValueT>::value>::type* = 0) {
  return absl::StrFormat("<unprintable value @ 0x%x",
                         reinterpret_cast<uint64_t>(&value));
}

// std::tuple entry point.
template <typename... ElementsT>
std::string DefaultPrintValue(const std::tuple<ElementsT...>& value) {
  return absl::StrCat("(", DefaultPrintValue(std::get<0>(value)), ", ",
                      DefaultPrintValue(std::get<1>(value)), ")");
}

// Common error-logging function.
// Not really a helper, like the above, but it's close enough to go here.
template <typename InputT, typename ResultT, typename PrintInputFnT,
          typename PrintResultFnT>
void DefaultLogError(int64_t index, InputT input, ResultT expected,
                     ResultT actual, PrintInputFnT print_input,
                     PrintResultFnT print_result) {
  std::string input_str = print_input(input);
  // Chop up the input and add indentation, if necessary.
  if (absl::StrContains(input_str, "\n")) {
    std::vector<std::string> pieces = absl::StrSplit(input_str, '\n');
    for (auto& piece : pieces) {
      piece = absl::StrCat("    ", piece);
    }
    input_str = absl::StrCat("\n", absl::StrJoin(pieces, "\n"));
  }
  std::string expected_str = print_result(expected);
  std::string actual_str = print_result(actual);
  absl::PrintF(
      "Mismatch at index %d:\n"
      "  Input   : %s\n"
      "  Expected: %s\n"
      "  Actual  : %s\n",
      index, input_str, expected_str, actual_str);
}

}  // namespace internal
}  // namespace xls

#endif  // XLS_TESTS_TESTBENCH_BUILDER_UTILS_H_
