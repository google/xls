// Copyright 2020 The XLS Authors
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

#ifndef XLS_SIMULATION_TESTBENCH_SIGNAL_CAPTURE_H_
#define XLS_SIMULATION_TESTBENCH_SIGNAL_CAPTURE_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/log/die_if_null.h"
#include "absl/types/span.h"
#include "xls/common/source_location.h"
#include "xls/ir/bits.h"
#include "xls/simulation/testbench_metadata.h"
#include "xls/simulation/testbench_stream.h"

namespace xls {
namespace verilog {

// Sentinel type for indicating an "X" value, in lieu of some real bits value.
struct IsX {};

// Sentinel type for indicating an "not X" value, in lieu of some real bits
// value.
struct IsNotX {};

using BitsOrX = std::variant<Bits, IsX>;

struct TestbenchExpectation {
  BitsOrX expected;
  xabsl::SourceLocation loc;
};

// Single (Bits*) or multiple (std::vector<Bits>*) values can be captured as an
// action.
using SignalCaptureAction =
    std::variant<TestbenchExpectation, Bits*, std::vector<Bits>*,
                 const TestbenchStream*>;

// Represents a single instance of a signal capture. Each corresponds to a
// particular $display statement in the testbench.
struct SignalCapture {
  std::string signal_name;
  int64_t signal_width;

  SignalCaptureAction action;

  // A unique identifier which associates this capture instance with a
  // particular line of simulation output. This integer is emitted along side
  // the captured value in the $display-ed string during simulation and is used
  // to associate the value back to a particular `Capture` (or `ExpectEq`, etc)
  // instance.
  int64_t instance_id;
};

// Data structure which allocates capture instances so that the instance ids are
// unique across all threads in the testbench.
class SignalCaptureManager {
 public:
  explicit SignalCaptureManager(const TestbenchMetadata* metadata)
      : metadata_(metadata) {}

  // Return a capture instance associated with a Capture/ExpectEq/ExpectX
  // action.
  SignalCapture Capture(std::string_view signal_name, Bits* bits);
  SignalCapture CaptureMultiple(std::string_view signal_name,
                                std::vector<Bits>* values);
  SignalCapture ExpectEq(std::string_view signal_name, const Bits& bits,
                         xabsl::SourceLocation loc);
  SignalCapture ExpectEq(std::string_view signal_name, uint64_t value,
                         xabsl::SourceLocation loc);
  SignalCapture ExpectX(std::string_view signal_name,
                        xabsl::SourceLocation loc);
  SignalCapture CaptureAndWriteToStream(std::string_view signal_name,
                                        const TestbenchStream* stream);

  absl::Span<const SignalCapture> signal_captures() const {
    return signal_captures_;
  }

  const TestbenchMetadata& metadata() const { return *metadata_; }

 private:
  const TestbenchMetadata* metadata_;
  std::vector<SignalCapture> signal_captures_;
};

// Data-structure representing the end of a cycle (one time unit before the
// rising edge of the clock). In the ModuleTestbench infrastructure signals are
// only sampled at the end of a cycle. The ModuleTestbenchThread API returns
// this object to enable capturing signals and `expect`ing their values. For
// example, in the following code `AtEndOfCycleWhen` returns a EndOfCycleEvent
// corresponding to the end of the cycle when `foo_valid` is first asserted, and
// at this point the value of `bar` is captured.
//
//    Bits bar;
//    testbench_thread.AtEndOfCycleWhen("foo_valid").Capture("bar", &bar);
class EndOfCycleEvent {
 public:
  explicit EndOfCycleEvent(const TestbenchMetadata* metadata,
                           SignalCaptureManager* capture_manager)
      : metadata_(ABSL_DIE_IF_NULL(metadata)),
        capture_manager_(capture_manager) {}

  // Captures the value of the signal. The given pointer value is written with
  // the signal value when Run is called.
  EndOfCycleEvent& Capture(std::string_view signal_name, Bits* value);

  // Captures multiple instances of the value of the given signal. This can be
  // used in blocks from `RepeatForever` and `Repeat` calls.
  EndOfCycleEvent& CaptureMultiple(std::string_view signal_name,
                                   std::vector<Bits>* values);

  // Captures the given signal and writes it to the given stream.
  EndOfCycleEvent& CaptureAndWriteToStream(std::string_view signal_name,
                                           const TestbenchStream* stream);

  // Expects the given signal is the given value (or X). An error is returned
  // during Run if this expectation is not met.
  //
  // "loc" indicates the source position in the test where the expectation was
  // created, and is displayed on expectation failure.
  EndOfCycleEvent& ExpectEq(
      std::string_view signal_name, const Bits& expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  EndOfCycleEvent& ExpectEq(
      std::string_view signal_name, uint64_t expected,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());
  EndOfCycleEvent& ExpectX(
      std::string_view signal_name,
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  absl::Span<const SignalCapture> signal_captures() const {
    return signal_captures_;
  }

 private:
  const TestbenchMetadata* metadata_;
  SignalCaptureManager* capture_manager_;

  // Set of instances of signal captures. Each corresponds to a particular
  // $display statement in the testbench.
  std::vector<SignalCapture> signal_captures_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_TESTBENCH_SIGNAL_CAPTURE_H_
