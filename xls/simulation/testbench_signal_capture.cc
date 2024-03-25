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

#include "xls/simulation/testbench_signal_capture.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "xls/common/source_location.h"
#include "xls/ir/bits.h"
#include "xls/simulation/testbench_stream.h"

namespace xls {
namespace verilog {

SignalCapture SignalCaptureManager::Capture(std::string_view signal_name,
                                            Bits* bits) {
  CHECK(metadata().HasPortNamed(signal_name));
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(
      SignalCapture{.signal_name = std::string{signal_name},
                    .signal_width = metadata().GetPortWidth(signal_name),
                    .action = bits,
                    .instance_id = instance});
  return signal_captures_.back();
}

SignalCapture SignalCaptureManager::CaptureMultiple(
    std::string_view signal_name, std::vector<Bits>* values) {
  CHECK(metadata().HasPortNamed(signal_name));
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(
      SignalCapture{.signal_name = std::string{signal_name},
                    .signal_width = metadata().GetPortWidth(signal_name),
                    .action = values,
                    .instance_id = instance});
  return signal_captures_.back();
}

SignalCapture SignalCaptureManager::CaptureAndWriteToStream(
    std::string_view signal_name, const TestbenchStream* stream) {
  CHECK(metadata().HasPortNamed(signal_name));
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(
      SignalCapture{.signal_name = std::string{signal_name},
                    .signal_width = stream->width,
                    .action = stream,
                    .instance_id = instance});
  return signal_captures_.back();
}

SignalCapture SignalCaptureManager::ExpectEq(std::string_view signal_name,
                                             const Bits& bits,
                                             xabsl::SourceLocation loc) {
  CHECK(metadata().HasPortNamed(signal_name));
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(SignalCapture{
      .signal_name = std::string{signal_name},
      .signal_width = metadata().GetPortWidth(signal_name),
      .action = TestbenchExpectation{.expected = bits, .loc = loc},
      .instance_id = instance});
  return signal_captures_.back();
}

SignalCapture SignalCaptureManager::ExpectEq(std::string_view signal_name,
                                             uint64_t value,
                                             xabsl::SourceLocation loc) {
  return ExpectEq(signal_name,
                  UBits(value, metadata().GetPortWidth(signal_name)), loc);
}

SignalCapture SignalCaptureManager::ExpectX(std::string_view signal_name,
                                            xabsl::SourceLocation loc) {
  CHECK(metadata().HasPortNamed(signal_name));
  int64_t instance = signal_captures_.size();
  signal_captures_.push_back(SignalCapture{
      .signal_name = std::string{signal_name},
      .signal_width = metadata().GetPortWidth(signal_name),
      .action = TestbenchExpectation{.expected = IsX(), .loc = loc},
      .instance_id = instance});
  return signal_captures_.back();
}

EndOfCycleEvent& EndOfCycleEvent::Capture(std::string_view signal_name,
                                          Bits* value) {
  signal_captures_.push_back(capture_manager_->Capture(signal_name, value));
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::CaptureMultiple(std::string_view signal_name,
                                                  std::vector<Bits>* values) {
  signal_captures_.push_back(
      capture_manager_->CaptureMultiple(signal_name, values));
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::CaptureAndWriteToStream(
    std::string_view signal_name, const TestbenchStream* stream) {
  signal_captures_.push_back(
      capture_manager_->CaptureAndWriteToStream(signal_name, stream));
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectEq(std::string_view signal_name,
                                           const Bits& expected,
                                           xabsl::SourceLocation loc) {
  signal_captures_.push_back(
      capture_manager_->ExpectEq(signal_name, expected, loc));
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectEq(std::string_view signal_name,
                                           uint64_t expected,
                                           xabsl::SourceLocation loc) {
  signal_captures_.push_back(
      capture_manager_->ExpectEq(signal_name, expected, loc));
  return *this;
}

EndOfCycleEvent& EndOfCycleEvent::ExpectX(std::string_view signal_name,
                                          xabsl::SourceLocation loc) {
  signal_captures_.push_back(capture_manager_->ExpectX(signal_name, loc));
  return *this;
}

}  // namespace verilog
}  // namespace xls
