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

#ifndef XLS_CODEGEN_BLOCK_CONVERSION_TEST_FIXTURE_H_
#define XLS_CODEGEN_BLOCK_CONVERSION_TEST_FIXTURE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_test_base.h"
#include "xls/tools/codegen.h"

namespace xls::verilog {

// Convenience functions for sensitizing and analyzing procs used to
// test blocks.
class BlockConversionTestFixture : public IrTestBase {
 protected:
  // A pair of cycle and value for returning traces.
  struct CycleAndValue {
    int64_t cycle;
    uint64_t value;
  };

  enum class SignalType : uint8_t { kInput, kOutput, kExpectedOutput };

  // Specification for a column when printing out a signal trace.
  struct SignalSpec {
    std::string signal_name;
    SignalType signal_type;
    bool active_low_reset = false;
  };

  // For cycles in range [first_cycle, last_cycle] inclusive,
  // add the IO signals as described in signals to io.
  absl::Status SetSignalsOverCycles(
      int64_t first_cycle, int64_t last_cycle,
      const absl::flat_hash_map<std::string, uint64_t>& signals,
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& io) const {
    CHECK_GE(first_cycle, 0);
    CHECK_GE(last_cycle, 0);
    CHECK_LE(first_cycle, last_cycle);

    if (io.size() <= last_cycle) {
      io.resize(last_cycle + 1);
    }

    for (const auto& [name, value] : signals) {
      for (int64_t i = first_cycle; i <= last_cycle; ++i) {
        io.at(i)[name] = value;
      }
    }

    return absl::OkStatus();
  }

  // For cycles in range [first_cycle, last_cycle] inclusive,
  // set given input signal to a incrementing value starting with start_val.
  //
  // One after the last signal value used is returned.
  absl::StatusOr<uint64_t> SetIncrementingSignalOverCycles(
      int64_t first_cycle, int64_t last_cycle, std::string_view signal_name,
      uint64_t signal_val,
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& io) const {
    CHECK_GE(first_cycle, 0);
    CHECK_GE(last_cycle, 0);
    CHECK_LE(first_cycle, last_cycle);

    if (io.size() <= last_cycle) {
      io.resize(last_cycle + 1);
    }

    for (int64_t i = first_cycle; i <= last_cycle; ++i) {
      io.at(i)[signal_name] = signal_val;
      ++signal_val;
    }

    return signal_val;
  }

  // For cycles in range [first_cycle, last_cycle] inclusive,  set given
  // input signal to uniformly random input in range [min_value, max_value].
  absl::Status SetRandomSignalOverCycles(
      int64_t first_cycle, int64_t last_cycle, std::string_view signal_name,
      uint64_t min_value, uint64_t max_value, absl::BitGenRef rng,
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& io) const {
    CHECK_GE(first_cycle, 0);
    CHECK_GE(last_cycle, 0);
    CHECK_LE(first_cycle, last_cycle);

    if (io.size() <= last_cycle) {
      io.resize(last_cycle + 1);
    }

    for (int64_t i = first_cycle; i <= last_cycle; ++i) {
      io.at(i)[signal_name] =
          absl::Uniform(absl::IntervalClosed, rng, min_value, max_value);
    }

    return absl::OkStatus();
  }

  // From either an input or output channel, retrieve the sequence of
  // sent/received data.
  //
  // Data is deemed sent/received if not under reset and valid and ready are 1.
  absl::StatusOr<std::vector<CycleAndValue>> GetChannelSequenceFromIO(
      const SignalSpec& data_signal, const SignalSpec& valid_signal,
      const SignalSpec& ready_signal, const SignalSpec& reset_signal,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> outputs)
      const {
    CHECK_EQ(inputs.size(), outputs.size());

    std::vector<CycleAndValue> sequence;

    for (int64_t i = 0; i < inputs.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          bool rst,
          FindWithinIOHashMaps(reset_signal, inputs.at(i), outputs.at(i)));

      XLS_ASSIGN_OR_RETURN(
          uint64_t data,
          FindWithinIOHashMaps(data_signal, inputs.at(i), outputs.at(i)));
      XLS_ASSIGN_OR_RETURN(
          bool data_vld,
          FindWithinIOHashMaps(valid_signal, inputs.at(i), outputs.at(i)));
      XLS_ASSIGN_OR_RETURN(
          bool data_rdy,
          FindWithinIOHashMaps(ready_signal, inputs.at(i), outputs.at(i)));

      bool rst_active = reset_signal.active_low_reset ? !rst : rst;

      if (data_vld && data_rdy && !rst_active) {
        sequence.push_back({i, data});
      }
    }

    return sequence;
  }

  // Log at verbose level 1, a table of signals and their values.
  absl::Status VLogTestPipelinedIO(
      absl::Span<const SignalSpec> table_spec, int64_t column_width,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> inputs,
      absl::Span<const absl::flat_hash_map<std::string, uint64_t>> outputs,
      std::optional<
          absl::Span<const absl::flat_hash_map<std::string, uint64_t>>>
          expected_outputs = std::nullopt) const {
    CHECK_EQ(inputs.size(), outputs.size());
    if (expected_outputs.has_value()) {
      CHECK_EQ(inputs.size(), expected_outputs->size());
    }

    std::string header;
    for (const SignalSpec& col : table_spec) {
      if (col.signal_type == SignalType::kExpectedOutput) {
        std::string signal_name_with_suffix =
            absl::StrCat(col.signal_name, "_e");
        absl::StrAppend(&header, absl::StrFormat(" %*s", column_width,
                                                 signal_name_with_suffix));
      } else {
        absl::StrAppend(&header,
                        absl::StrFormat(" %*s", column_width, col.signal_name));
      }
    }

    VLOG(1) << header;

    for (int64_t i = 0; i < inputs.size(); ++i) {
      std::string row;

      for (const SignalSpec& col : table_spec) {
        std::string_view signal_name = col.signal_name;
        SignalType signal_type = col.signal_type;

        CHECK(signal_type == SignalType::kInput ||
              signal_type == SignalType::kOutput ||
              signal_type == SignalType::kExpectedOutput);

        uint64_t signal_value = 0;
        if (signal_type == SignalType::kInput) {
          signal_value = inputs.at(i).at(signal_name);
        } else if (signal_type == SignalType::kOutput) {
          signal_value = outputs.at(i).at(signal_name);
        } else {
          CHECK(expected_outputs.has_value());
          signal_value = expected_outputs->at(i).at(signal_name);
        }

        absl::StrAppend(&row,
                        absl::StrFormat(" %*d", column_width, signal_value));
      }

      VLOG(1) << row;
    }

    return absl::OkStatus();
  }

  // Find signal value either the input or output hash maps depending on the
  // spec.
  absl::StatusOr<uint64_t> FindWithinIOHashMaps(
      const SignalSpec& signal,
      const absl::flat_hash_map<std::string, uint64_t>& inputs,
      const absl::flat_hash_map<std::string, uint64_t>& outputs) const {
    SignalType signal_type = signal.signal_type;
    std::string_view signal_name = signal.signal_name;

    if (signal_type == SignalType::kInput) {
      if (!inputs.contains(signal_name)) {
        return absl::InternalError(
            absl::StrFormat("%s not found in input", signal_name));
      }
      return inputs.at(signal_name);
    }
    if (signal_type == SignalType::kOutput ||
        signal_type == SignalType::kExpectedOutput) {
      if (!outputs.contains(signal_name)) {
        return absl::InternalError(
            absl::StrFormat("%s not found in output", signal_name));
      }
      return outputs.at(signal_name);
    }

    return absl::InternalError(absl::StrFormat(
        "Unsupported SignalType %d for %s", signal_type, signal_name));
  }
};

}  // namespace xls::verilog
#endif  // XLS_CODEGEN_BLOCK_CONVERSION_TEST_FIXTURE_H_
