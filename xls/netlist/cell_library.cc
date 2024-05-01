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

#include "xls/netlist/cell_library.h"

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/netlist/netlist.pb.h"

namespace xls {
namespace netlist {

absl::StatusOr<CellKind> CellKindFromProto(CellKindProto proto) {
  switch (proto) {
    case FLOP:
      return CellKind::kFlop;
    case INVERTER:
      return CellKind::kInverter;
    case BUFFER:
      return CellKind::kBuffer;
    case NAND:
      return CellKind::kNand;
    case NOR:
      return CellKind::kNor;
    case MULTIPLEXER:
      return CellKind::kMultiplexer;
    case XOR:
      return CellKind::kXor;
    case OTHER:
      return CellKind::kOther;
    // Note: since this is a proto enum there are sentinel values defined in
    // addition to the "real" above, which is why the enumeration of cases is
    // not exhaustive.
    case INVALID:
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid proto value for conversion to CellKind: %d", proto));
  }
}

absl::StatusOr<StateTableSignal> StateTableSignalFromProto(
    StateTableSignalProto proto) {
  switch (proto) {
    case STATE_TABLE_SIGNAL_INVALID:
      return StateTableSignal::kInvalid;
    case STATE_TABLE_SIGNAL_LOW:
      return StateTableSignal::kLow;
    case STATE_TABLE_SIGNAL_HIGH:
      return StateTableSignal::kHigh;
    case STATE_TABLE_SIGNAL_DONTCARE:
      return StateTableSignal::kDontCare;
    case STATE_TABLE_SIGNAL_HIGH_OR_LOW:
      return StateTableSignal::kHighOrLow;
    case STATE_TABLE_SIGNAL_LOW_OR_HIGH:
      return StateTableSignal::kLowOrHigh;
    case STATE_TABLE_SIGNAL_RISING:
      return StateTableSignal::kRising;
    case STATE_TABLE_SIGNAL_FALLING:
      return StateTableSignal::kFalling;
    case STATE_TABLE_SIGNAL_NOT_RISING:
      return StateTableSignal::kNotRising;
    case STATE_TABLE_SIGNAL_NOT_FALLING:
      return StateTableSignal::kNotFalling;
    case STATE_TABLE_SIGNAL_NOCHANGE:
      return StateTableSignal::kNoChange;
    case STATE_TABLE_SIGNAL_X:
      return StateTableSignal::kX;
    // Note: since this is a proto enum there are sentinel values defined in
    // addition to the "real" ones above, which is why the enumeration above is
    // not exhaustive.
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid proto value for conversion to StateTableSignal: %d", proto));
  }
}

absl::StatusOr<StateTableSignalProto> ProtoFromStateTableSignal(
    StateTableSignal signal) {
  switch (signal) {
    case StateTableSignal::kInvalid:
      return STATE_TABLE_SIGNAL_INVALID;
    case StateTableSignal::kLow:
      return STATE_TABLE_SIGNAL_LOW;
    case StateTableSignal::kHigh:
      return STATE_TABLE_SIGNAL_HIGH;
    case StateTableSignal::kDontCare:
      return STATE_TABLE_SIGNAL_DONTCARE;
    case StateTableSignal::kHighOrLow:
      return STATE_TABLE_SIGNAL_HIGH_OR_LOW;
    case StateTableSignal::kLowOrHigh:
      return STATE_TABLE_SIGNAL_LOW_OR_HIGH;
    case StateTableSignal::kRising:
      return STATE_TABLE_SIGNAL_RISING;
    case StateTableSignal::kFalling:
      return STATE_TABLE_SIGNAL_FALLING;
    case StateTableSignal::kNotRising:
      return STATE_TABLE_SIGNAL_NOT_RISING;
    case StateTableSignal::kNotFalling:
      return STATE_TABLE_SIGNAL_NOT_FALLING;
    case StateTableSignal::kNoChange:
      return STATE_TABLE_SIGNAL_NOCHANGE;
    case StateTableSignal::kX:
      return STATE_TABLE_SIGNAL_X;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid proto value for conversion to StateTableSignal: %d",
          static_cast<int>(signal)));
  }
}

std::string CellKindToString(CellKind kind) {
  switch (kind) {
    case CellKind::kFlop:
      return "flop";
    case CellKind::kInverter:
      return "inverter";
    case CellKind::kBuffer:
      return "buffer";
    case CellKind::kNand:
      return "nand";
    case CellKind::kNor:
      return "nor";
    case CellKind::kXor:
      return "xor";
    case CellKind::kMultiplexer:
      return "multiplexer";
    case CellKind::kOther:
      return "other";
  }
  return absl::StrFormat("<invalid CellKind(%d)>", static_cast<int64_t>(kind));
}

}  // namespace netlist
}  // namespace xls
