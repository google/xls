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

#ifndef XLS_CONTRIB_ICE40_WRAP_IO_H_
#define XLS_CONTRIB_ICE40_WRAP_IO_H_

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast/vast.h"
#include "xls/contrib/ice40/io_strategy.h"

namespace xls {
namespace verilog {

// Control codes of the I/O state machine. These codes are not interpreted as
// data, but rather initiate actions within the state machine. Passing data
// bytes equal to these values requires an escape sequnce.
enum IOControlCode : uint8_t {
  // Resets the I/O state machine and the device function. This control code
  // cannot be escaped (a reset is initiated even if the previous character
  // is the escape control code).
  kReset = 0xfe,

  // Escapes the next byte received. The interpretation of the escaped byte is
  // below.
  kEscape = 0xff,
};

// Escape codes of the I/O state machine. These bytes are sent immediately
// following the escape control code. An unrecognized escaped byte will be
// interpretted as a data byte of that value.
enum IOEscapeCode : uint8_t {
  // Interpreted as a data byte equal to the "reset" control code value
  // (IOControlCode::kReset).
  kResetByte = 0x00,

  // Interpreted as a data byte equal to the "escape" control code value
  // (IOControlCode::kEscape).
  kEscapeByte = 0xff,
};

// Decorates a Verilog module, that represents an HLS-codegen'd function entry
// point, with an I/O state machine.
//
// The I/O state machine receives data for an invocation one byte at a time,
// building up a buffer of inputs as a big shift register.
//
// Once all the data has been received, the HLS-generated function is "invoked"
// (we put invoked in quotes because it may be a continuous function that
// doesn't require any particular triggering), and after some latency L we know
// the output data is ready to transmit back to the host as a response.
//
// Args:
//  module_name: Name of the module being instantiated as the "device function"
//    that we're going to invoke. This should already be defined in the Verilog
//    file f (we refer to it as a free variable).
//  instance_name: The resulting WrapIO module instantiates the "device
//    function" -- instance_name is the name that we should give to the "device
//    function" instance that is created.
//  latency: Latency for the "device function" module to produce a result, in
//    cycles, once input has been presented to it.
//
// TODO(leary): 2019-03-25 We'll want to change the I/O mechanism into a
// pluggable strategy, right now this assumes ICE40 UART, but just as easily we
// should be able to plug in something like PCIe TLP handling.
absl::StatusOr<Module*> WrapIO(std::string_view module_name,
                               std::string_view instance_name,
                               const ModuleSignature& signature,
                               IOStrategy* io_strategy, VerilogFile* f);

// Creates and returns a module which controls the input to the I/O state
// machine. This module has a byte-wide input with ready/valid flow control and
// an arbitrary width output with ready/valid flow control. Input is accepted
// byte-by-byte and shifted into the (potentially larger) arbitrary width output
// where the first byte is the MSB.
//
// This module is intended to be used within WrapIO and is exposed in the header
// for testing purposes.
// TODO(meheff): Hook up this module into WrapIO.
absl::StatusOr<Module*> InputControllerModule(const ModuleSignature& signature,
                                              VerilogFile* f);

// Creates and returns the module which controls the reset of the I/O state
// machine via the reset control code (IOControlCode::kReset). This is
// instantiated within the InputControlModule and is exposed in the header for
// testing purposes.
absl::StatusOr<Module*> InputResetModule(VerilogFile* f);

// Creates and returns a shift register used by the InputControlerModuler. Takes
// a byte at a time, and output is an arbitrary width given by 'bit_count'.
absl::StatusOr<Module*> InputShiftRegisterModule(int64_t bit_count,
                                                 VerilogFile* f);

// Creates and returns a module which controls the output side of the I/O state
// machine. This module has an arbitrary width input to match the output of the
// device function. The output of the output controller is byte-wide. Both have
// ready/valid flow control.
absl::StatusOr<Module*> OutputControllerModule(const ModuleSignature& signature,
                                               VerilogFile* f);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CONTRIB_ICE40_WRAP_IO_H_
