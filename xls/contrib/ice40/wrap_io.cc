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

#include "xls/contrib/ice40/wrap_io.h"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/finite_state_machine.h"
#include "xls/codegen/vast.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {

namespace {

// Abstraction gathering signals for a ready/valid interface.
struct ReadyValid {
  LogicRef* ready;
  LogicRef* valid;
  LogicRef* data;
};

// Instantiates the given device function module which has fixed latency
// interface.
absl::Status InstantiateFixedLatencyDeviceFunction(
    const ModuleSignature& signature, LogicRef* clk, LogicRef* rst_n,
    ReadyValid input, ReadyValid output, int64_t latency, Module* m) {
  XLS_RET_CHECK_EQ(signature.data_inputs().size(), 1);
  XLS_RET_CHECK_EQ(signature.data_outputs().size(), 1);
  const PortProto& input_port = signature.data_inputs().front();
  const PortProto& output_port = signature.data_outputs().front();

  // Construct an FSM which matches the ready/valid interfaces of the input and
  // output controllers with the fixed latency of the device function.
  // TODO(meheff): Expose use_system_verilog as an option in the WrapIO API
  // rather than hard-coding it as false.
  VerilogFile* f = m->file();
  FsmBuilder fsm("fixed_latency_fsm", m, clk,
                 /*use_system_verilog=*/false,
                 Reset{rst_n, /*asynchronous=*/false, /*active_low=*/true});
  auto idle_state = fsm.AddState("Idle");
  auto computing_state = fsm.AddState("Computing");
  auto done_state = fsm.AddState("Done");

  auto input_ready_output = fsm.AddOutput1("input_ready_reg", 0);
  auto output_valid_output = fsm.AddOutput1("output_valid_reg", 0);

  XLS_RET_CHECK_GE(latency, 1);
  auto cycle_counter = fsm.AddDownCounter(
      "cycle_counter",
      std::max(int64_t{1}, Bits::MinBitCountUnsigned(latency - 1)));

  // This relies on the output ready staying asserted for the duration of the
  // computation.
  idle_state
      ->OnCondition(f->LogicalAnd(input.valid, output.ready, SourceInfo()))
      .NextState(computing_state)
      .SetCounter(cycle_counter, latency - 1);

  computing_state->OnCounterIsZero(cycle_counter).NextState(done_state);

  done_state->NextState(idle_state)
      .SetOutput(input_ready_output, 1)
      .SetOutput(output_valid_output, 1);

  XLS_RETURN_IF_ERROR(fsm.Build());
  m->Add<ContinuousAssignment>(SourceInfo(), input.ready,
                               input_ready_output->logic_ref);
  m->Add<ContinuousAssignment>(SourceInfo(), output.valid,
                               output_valid_output->logic_ref);

  std::vector<Connection> connections;
  if (signature.proto().has_clock_name()) {
    connections.push_back({signature.proto().clock_name(), clk});
  }

  if (signature.proto().has_reset()) {
    XLS_RET_CHECK(signature.proto().reset().active_low());
    connections.push_back({signature.proto().reset().name(), rst_n});
  }

  connections.push_back({input_port.name(), input.data});
  connections.push_back({output_port.name(), output.data});

  m->Add<Instantiation>(SourceInfo(), signature.module_name(),
                        "device_function",
                        /*parameters=*/std::vector<Connection>{},
                        /*connections=*/connections);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Module*> WrapIO(std::string_view module_name,
                               std::string_view instance_name,
                               const ModuleSignature& signature,
                               IOStrategy* io_strategy, VerilogFile* f) {
  XLS_ASSIGN_OR_RETURN(Module * input_controller_m,
                       InputControllerModule(signature, f));
  XLS_ASSIGN_OR_RETURN(Module * output_controller_m,
                       OutputControllerModule(signature, f));

  // We're creating a module that *wraps* the compute module with I/O
  // components.
  Module* io_wrapper = f->AddModule("io_wrapper", SourceInfo());

  LogicRef* clk =
      io_wrapper->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      io_wrapper->AddWire("rst_n", f->ScalarType(SourceInfo()), SourceInfo());
  Reset reset{rst_n, /*asynchronous=*/false, /*active_low=*/true};
  XLS_RETURN_IF_ERROR(
      io_strategy->AddTopLevelDependencies(clk, reset, io_wrapper));

  IOStrategy::Input input_signals = {
      .rx_byte = io_wrapper->AddWire(
          "rx_byte", f->BitVectorType(8, SourceInfo()), SourceInfo()),
      .rx_byte_valid = io_wrapper->AddWire(
          "rx_byte_valid", f->ScalarType(SourceInfo()), SourceInfo()),
      .rx_byte_done = io_wrapper->AddWire(
          "rx_byte_done", f->ScalarType(SourceInfo()), SourceInfo()),
  };
  IOStrategy::Output output_signals = {
      .tx_byte = io_wrapper->AddWire(
          "tx_byte", f->BitVectorType(8, SourceInfo()), SourceInfo()),
      .tx_byte_valid = io_wrapper->AddWire(
          "tx_byte_valid", f->ScalarType(SourceInfo()), SourceInfo()),
      .tx_byte_ready = io_wrapper->AddWire(
          "tx_byte_ready", f->ScalarType(SourceInfo()), SourceInfo()),
  };
  XLS_RETURN_IF_ERROR(io_strategy->InstantiateIOBlocks(
      input_signals, output_signals, io_wrapper));

  LogicRef* flat_input = io_wrapper->AddWire(
      "flat_input",
      f->BitVectorType(signature.TotalDataInputBits(), SourceInfo()),
      SourceInfo());
  LogicRef* flat_input_valid = io_wrapper->AddWire(
      "flat_input_valid", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* flat_input_ready = io_wrapper->AddWire(
      "flat_input_ready", f->ScalarType(SourceInfo()), SourceInfo());
  {
    std::vector<Connection> connections;
    connections.push_back(Connection{"clk", clk});
    connections.push_back(Connection{"byte_in", input_signals.rx_byte});
    connections.push_back(
        Connection{"byte_in_valid", input_signals.rx_byte_valid});
    connections.push_back(
        Connection{"byte_in_ready", input_signals.rx_byte_done});
    connections.push_back(Connection{"data_out", flat_input});
    connections.push_back(Connection{"data_out_valid", flat_input_valid});
    connections.push_back(Connection{"data_out_ready", flat_input_ready});
    connections.push_back(
        Connection{"rst_n_in", f->Literal(1, 1, SourceInfo())});
    connections.push_back(Connection{"rst_n_out", rst_n});
    io_wrapper->Add<Instantiation>(
        SourceInfo(), input_controller_m->name(), "input_controller",
        /*parameters=*/absl::Span<const Connection>(), connections);
  }
  LogicRef* flat_output = io_wrapper->AddWire(
      "flat_output",
      f->BitVectorType(signature.TotalDataOutputBits(), SourceInfo()),
      SourceInfo());
  LogicRef* flat_output_valid = io_wrapper->AddWire(
      "flat_output_valid", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* flat_output_ready = io_wrapper->AddWire(
      "flat_output_ready", f->ScalarType(SourceInfo()), SourceInfo());

  {
    std::vector<Connection> connections;
    connections.push_back(Connection{"clk", clk});
    connections.push_back(Connection{"rst_n", rst_n});
    connections.push_back(Connection{"data_in", flat_output});
    connections.push_back(Connection{"data_in_valid", flat_output_valid});
    connections.push_back(Connection{"data_in_ready", flat_output_ready});
    connections.push_back(Connection{"byte_out", output_signals.tx_byte});
    connections.push_back(
        Connection{"byte_out_valid", output_signals.tx_byte_valid});
    connections.push_back(
        Connection{"byte_out_ready", output_signals.tx_byte_ready});
    io_wrapper->Add<Instantiation>(
        SourceInfo(), output_controller_m->name(), "output_controller",
        /*parameters=*/absl::Span<const Connection>(), connections);
  }

  ReadyValid input{flat_input_ready, flat_input_valid, flat_input};
  ReadyValid output{flat_output_ready, flat_output_valid, flat_output};

  if (signature.proto().has_pipeline()) {
    XLS_RETURN_IF_ERROR(InstantiateFixedLatencyDeviceFunction(
        signature, clk, rst_n, input, output,
        signature.proto().pipeline().latency(), io_wrapper));
  } else if (signature.proto().has_fixed_latency()) {
    XLS_RETURN_IF_ERROR(InstantiateFixedLatencyDeviceFunction(
        signature, clk, rst_n, input, output,
        signature.proto().fixed_latency().latency(), io_wrapper));
  } else {
    return absl::UnimplementedError("Unsupported interface");
  }

  return io_wrapper;
}

// Returns a hex-formatted byte-sized VAST literal of the given value.
static Literal* Hex8Literal(uint8_t value, VerilogFile* f) {
  return f->Literal(value, 8, SourceInfo(), FormatPreference::kHex);
}

absl::StatusOr<Module*> InputResetModule(VerilogFile* f) {
  Module* m = f->AddModule("input_resetter", SourceInfo());
  auto clk = m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
  auto byte_in =
      m->AddInput("byte_in", f->BitVectorType(8, SourceInfo()), SourceInfo());
  auto byte_in_ready =
      m->AddOutput("byte_in_ready", f->ScalarType(SourceInfo()), SourceInfo());
  auto byte_in_valid =
      m->AddInput("byte_in_valid", f->ScalarType(SourceInfo()), SourceInfo());
  auto rst_n_in =
      m->AddInput("rst_n_in", f->ScalarType(SourceInfo()), SourceInfo());
  auto rst_n_out =
      m->AddOutput("rst_n_out", f->ScalarType(SourceInfo()), SourceInfo());

  LocalParamItemRef* reset_control_code =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("ResetControlCode", Hex8Literal(IOControlCode::kReset, f),
                    SourceInfo());

  // TODO(meheff): Expose use_system_verilog as an option in the WrapIO API
  // rather than hard-coding it as false.
  FsmBuilder fsm("reset_fsm", m, clk, /*use_system_verilog=*/false,
                 Reset{rst_n_in, /*asynchronous=*/false, /*active_low=*/true});
  auto idle_state = fsm.AddState("Idle");
  auto reset_state = fsm.AddState("Reset");

  auto rst_n_output = fsm.AddOutput1("rst_n_reg", 1);
  auto byte_in_ready_output = fsm.AddOutput1("byte_in_ready_reg", 0);

  // If byte_in is the reset control code and byte_in_valid is asserted then
  // assert the reset signal.
  idle_state
      ->OnCondition(f->LogicalAnd(
          byte_in_valid, f->Equals(byte_in, reset_control_code, SourceInfo()),
          SourceInfo()))
      .NextState(reset_state);

  // In the reset state, assert byte_in_ready to clear the reset control code.
  reset_state->SetOutput(byte_in_ready_output, 1)
      .SetOutput(rst_n_output, 0)
      .NextState(idle_state);
  XLS_RETURN_IF_ERROR(fsm.Build());

  m->Add<ContinuousAssignment>(SourceInfo(), byte_in_ready,
                               byte_in_ready_output->logic_ref);
  m->Add<ContinuousAssignment>(
      SourceInfo(), rst_n_out,
      f->LogicalAnd(rst_n_in, rst_n_output->logic_ref, SourceInfo()));

  return m;
}

absl::StatusOr<Module*> InputShiftRegisterModule(int64_t bit_count,
                                                 VerilogFile* f) {
  Module* m = f->AddModule("input_shifter", SourceInfo());
  LogicRef* clk = m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* clear =
      m->AddInput("clear", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* byte_in =
      m->AddInput("byte_in", f->BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* write_en =
      m->AddInput("write_en", f->ScalarType(SourceInfo()), SourceInfo());

  LogicRef* data_out = m->AddOutput(
      "data_out", f->BitVectorType(bit_count, SourceInfo()), SourceInfo());
  LogicRef* done =
      m->AddOutput("done", f->ScalarType(SourceInfo()), SourceInfo());

  const int64_t n_bytes = CeilOfRatio(bit_count, int64_t{8});
  LocalParamItemRef* n_bytes_ref =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("TotalInputBytes", f->PlainLiteral(n_bytes, SourceInfo()),
                    SourceInfo());

  LogicRef* data_reg = m->AddReg(
      "data", f->BitVectorType(bit_count, SourceInfo()), SourceInfo());
  LogicRef* data_reg_next = m->AddReg(
      "data_next", f->BitVectorType(bit_count, SourceInfo()), SourceInfo());

  // A counter which keeps track of the number of bytes shifted in. When the
  // counter reaches zero, the register is full and 'done' is asserted.
  XLS_RET_CHECK_GT(n_bytes, 0);
  LogicRef* byte_countdown = m->AddReg(
      "byte_countdown",
      f->BitVectorType(Bits::MinBitCountUnsigned(n_bytes), SourceInfo()),
      SourceInfo());
  LogicRef* byte_countdown_next = m->AddReg(
      "byte_countdown_next",
      f->BitVectorType(Bits::MinBitCountUnsigned(n_bytes), SourceInfo()),
      SourceInfo());

  // Logic for the counter and shift register:
  //
  //   if (clear) {
  //     byte_countdown_next = ${n_bytes};
  //   } else if (write_en) {
  //     data_reg_next = (data_reg << 8) | byte_in;
  //     byte_countdown_next = byte_countdown - 1;
  //   } else {
  //     data_reg_next = data_reg;
  //     byte_countdown_next = byte_countdown;
  //   }
  auto ac = m->Add<Always>(SourceInfo(), std::vector<SensitivityListElement>(
                                             {ImplicitEventExpression()}));
  auto cond = ac->statements()->Add<Conditional>(SourceInfo(), clear);
  cond->consequent()->Add<BlockingAssignment>(SourceInfo(), byte_countdown_next,
                                              n_bytes_ref);
  auto else_write_en = cond->AddAlternate(write_en);
  else_write_en->Add<BlockingAssignment>(
      SourceInfo(), data_reg_next,
      f->BitwiseOr(
          f->Shll(data_reg, f->PlainLiteral(8, SourceInfo()), SourceInfo()),
          byte_in, SourceInfo()));
  else_write_en->Add<BlockingAssignment>(
      SourceInfo(), byte_countdown_next,
      f->Sub(byte_countdown, f->PlainLiteral(1, SourceInfo()), SourceInfo()));
  auto els = cond->AddAlternate();
  els->Add<BlockingAssignment>(SourceInfo(), byte_countdown_next,
                               byte_countdown);
  els->Add<BlockingAssignment>(SourceInfo(), data_reg_next, data_reg);

  auto af = m->Add<AlwaysFlop>(SourceInfo(), clk);
  af->AddRegister(data_reg, data_reg_next, SourceInfo());
  af->AddRegister(byte_countdown, byte_countdown_next, SourceInfo());

  m->Add<ContinuousAssignment>(
      SourceInfo(), done,
      f->Equals(byte_countdown, f->PlainLiteral(0, SourceInfo()),
                SourceInfo()));
  m->Add<ContinuousAssignment>(SourceInfo(), data_out, data_reg);

  return m;
}

// Constructs a module which decodes an input byte based on whether the state
// machine is in an escaped state (previoius input byte was
// IOControlCode::kEscape). The module is purely combinational.
static absl::StatusOr<Module*> EscapeDecoderModule(VerilogFile* f) {
  Module* m = f->AddModule("escape_decoder", SourceInfo());
  LogicRef* byte_in =
      m->AddInput("byte_in", f->BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* byte_out =
      m->AddOutput("byte_out", f->BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* is_escaped =
      m->AddInput("is_escaped", f->ScalarType(SourceInfo()), SourceInfo());

  // Logic for the counter and shift register:
  //
  //   if (is_escaped && byte_in == IOEscapeCode::kResetByte) {
  //     byte_out = IOControlCode::kReset;
  //   } else if (is_escaped && byte_in == IOEscapeCode::kEscapeByte) {
  //     byte_out = IOControlCode::kEscape;
  //   } else {
  //     byte_out = byte_in;
  //   }
  LocalParamItemRef* escaped_reset_byte =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("EscapedResetByte",
                    Hex8Literal(IOEscapeCode::kResetByte, f), SourceInfo());
  LocalParamItemRef* escaped_escape_byte =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("EscapedEscapedByte",
                    Hex8Literal(IOEscapeCode::kResetByte, f), SourceInfo());
  LocalParamItemRef* reset_control_code =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("ResetControlCode", Hex8Literal(IOControlCode::kReset, f),
                    SourceInfo());
  LocalParamItemRef* escape_control_code =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("EscapeControlCode", Hex8Literal(IOControlCode::kEscape, f),
                    SourceInfo());
  LogicRef* byte_out_reg = m->AddReg(
      "byte_out_reg", f->BitVectorType(8, SourceInfo()), SourceInfo());
  auto ac = m->Add<Always>(SourceInfo(), std::vector<SensitivityListElement>(
                                             {ImplicitEventExpression()}));
  auto cond = ac->statements()->Add<Conditional>(
      SourceInfo(),
      f->LogicalAnd(is_escaped,
                    f->Equals(byte_in, escaped_reset_byte, SourceInfo()),
                    SourceInfo()));
  cond->consequent()->Add<BlockingAssignment>(SourceInfo(), byte_out_reg,
                                              reset_control_code);
  cond
      ->AddAlternate(f->LogicalAnd(
          is_escaped, f->Equals(byte_in, escaped_escape_byte, SourceInfo()),
          SourceInfo()))
      ->Add<BlockingAssignment>(SourceInfo(), byte_out_reg,
                                escape_control_code);
  cond->AddAlternate()->Add<BlockingAssignment>(SourceInfo(), byte_out_reg,
                                                byte_in);

  m->Add<ContinuousAssignment>(SourceInfo(), byte_out, byte_out_reg);

  return m;
}

absl::StatusOr<Module*> InputControllerModule(const ModuleSignature& signature,
                                              VerilogFile* f) {
  XLS_ASSIGN_OR_RETURN(Module * reset_m, InputResetModule(f));
  XLS_ASSIGN_OR_RETURN(
      Module * shift_m,
      InputShiftRegisterModule(signature.TotalDataInputBits(), f));
  XLS_ASSIGN_OR_RETURN(Module * decoder_m, EscapeDecoderModule(f));

  Module* m = f->AddModule("input_controller", SourceInfo());
  LogicRef* clk = m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());

  // Byte-wide input with ready/valid flow control.
  LogicRef* byte_in =
      m->AddInput("byte_in", f->BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* byte_in_valid =
      m->AddInput("byte_in_valid", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* byte_in_ready =
      m->AddOutput("byte_in_ready", f->ScalarType(SourceInfo()), SourceInfo());

  // Arbitrary width output with ready/valid flow control.
  LogicRef* data_out = m->AddOutput(
      "data_out",
      f->BitVectorType(signature.TotalDataInputBits(), SourceInfo()),
      SourceInfo());
  LogicRef* data_out_ready =
      m->AddInput("data_out_ready", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* data_out_valid =
      m->AddOutput("data_out_valid", f->ScalarType(SourceInfo()), SourceInfo());

  // The external reset signal.
  LogicRef* rst_n_in =
      m->AddInput("rst_n_in", f->ScalarType(SourceInfo()), SourceInfo());

  // The reset signal generated by the input controller. This is based on the
  // external reset signal and any reset control code passed in via the input.
  LogicRef* rst_n_out =
      m->AddOutput("rst_n_out", f->ScalarType(SourceInfo()), SourceInfo());

  // The byte_in ready signal generated by the reset FSM. This is used to ack
  // the input byte when it is a reset control code.
  LogicRef* reset_fsm_byte_in_ready = m->AddWire(
      "reset_fsm_byte_in_ready", f->ScalarType(SourceInfo()), SourceInfo());
  {
    std::vector<Connection> connections;
    connections.push_back(Connection{"clk", clk});
    connections.push_back(Connection{"byte_in", byte_in});
    connections.push_back(Connection{"byte_in_valid", byte_in_valid});
    connections.push_back(Connection{"byte_in_ready", reset_fsm_byte_in_ready});
    connections.push_back(Connection{"rst_n_in", rst_n_in});
    connections.push_back(Connection{"rst_n_out", rst_n_out});
    m->Add<Instantiation>(SourceInfo(), reset_m->name(), "resetter",
                          /*parameters=*/absl::Span<const Connection>(),
                          connections);
  }

  // Shift register used to accumulate the input bytes into an arbitrary width
  // register for passing to the device function.
  LogicRef* shifter_clear =
      m->AddReg("shifter_clear", f->ScalarType(SourceInfo()), SourceInfo(),
                /*init=*/f->Literal(UBits(1, 1), SourceInfo()));
  LogicRef* shifter_byte_in = m->AddWire(
      "shifter_byte_in", f->BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* shifter_write_en =
      m->AddReg("shifter_write_en", f->ScalarType(SourceInfo()), SourceInfo(),
                f->Literal(UBits(0, 1), SourceInfo()));
  LogicRef* shifter_done =
      m->AddWire("shifter_done", f->ScalarType(SourceInfo()), SourceInfo());
  {
    std::vector<Connection> connections;
    connections.push_back(Connection{"clk", clk});
    connections.push_back(Connection{"clear", shifter_clear});
    connections.push_back(Connection{"byte_in", shifter_byte_in});
    connections.push_back(Connection{"write_en", shifter_write_en});
    connections.push_back(Connection{"data_out", data_out});
    connections.push_back(Connection{"done", shifter_done});
    m->Add<Instantiation>(SourceInfo(), shift_m->name(), "shifter",
                          /*parameters=*/absl::Span<const Connection>(),
                          connections);
  }

  // TODO(meheff): Expose use_system_verilog as an option in the WrapIO API
  // rather than hard-coding it as false.
  FsmBuilder fsm("rx_fsm", m, clk, /*use_system_verilog=*/false,
                 Reset{rst_n_out, /*asynchronous=*/false, /*active_low=*/true});
  auto init_state = fsm.AddState("Init");
  auto idle_state = fsm.AddState("Idle");
  auto input_valid_state = fsm.AddState("InputValid");
  auto data_done_state = fsm.AddState("DataDone");

  auto shifter_clear_output = fsm.AddExistingOutput(
      shifter_clear, /*default_value=*/f->PlainLiteral(0, SourceInfo()));
  auto shifter_write_en_output = fsm.AddExistingOutput(
      shifter_write_en, /*default_value=*/f->PlainLiteral(0, SourceInfo()));
  auto data_out_valid_output = fsm.AddOutput1("data_out_valid_reg", 0);
  auto byte_in_ready_output = fsm.AddOutput1("byte_in_ready_reg", 0);

  auto is_escaped_reg =
      fsm.AddRegister("is_escaped", f->ScalarType(SourceInfo()),
                      f->PlainLiteral(0, SourceInfo()));

  // The initial state clears the input shift register.
  init_state->SetOutput(shifter_clear_output, 1).NextState(idle_state);

  idle_state->OnCondition(shifter_done)
      .NextState(data_done_state)
      .ElseOnCondition(byte_in_valid)
      .NextState(input_valid_state);

  input_valid_state->SetOutput(byte_in_ready_output, 1)
      .NextState(idle_state)

      // Not currently in escaped state and escape character received. Enter the
      // escaped state.
      .OnCondition(f->LogicalAnd(
          f->LogicalNot(is_escaped_reg->logic_ref, SourceInfo()),
          f->Equals(byte_in, Hex8Literal(IOControlCode::kEscape, f),
                    SourceInfo()),
          SourceInfo()))
      .SetRegisterNext(is_escaped_reg, 1)

      // Data byte received.
      .Else()
      .SetRegisterNext(is_escaped_reg, 0)
      .SetOutput(shifter_write_en_output, 1);

  // Input is complete. Assert output valid and wait for ready signal.
  data_done_state->SetOutput(data_out_valid_output, 1)
      .OnCondition(data_out_ready)
      .NextState(init_state);

  XLS_RETURN_IF_ERROR(fsm.Build());

  m->Add<ContinuousAssignment>(SourceInfo(), data_out_valid,
                               data_out_valid_output->logic_ref);

  // The byte_in_ready signal can come from the FSM or the reset module (in case
  // of receiving a reset IO code). Or them together to generate the output
  // signal.
  m->Add<ContinuousAssignment>(
      SourceInfo(), byte_in_ready,
      f->LogicalOr(byte_in_ready_output->logic_ref, reset_fsm_byte_in_ready,
                   SourceInfo()));

  // Filter all byte inputs through the escape decoder.
  {
    std::vector<Connection> connections;
    connections.push_back(Connection{"byte_in", byte_in});
    connections.push_back(Connection{"byte_out", shifter_byte_in});
    connections.push_back(Connection{"is_escaped", is_escaped_reg->logic_ref});
    m->Add<Instantiation>(SourceInfo(), decoder_m->name(), "decoder",
                          /*parameters=*/absl::Span<const Connection>(),
                          connections);
  }

  return m;
}

absl::StatusOr<Module*> OutputControllerModule(const ModuleSignature& signature,
                                               VerilogFile* f) {
  const int64_t output_bits = signature.TotalDataOutputBits();

  Module* m = f->AddModule("output_controller", SourceInfo());
  LogicRef* clk = m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      m->AddInput("rst_n", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* data_in = m->AddInput(
      "data_in", f->BitVectorType(output_bits, SourceInfo()), SourceInfo());
  LogicRef* data_in_valid =
      m->AddInput("data_in_valid", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* data_in_ready =
      m->AddOutput("data_in_ready", f->ScalarType(SourceInfo()), SourceInfo());

  LogicRef* byte_out =
      m->AddOutput("byte_out", f->BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* byte_out_ready =
      m->AddInput("byte_out_ready", f->ScalarType(SourceInfo()), SourceInfo());
  LogicRef* byte_out_valid =
      m->AddOutput("byte_out_valid", f->ScalarType(SourceInfo()), SourceInfo());

  // TODO(meheff): Expose use_system_verilog as an option in the WrapIO API
  // rather than hard-coding it as false.
  FsmBuilder fsm("output_controller", m, clk, /*use_system_verilog=*/false,
                 Reset{rst_n, /*asynchronous=*/false, /*active_low=*/true});

  auto idle_state = fsm.AddState("Idle");
  auto shifting_state = fsm.AddState("Shifting");
  auto valid_state = fsm.AddState("Valid");
  auto holding_state = fsm.AddState("HoldingData");

  auto data_in_ready_output = fsm.AddOutput1("data_in_ready_reg", 0);
  auto byte_out_valid_output = fsm.AddOutput1("byte_out_valid_reg", 0);
  auto shift_reg = fsm.AddRegister("shift_out_reg", output_bits);

  const int64_t output_bytes = CeilOfRatio(output_bits, int64_t{8});
  auto byte_counter =
      fsm.AddRegister("byte_counter", Bits::MinBitCountUnsigned(output_bytes));

  idle_state->SetOutput(data_in_ready_output, 1)
      .OnCondition(data_in_valid)
      .SetRegisterNextAsExpression(shift_reg, data_in)
      .SetRegisterNext(byte_counter, output_bytes)
      .NextState(shifting_state);

  // Shift and output bytes one at a time until the byte counter reaches zero.
  shifting_state
      ->OnCondition(f->Equals(byte_counter->logic_ref,
                              f->PlainLiteral(0, SourceInfo()), SourceInfo()))
      .NextState(idle_state)
      .Else()
      .SetOutput(byte_out_valid_output, 1)
      .NextState(valid_state);

  // The tx UART requires asserting byte_out valid for a cycle before checking
  // byte_out ready and holding the data for a cycle after byte_out_ready is
  // asserted (called done in the UART code). These additional states add the
  // necessary delays.
  // TODO(meheff): convert the UARTs to a ready/valid interface.
  valid_state->SetOutput(byte_out_valid_output, 1)
      .OnCondition(byte_out_ready)
      .NextState(holding_state);

  holding_state->NextState(shifting_state)
      .SetOutput(byte_out_valid_output, 0)
      .SetRegisterNextAsExpression(
          shift_reg, f->Shrl(shift_reg->logic_ref,
                             f->PlainLiteral(8, SourceInfo()), SourceInfo()))
      .SetRegisterNextAsExpression(
          byte_counter, f->Sub(byte_counter->logic_ref,
                               f->PlainLiteral(1, SourceInfo()), SourceInfo()));

  XLS_RETURN_IF_ERROR(fsm.Build());

  // The data output of the module is the LSB of the shift register.
  m->Add<ContinuousAssignment>(
      SourceInfo(), byte_out,
      f->Slice(shift_reg->logic_ref, f->PlainLiteral(7, SourceInfo()),
               f->PlainLiteral(0, SourceInfo()), SourceInfo()));
  m->Add<ContinuousAssignment>(SourceInfo(), byte_out_valid,
                               byte_out_valid_output->logic_ref);
  m->Add<ContinuousAssignment>(SourceInfo(), data_in_ready,
                               data_in_ready_output->logic_ref);

  return m;
}

}  // namespace verilog
}  // namespace xls
