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

// Implementation: we count down in quarter-baud amounts, sampling three
// points from the signal, at 1/4, 1/2, and 3/4. If they agree, we're happy.
// If they disagree, something is wrong and we transition to an error state.
//
// This comes at some complexity / area, but is nicer than sampling once in
// the anticipated center of the baud, because it can help us detect things
// like misconfigured baud rates or faulty transmitters. Ideally how we sample
// the signal could be some pluggable strategy, but we just choose the more
// robust checking-sampler here.
//
// If we see a voltage transition before the "4/4" baud point, we *could*
// adjust our notion of how long a baud actually is. However, given that the
// stop-bit / start-bit transition already is a "synchronizing" event, we'll
// avoid doing fancy things like that.

module uart_receiver(
    input wire        clk,
    input wire        rst_n,
    input wire        rx,
    input wire        rx_byte_done,
    output wire       clear_to_send_out,
    output wire [7:0] rx_byte_out,
    output wire       rx_byte_valid_out
);
  parameter ClocksPerBaud = 1250;

  localparam QuarterBaud   = (ClocksPerBaud >> 2);
  localparam CountdownStart = QuarterBaud-1;

  // Note: we shave one bit off the full clocks-per-baud-sized value because
  // we only count down from the quarter-baud value.
  localparam CountdownBits = $clog2(ClocksPerBaud >> 1);
  localparam
    StateIdle              = 'd0,
    StateStart             = 'd1,
    StateData              = 'd2,
    StateStop              = 'd3,
    // Sampled inconsistent values from the receive line.
    StateInconsistentError = 'd4,
    // Attempted to receive a byte when the previous byte hasn't been popped.
    StateClobberError      = 'd5,
    // Did not sample three '1' values for stop bit.
    StateStopBitError      = 'd6;
  localparam StateBits = 3;

  `define ERROR_STATES StateInconsistentError,StateClobberError,StateStopBitError

  reg [StateBits-1:0]     state         = StateIdle;
  reg [2:0]               samples       = 0;
  reg [1:0]               sample_count  = 0;
  reg [2:0]               data_bitno    = 0;
  reg [CountdownBits-1:0] rx_countdown  = 0;
  reg [7:0]               rx_byte       = 0;
  reg                     rx_byte_valid = 0;

  reg [StateBits-1:0]     state_next;
  // The data we've sampled from the current baud.
  reg [2:0]               samples_next;
  // Tracks the number of samples we've collected from the current baud.
  reg [1:0]               sample_count_next;
  // Tracks the bit number that we're populating in the received byte.
  reg [2:0]               data_bitno_next;
  // Counts down to the next sampling / state transition activity.
  reg [CountdownBits-1:0] rx_countdown_next;
  // Storage for the received byte.
  reg [7:0]               rx_byte_next;
  // Indicator that the received byte is complete and ready for consumption.
  reg                     rx_byte_valid_next;

  // Note: only in an idle or stop state can the outside world signal that the
  // received byte is done (that is, without breaking the interface contract).
  // Also, any time that the rx bit is not 1, something is currently in the
  // process of being transmitted.
  assign clear_to_send_out =
    (state == StateIdle || state == StateStop) && (rx_byte_valid == 0) && (rx == 1);
  assign rx_byte_out       = rx_byte;
  assign rx_byte_valid_out = rx_byte_valid;

  // Several states transition when the countdown is finished and we've
  // sampled three bits.
  wire ready_to_transition;
  assign ready_to_transition = rx_countdown == 0 && sample_count == 3;

  // Check that the samples all contain the same sampled bit-value.
  // This is the case when the and-reduce is equivalent to the or-reduce.
  wire samples_consistent;
  assign samples_consistent = &(samples) == |(samples);

  // State updates.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    state_next = state;
    case (state)
      StateIdle: begin
        if (rx == 0) begin
          state_next = rx_byte_valid ? StateClobberError : StateStart;
        end
      end
      StateStart: begin
        if (rx_countdown == 0 && sample_count == 3) begin
          state_next = samples_consistent ? StateData : StateInconsistentError;
        end
      end
      StateData: begin
        if (rx_countdown == 0 && sample_count == 3) begin
          if (!samples_consistent) begin
            state_next = StateInconsistentError;
          end else if (data_bitno == 7) begin
            state_next = StateStop;
          end else begin
            state_next = StateData;
          end
        end
      end
      StateStop: begin
        if (rx_countdown == 0 && sample_count == 3) begin
          if (&(samples) != 1) begin
            // All stop-bit samples should be one.
            state_next = StateStopBitError;
          end else if (rx == 0) begin
            // Permit transition straight to start state.
            state_next = StateStart;
          end else begin
            // If the rx signal isn't already dropped to zero (indicating
            // a start bit), we transition through the idle state.
            state_next = StateIdle;
          end
        end
      end
      `ERROR_STATES: begin
        // Stay in this state until we're externally reset.
      end
      default: begin
        state_next = 3'hX;
      end
    endcase
  end

  // Non-state updates.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    samples_next       = samples;
    sample_count_next  = sample_count;
    data_bitno_next    = data_bitno;
    rx_countdown_next  = rx_countdown;
    rx_byte_next       = rx_byte;
    rx_byte_valid_next = rx_byte_valid;

    case (state)
      StateIdle: begin
        samples_next = 0;
        sample_count_next = 0;
        rx_countdown_next = CountdownStart;
        data_bitno_next = 0;
        // Maintain the current "valid" value, dropping it if the outside
        // world claims the byte is done.
        rx_byte_valid_next = rx_byte_valid && !rx_byte_done;
      end
      StateStart: begin
        if (rx_countdown == 0) begin
          // Perform a sample.
          samples_next[sample_count] = rx;
          sample_count_next = sample_count + 1;
          rx_countdown_next = CountdownStart;
        end else begin
          rx_countdown_next = rx_countdown - 1;
        end
        // Note: the following line is unnecessary, just makes waveforms
        // easier to read (not shifting out garbage data).
        rx_byte_next = 0;
        data_bitno_next = 0;
      end
      StateData: begin
        if (rx_countdown == 0) begin
          if (sample_count == 3) begin
            // Shift in the newly sampled bit. (Once we're done, the first
            // sampled bit will be the LSb.)
            rx_byte_next = {samples[0], rx_byte[7:1]};
            // We flag the byte is done as early as we can, so we flop the new
            // state in before we transition to the StateStop state. (Note:
            // when we've just shifted in bitno 7 the data is ready).
            rx_byte_valid_next = data_bitno == 7 && samples_consistent;
            data_bitno_next = data_bitno + 1;
            sample_count_next = 0;
          end else begin
            samples_next[sample_count] = rx;
            sample_count_next = sample_count + 1;
          end
          rx_countdown_next = CountdownStart;
        end else begin
          rx_countdown_next = rx_countdown - 1;
        end
      end
      StateStop: begin
        // If the caller has dropped the valid signal by flagging "byte done"
        // during the stop state we keep that property "sticky".
        rx_byte_valid_next = rx_byte_valid == 0 || rx_byte_done ? 0 : 1;
        // Set this up because we may transition StateStop->StateStart
        // directly.
        data_bitno_next = 0;
        if (rx_countdown == 0) begin
          samples_next[sample_count] = rx;
          sample_count_next = sample_count + 1;
          rx_countdown_next = CountdownStart;
        end else begin
          rx_countdown_next = rx_countdown - 1;
        end
      end
      `ERROR_STATES: begin
        rx_byte_next = 'hff;
        rx_byte_valid_next = 0;
      end
      default: begin
        samples_next       = 'hX;
        sample_count_next  = 'hX;
        data_bitno_next    = 'hX;
        rx_countdown_next  = 'hX;
        rx_byte_next       = 'hX;
        rx_byte_valid_next = 'hX;
      end
    endcase
  end

  // Note: our version of iverilog has no support for always_ff.
  always @ (posedge clk) begin
    if (rst_n == 0) begin
      state         <= StateIdle;
      samples       <= 0;
      sample_count  <= 0;
      data_bitno    <= 0;
      rx_countdown  <= 0;
      rx_byte       <= 0;
      rx_byte_valid <= 0;
    end else begin
      state         <= state_next;
      samples       <= samples_next;
      sample_count  <= sample_count_next;
      data_bitno    <= data_bitno_next;
      rx_countdown  <= rx_countdown_next;
      rx_byte       <= rx_byte_next;
      rx_byte_valid <= rx_byte_valid_next;
    end
  end

endmodule
