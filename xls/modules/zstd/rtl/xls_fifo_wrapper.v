// simple fifo implementation
module xls_fifo_wrapper (
clk, rst,
push_ready, push_data, push_valid,
pop_ready,  pop_data,  pop_valid);
  parameter Width = 32,
            Depth = 32,
            EnableBypass = 0,
            RegisterPushOutputs = 1,
            RegisterPopOutputs = 1;
  localparam AddrWidth = $clog2(Depth) + 1;
  input  wire             clk;
  input  wire             rst;
  output wire             push_ready;
  input  wire [Width-1:0] push_data;
  input  wire             push_valid;
  input  wire             pop_ready;
  output wire [Width-1:0] pop_data;
  output wire             pop_valid;

  // Require depth be 1 and bypass disabled.
  //initial begin
  //  if (EnableBypass || Depth != 1 || !RegisterPushOutputs || RegisterPopOutputs) begin
  //    // FIFO configuration not supported.
  //    $fatal(1);
  //  end
  //end


  reg [Width-1:0] mem;
  reg full;

  assign push_ready = !full;
  assign pop_valid = full;
  assign pop_data = mem;

  always @(posedge clk) begin
    if (rst == 1'b1) begin
      full <= 1'b0;
    end else begin
      if (push_valid && push_ready) begin
        mem <= push_data;
        full <= 1'b1;
      end else if (pop_valid && pop_ready) begin
        mem <= mem;
        full <= 1'b0;
      end else begin
        mem <= mem;
        full <= full;
      end
    end
  end
endmodule
