module identity_of_mul(
  input wire clk,
  input wire [7:0] x,
  input wire [7:0] y,
  output wire [7:0] out
);
  // lint_off MULTIPLY
  function automatic [7:0] umul8b_8b_x_8b (input reg [7:0] lhs, input reg [7:0] rhs);
    begin
      umul8b_8b_x_8b = lhs * rhs;
    end
  endfunction
  // lint_on MULTIPLY

  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [7:0] p0_x;
  reg [7:0] p0_y;
  always @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
  end

  // ===== Pipe stage 1:
  wire [7:0] p1_umul_11_comb;
  assign p1_umul_11_comb = umul8b_8b_x_8b(p0_x, p0_y);

  // Registers for pipe stage 1:
  reg [7:0] p1_identity_12;
  always @ (posedge clk) begin
    p1_identity_12 <= p1_umul_11_comb;
  end
  assign out = p1_identity_12;
endmodule
