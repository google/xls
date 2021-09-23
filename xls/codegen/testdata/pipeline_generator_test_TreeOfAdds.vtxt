module x_plus_y_plus_z(
  input wire clk,
  input wire [31:0] a,
  input wire [31:0] b,
  input wire [31:0] c,
  input wire [31:0] d,
  input wire [31:0] e,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [31:0] p0_a;
  reg [31:0] p0_b;
  reg [31:0] p0_c;
  reg [31:0] p0_d;
  reg [31:0] p0_e;
  always @ (posedge clk) begin
    p0_a <= a;
    p0_b <= b;
    p0_c <= c;
    p0_d <= d;
    p0_e <= e;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_add_25_comb;
  wire [31:0] p1_add_26_comb;
  assign p1_add_25_comb = p0_a + p0_b;
  assign p1_add_26_comb = p1_add_25_comb + p0_c;

  // Registers for pipe stage 1:
  reg [31:0] p1_d;
  reg [31:0] p1_e;
  reg [31:0] p1_add_26;
  always @ (posedge clk) begin
    p1_d <= p0_d;
    p1_e <= p0_e;
    p1_add_26 <= p1_add_26_comb;
  end

  // ===== Pipe stage 2:
  wire [31:0] p2_add_33_comb;
  wire [31:0] p2_add_34_comb;
  assign p2_add_33_comb = p1_add_26 + p1_d;
  assign p2_add_34_comb = p2_add_33_comb + p1_e;

  // Registers for pipe stage 2:
  reg [31:0] p2_add_34;
  always @ (posedge clk) begin
    p2_add_34 <= p2_add_34_comb;
  end
  assign out = p2_add_34;
endmodule
