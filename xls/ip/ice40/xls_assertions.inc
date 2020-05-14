// Assertions to be `included into Verilog testbench modules for testing.
//
// TODO(leary): Move to third_party/xls/ip/common/xls_assertions.inc when we
// have a way to package up verilog modules for iverilog usage.

task automatic xls_assert(
  input want,
  input got,
  input reg [32*8:0] message
);
  if (want != got) begin
    $display("ERROR: assertion failure @ %t; want %b got %b; %s\n", $time,
             want, got, message);
  end
endtask

task automatic xls_assert_int_eq(
  input integer want,
  input integer got,
  input reg [32 * 8:0] message);
  if (want != got) begin
    $display("ERROR: assertion failure @ %t; want %x got %x; %s\n", $time,
             want, got, message);
  end
endtask
