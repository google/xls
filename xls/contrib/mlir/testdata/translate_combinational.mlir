// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls %s --main-function=combinational -- 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=XLS
// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-verilog %s --main-function=combinational -- 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=VERILOG-COMB

// XLS: fn combinational([[ARG0:.*]]: bits[8]{{.*}}, [[ARG1:.*]]: bits[8]{{.*}}, [[ARG2:.*]]: bits[8]{{.*}}) -> bits[8] {
// XLS:   [[DIFF:[a-z0-9.]+]]: bits[8] = sub([[ARG0]], [[ARG1]]
// XLS:   [[UMUL1:[a-z0-9.]+]]: bits[8] = umul([[DIFF]], [[DIFF]]
// XLS:   [[UMUL2:[a-z0-9.]+]]: bits[8] = umul([[ARG2]], [[DIFF]]
// XLS:   ret
// XLS-SAME: add([[UMUL1]], [[UMUL2]]
// XLS: fn combinational_with_tuple({{.*}}) -> (bits[8], bits[16])

// VERILOG-COMB: module combinational(
// VERILOG-COMB:     input wire [7:0] [[ARG0:.*]],
// VERILOG-COMB:     input wire [7:0] [[ARG1:.*]],
// VERILOG-COMB:     input wire [7:0] [[ARG2:.*]],
// VERILOG-COMB:     output wire [7:0] [[OUT:.*]]
// VERILOG-COMB: function automatic [7:0] [[UMULFN:.*]] (
// VERILOG-COMB:      = {{.*}} * {{.*}};
// VERILOG-COMB:   assign [[DIFF:.*]] = [[ARG0]] - [[ARG1]];
// VERILOG-COMB:   assign [[UMUL1:.*]] = [[UMULFN]]([[DIFF]], [[DIFF]]);
// VERILOG-COMB:   assign [[UMUL2:.*]] = [[UMULFN]]([[ARG2]], [[DIFF]]);
// VERILOG-COMB:   assign [[OUT:.*]]= [[UMUL1]] + [[UMUL2]];
// VERILOG-COMB: endmodule

module @pkg {
func.func @combinational(%a: i8, %b: i8, %c: i8) -> i8 {
  %diff = xls.sub %a, %b: i8
  %umul.5 = xls.umul %diff, %diff : i8
  %umul.6 = xls.umul %c, %diff : i8
  %the_output = xls.add %umul.5, %umul.6 : i8
  return %the_output : i8
}

func.func @combinational_with_tuple(%a: i8, %b: i16) -> (i8, i16) {
  return %a, %b : i8, i16
}
}

