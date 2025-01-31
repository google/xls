// RUN: xls_translate --mlir-xls-to-xls %s -- --ffi_fallback_delay_ps=100 --generator=combinational 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=XLS
// RUN: xls_translate --mlir-xls-to-verilog %s -- --ffi_fallback_delay_ps=100 --generator=combinational 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=VERILOG

// XLS: #[ffi_proto("""code_template: "int_to_float {fn}(.x({x}), .out({return}))"
// XLS: fn __struct_type__int_to_float
// XLS-SAME: x

// VERILOG: int_to_float __struct_type__int_to_float__bitsreturn___invoke_{{.*}}_inst(.x({{.*}}), .out({{.*}}))

xls.import_dslx_file_package "xls/contrib/mlir/testdata/struct_type.x" as @struct_type
func.func private @int_to_float(%a: i32) -> f32 attributes
  {xls.linkage = #xls.translation_linkage<@struct_type:"int_to_float", foreign>}
func.func private @float_to_int(%a: f32) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@struct_type:"float_to_int", foreign>}

func.func @main(%arg0: i32) -> i32 attributes {xls = true} {
  %0 = call @int_to_float(%arg0) : (i32) -> f32
  %1 = call @float_to_int(%0) : (f32) -> i32
  return %1 : i32
}
