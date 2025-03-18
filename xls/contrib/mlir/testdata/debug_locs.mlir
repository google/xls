// RUN: xls_translate --mlir-xls-to-xls --main-function=foo_proc %s > %t
// RUN: FileCheck --check-prefix=XLS %s < %t
// RUN: xls_translate --xls-to-mlir-xls %t --mlir-print-debuginfo | FileCheck --check-prefix=MLIR %s

// XLS: file_number [[$FNO:.*]] "add_one.x"

module {
  // XLS-LABEL:  fn add_one(arg_foo: bits[32]
  // MLIR-LABEL: func.func @add_one
  // MLIR-SAME:    i32 loc("arg_foo")
  func.func @add_one(%arg0: i32 loc("arg_foo")) -> i32 {

    // MLIR-LABEL: "xls.constant_scalar"
    // MLIR-SAME:    loc(#[[$LOC1:.*]])
    // XLS-LABEL:  literal
    // XLS-SAME:     pos=[([[$FNO]],42,7)]
    %0 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32 loc(#loc1)

    // MLIR-LABEL: xls.add
    // MLIR-SAME:    loc(#[[$LOC2:.*]])
    // XLS-LABEL:  add
    // XLS-SAME:     pos=[([[$FNO]],1999,3)]
    %1 = xls.add %arg0, %0 : i32 loc(#loc2)

    return %1 : i32 loc(unknown)
  } loc(unknown)

  // XLS-LABEL:  proc foo_proc(state_elem: bits[32]
  // MLIR-LABEL: xls.eproc @foo_proc
  // MLIR-SAME:    i32 loc("state_elem")
  xls.eproc @foo_proc(%arg0: i32 loc("state_elem")) zeroinitializer {
    %result = func.call @add_one(%arg0) : (i32) -> i32
    xls.yield %result : i32
  }

} loc(unknown)


// MLIR-CHECK: #[[$LOC1]] = loc("add_one.x":42:7)
#loc1 = loc("add_one.x":42:7)
// MLIR-CHECK: #[[$LOC2]] = loc("add_one.x":1999:3)
#loc2 = loc("add_one.x":1999:3)
