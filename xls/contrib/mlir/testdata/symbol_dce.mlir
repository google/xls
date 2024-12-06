// RUN: xls/contrib/mlir/xls_opt %s -symbol-dce -- 2>&1 | FileCheck %s

// CHECK-NOT: @mychan
// CHECK-NOT: @dot_product
xls.chan @mychan : i32
xls.import_dslx_file_package "xls/contrib/mlir/testdata/i32/dot_product.x" as @dot_product

// CHECK-NOT: @sproc
// CHECK-NOT: @extern_sproc
xls.extern_sproc @extern_sproc (arg0: !xls.schan<i32, in>)
xls.sproc @sproc(%arg0: !xls.schan<i32, in>) {
  spawns {
    xls.spawn @extern_sproc(%arg0) : !xls.schan<i32, in>
    xls.yield
  }
  next(%state: i32) zeroinitializer {
    xls.yield %state : i32
  }
}

// CHECK: @top_sproc
// CHECK: @top_extern_sproc
xls.extern_sproc @top_extern_sproc (arg0: !xls.schan<i32, in>)
xls.sproc @top_sproc(%arg0: !xls.schan<i32, in>) top {
  spawns {
    xls.spawn @top_extern_sproc(%arg0) : !xls.schan<i32, in>
    xls.yield
  }
  next(%state: i32) zeroinitializer {
    xls.yield %state : i32
  }
}
