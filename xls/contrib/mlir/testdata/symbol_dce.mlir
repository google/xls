// RUN: xls/contrib/mlir/xls_opt %s -symbol-dce -- 2>&1 | FileCheck %s

// CHECK-NOT: @mychan
// CHECK-NOT: @dot_product
xls.chan @mychan : i32
xls.import_dslx_file_package "xls/contrib/mlir/testdata/i32/dot_product.x" as @dot_product
