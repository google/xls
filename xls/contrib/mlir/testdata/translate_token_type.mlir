// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=XLS

// XLS: fn tokenret{{.*}}: token{{.*}}) -> token
func.func @tokenret(%arg0: !xls.token loc("a")) -> !xls.token {
  return %arg0 : !xls.token
}
