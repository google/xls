// RUN: xls/contrib/mlir/xls_translate -mlir-xls-to-xls %s 2>&1

// Verify that importing different functions from files with the same stem get
// resolved properly. This is a regression test for where we did not consider
// the mangled name post merging.

xls.import_dslx_file_package "xls/contrib/mlir/testdata/i32/dot_product.x" as @dot_product
func.func private @dot_32(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i32>) -> i32 attributes
  {xls.linkage = #xls.translation_linkage<@dot_product:"dot_product_fixed_test">}

xls.import_dslx_file_package "xls/contrib/mlir/testdata/i16/dot_product.x" as @dot_product_0
func.func private @dot_16(%a: !xls.array<4 x i16>, %b: !xls.array<4 x i16>) -> i16 attributes
  {xls.linkage = #xls.translation_linkage<@dot_product_0:"dot_product_fixed_test">}

func.func @sub(%a: !xls.array<4 x i32>, %b: !xls.array<4 x i16>) -> (i32, i16) {
  %0 = call @dot_32(%a, %a) : (!xls.array<4 x i32>, !xls.array<4 x i32>) -> i32
  %1 = call @dot_16(%b, %b) : (!xls.array<4 x i16>, !xls.array<4 x i16>) -> i16
  func.return %0, %1 : i32, i16
}
