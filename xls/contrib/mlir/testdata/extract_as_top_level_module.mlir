// RUN: xls_opt -test-extract-as-top-level-module -split-input-file -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: Testing : "simple"
module attributes {test.name = "simple"} {
  // CHECK-NOT: func @nope
  // CHECK: func @simple() -> i32
  // CHECK-NOT: func @nope
  func.func @simple() -> i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
  func.func @nope() ->i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: Testing : "eproc"
module attributes {test.name = "eproc"} {
  // CHECK-NOT: func @nope
  // CHECK: xls.chan @yeschan {recv_supported = false} : i32
  // CHECK-NOT: xls.chan @nochan
  // CHECK: xls.eproc @eproc
  // CHECK-NOT: func @nope
  xls.chan @yeschan : i32
  xls.chan @nochan : i32
  func.func @nope(%arg0: !xls.token, %arg1: i32, %arg2: i1) ->i32 {
    %0 = arith.constant 1 : i32
    %r = xls.send %arg0, %arg1, %arg2, @nochan : i32
    return %0 : i32
  }
  xls.eproc @eproc(%arg0: i32, %arg1: !xls.token, %arg2: i32, %arg3: i1) zeroinitializer {
    %0 = arith.constant 1 : i32
    %r = xls.send %arg1, %arg2, %arg3, @yeschan : i32
    xls.yield %arg0, %arg1, %arg2, %arg3 : i32, !xls.token, i32, i1
  }
}

// -----

// CHECK-LABEL: Testing : "transitive"
module attributes {test.name = "transitive"} {
  // CHECK: xls.import_dslx_file_package
  // CHECK: func.func private @bar2
  // CHECK-NOT: func.func private @bar3
  // CHECK: func @maybe
  // CHECK: func @yes
  // CHECK: func @transitive() -> i32

  xls.import_dslx_file_package "xls/contrib/mlir/testdata/i32/dot_product.x" as @dot_product
  func.func private @bar2(%a: i32, %b: i32) -> i32 attributes
    {xls.linkage = #xls.translation_linkage<@dot_product:"dot_product_fixed_test">}
  func.func private @bar3(%a: i32, %b: i32) -> i32 attributes
    {xls.linkage = #xls.translation_linkage<@dot_product:"dot_product_fixed_test">}
  func.func @maybe() ->i32 {
    %0 = arith.constant 1 : i32
    call @bar2(%0, %0) : (i32, i32) -> i32
    return %0 : i32
  }
  func.func @no() ->i32 {
    %0 = func.call @maybe() : () -> i32
    return %0 : i32
  }
  func.func @yes() ->i32 {
    %0 = func.call @maybe() : () -> i32
    return %0 : i32
  }
  func.func @transitive() -> i32 {
    %0 = func.call @yes() : () -> i32
    return %0 : i32
  }
}

// -----

// CHECK-LABEL: Testing : "sproc"
module attributes {test.name = "sproc"} {
  // CHECK-NOT: func @nope
  // CHECK: xls.sproc @leaf
  // CHECK-NOT: xls.sproc @unrelated
  // CHECK: xls.sproc @sproc(%arg0: !xls.schan<i32, in>) top attributes {boundary_channel_names = ["arg0"]}
  func.func @nope(%arg0: !xls.token, %arg1: i32, %arg2: i1) ->i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }

  xls.sproc @leaf(%arg0: !xls.schan<i32, in>) {
    spawns {
      xls.yield
    }
    next(%state: i32) zeroinitializer {
      xls.yield %state : i32
    }
  }

  xls.sproc @sproc(%arg0: !xls.schan<i32, in>) {
    spawns {
      xls.spawn @leaf(%arg0) : !xls.schan<i32, in>
      xls.yield
    }
    next(%state: i32) zeroinitializer {
      xls.yield %state : i32
    }
  }

  xls.sproc @unrelated(%arg0: !xls.schan<i32, in>) {
    spawns {
      xls.spawn @leaf(%arg0) : !xls.schan<i32, in>
      xls.yield
    }
    next(%state: i32) zeroinitializer {
      xls.yield %state : i32
    }
  }
}
