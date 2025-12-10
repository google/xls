// RUN: xls_opt -skip-empty-top-eproc=top-proc-name=bar -split-input-file %s 2>&1 | FileCheck %s


// CHECK-LABEL: module
// CHECK-NEXT:   xls.eproc @bar(%arg0: i1, %arg1: i32) zeroinitializer {
// CHECK-NEXT:     xls.next_value [%arg0, %arg1], [%arg0, %arg1] : (i32, i32) -> i32
// CHECK-NEXT:     xls.yield
// CHECK-NEXT:   }
// CHECK: xls.instantiate_eproc @bar (@x as @pred, @y as @arg)
// CHECK-NOT: @foo

module {

xls.eproc @foo(%pred: i1, %arg: i32) zeroinitializer {
  %0 = xls.next_value [%pred, %arg], [%pred, %arg] : (i32, i32) -> i32
  xls.yield %pred, %0 : i1, i32
}

xls.chan @x : i1
xls.chan @pred : i1
xls.chan @y : i32
xls.chan @arg : i32
xls.instantiate_eproc @foo (@x as @pred, @y as @arg)

xls.eproc @bar() zeroinitializer {
  xls.yield
}
xls.instantiate_eproc @bar ()

}

// -----

// CHECK-LABEL: module
// CHECK-NEXT:   xls.eproc @bar(%arg0: i1, %arg1: i32) zeroinitializer {
// CHECK-NEXT:     xls.next_value [%arg0, %arg1], [%arg0, %arg1] : (i32, i32) -> i32
// CHECK-NEXT:     xls.yield
// CHECK-NEXT:   }
// CHECK: xls.instantiate_eproc @bar (@x as @pred, @y as @arg)
// CHECK-NOT: @foo

module {

xls.eproc @bar() zeroinitializer {
  xls.yield
}
xls.instantiate_eproc @bar ()

xls.eproc @foo(%pred: i1, %arg: i32) zeroinitializer {
  %0 = xls.next_value [%pred, %arg], [%pred, %arg] : (i32, i32) -> i32
  xls.yield %pred, %0 : i1, i32
}

xls.chan @x : i1
xls.chan @pred : i1
xls.chan @y : i32
xls.chan @arg : i32
xls.instantiate_eproc @foo (@x as @pred, @y as @arg)

}

// -----


// This test won't be rewritten, since the top eproc isn't the empty one.
// CHECK-LABEL: module
// CHECK: xls.eproc @foo()
// CHECK: xls.eproc @bar(%{{.*}}: i1, %{{.*}}: i32)

module {

xls.eproc @foo() zeroinitializer {
  xls.yield
}
xls.instantiate_eproc @foo ()

xls.eproc @bar(%pred: i1, %arg: i32) zeroinitializer {
  %0 = xls.next_value [%pred, %arg], [%pred, %arg] : (i32, i32) -> i32
  xls.yield %pred, %0 : i1, i32
}

xls.chan @x : i1
xls.chan @pred : i1
xls.chan @y : i32
xls.chan @arg : i32
xls.instantiate_eproc @bar (@x as @pred, @y as @arg)

}
