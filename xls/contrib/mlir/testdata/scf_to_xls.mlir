// RUN: xls_opt -scf-to-xls %s 2>&1 | FileCheck %s

// CHECK:       func.func @reduce(%arg0: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = xls.for inits(%c0_i32) invariants(%arg0) {
// CHECK-NEXT:    ^bb0(%indvar: index, %carry: i32, %invariant: i32):
// CHECK-NEXT:      %1 = arith.addi %carry, %invariant : i32
// CHECK-NEXT:      xls.yield %1 : i32
// CHECK-NEXT:    } {trip_count = 1024 : i64} : (i32, i32) -> i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
func.func @reduce(%addend: i32) -> (i32) attributes { "xls" = true } {
  %lb = arith.constant 0 : index
  %ub = arith.constant 1024 : index
  %step = arith.constant 1 : index
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : i32
  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (i32) {
    %sum_next = arith.addi %sum_iter, %addend : i32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %sum_next : i32
  }
  return %sum : i32
}

// CHECK:      func.func @reduce_arity_2(%arg0: i32, %arg1: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0:2 = xls.for inits(%c0_i32, %c0_i32) invariants(%arg1, %arg0) {
// CHECK-NEXT:    ^bb0(%indvar: index, %carry: i32, %carry_0: i32, %invariant: i32, %invariant_1: i32):
// CHECK-NEXT:      %1 = arith.muli %carry, %invariant : i32
// CHECK-NEXT:      %2 = arith.addi %1, %invariant_1 : i32
// CHECK-NEXT:      xls.yield %2, %carry_0 : i32, i32
// CHECK-NEXT:    } {trip_count = 1024 : i64} : (i32, i32, i32, i32) -> (i32, i32)
// CHECK-NEXT:    return %0#0 : i32
// CHECK-NEXT:  }
func.func @reduce_arity_2(%addend: i32, %mulend: i32) -> (i32) attributes { "xls" = true } {
  %lb = arith.constant 0 : index
  %ub = arith.constant 1024 : index
  %step = arith.constant 1 : index
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : i32
  // iter_args binds initial values to the loop's region arguments.
  %sum:2 = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0, %carry = %sum_0) -> (i32, i32) {
    %tmp = arith.muli %sum_iter, %mulend : i32
    %sum_next = arith.addi %tmp, %addend : i32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %sum_next, %carry : i32, i32
  }
  return %sum#0 : i32
}

// CHECK:       func.func @triple_nest(%arg0: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = xls.for inits(%c0_i32) invariants(%arg0) {
// CHECK-NEXT:    ^bb0(%indvar: index, %carry: i32, %invariant: i32):
// CHECK-NEXT:      %1 = xls.for inits(%carry) invariants(%invariant) {
// CHECK-NEXT:      ^bb0(%indvar_0: index, %carry_1: i32, %invariant_2: i32):
// CHECK-NEXT:        %2 = xls.for inits(%carry_1) invariants(%carry_1, %invariant_2) {
// CHECK-NEXT:        ^bb0(%indvar_3: index, %carry_4: i32, %invariant_5: i32, %invariant_6: i32):
// CHECK-NEXT:          %3 = arith.addi %invariant_5, %invariant_6 : i32
// CHECK-NEXT:          xls.yield %3 : i32
// CHECK-NEXT:        } {trip_count = 1024 : i64} : (i32, i32, i32) -> i32
// CHECK-NEXT:        xls.yield %2 : i32
// CHECK-NEXT:      } {trip_count = 1024 : i64} : (i32, i32) -> i32
// CHECK-NEXT:      xls.yield %1 : i32
// CHECK-NEXT:    } {trip_count = 1024 : i64} : (i32, i32) -> i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
func.func @triple_nest(%addend: i32) -> (i32) attributes { "xls" = true } {
  %lb = arith.constant 0 : index
  %ub = arith.constant 1024 : index
  %step = arith.constant 1 : index
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : i32
  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.for %iv = %lb to %ub step %step
      iter_args(%sum_iter = %sum_0) -> (i32) {
    %sum2 = scf.for %iv2 = %lb to %ub step %step
        iter_args(%sum_iter2 = %sum_iter) -> (i32) {
      %sum3 = scf.for %iv3 = %lb to %ub step %step
          iter_args(%sum_iter3 = %sum_iter2) -> (i32) {
        %sum_next = arith.addi %sum_iter2, %addend : i32
        scf.yield %sum_next : i32
      }
      scf.yield %sum3 : i32
    }
    scf.yield %sum2 : i32
  }
  return %sum : i32
}

// CHECK:       func.func @if(%arg0: i32, %arg1: i1) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = xls.not %arg1 : i1
// CHECK-NEXT:    %1 = arith.addi %c0_i32, %arg0 : i32
// CHECK-NEXT:    %2 = arith.subi %c0_i32, %arg0 : i32
// CHECK-NEXT:    %3 = xls.sel %arg1 in [%2] else %1 : (i1, [i32], i32) -> i32
// CHECK-NEXT:    return %3 : i32
// CHECK-NEXT:  }
func.func @if(%addend: i32, %cond: i1) -> (i32) attributes { "xls" = true } {
  %lb = arith.constant 0 : index
  %ub = arith.constant 1024 : index
  %step = arith.constant 1 : index
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : i32
  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.if %cond -> (i32) {
    %a = arith.addi %sum_0, %addend : i32
    scf.yield %a : i32
  } else {
    %b = arith.subi %sum_0, %addend : i32
    scf.yield %b : i32
  }
  return %sum : i32
}

// CHECK-LABEL: func.func @single_armed_if(%arg0: i32, %arg1: i1) attributes {xls = true} {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = xls.not %arg1 : i1
// CHECK-NEXT:    %1 = arith.addi %c0_i32, %arg0 : i32
// CHECK-NEXT:    %2 = arith.subi %c0_i32, %arg0 : i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @single_armed_if(%addend: i32, %cond: i1) -> () attributes { "xls" = true } {
  %sum_0 = arith.constant 0 : i32
  scf.if %cond {
    %a = arith.addi %sum_0, %addend : i32
    scf.yield
  } else {
    %b = arith.subi %sum_0, %addend : i32
    scf.yield
  }
  return
}

// CHECK-LABEL:   @index_switch
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = "xls.constant_scalar"() <{value = 2 : index}> : () -> index
// CHECK-NEXT:    %1 = xls.eq %arg0, %0 : (index, index) -> i1
// CHECK-NEXT:    %2 = "xls.constant_scalar"() <{value = 5 : index}> : () -> index
// CHECK-NEXT:    %3 = xls.eq %arg0, %2 : (index, index) -> i1
// CHECK-NEXT:    %4 = xls.or %1, %3 : i1
// CHECK-NEXT:    %5 = xls.not %4 : i1
// CHECK-NEXT:    %6 = "xls.constant_scalar"() <{value = 2 : index}> : () -> index
// CHECK-NEXT:    %7 = xls.eq %arg0, %6 : (index, index) -> i1
// CHECK-NEXT:    %8 = "xls.constant_scalar"() <{value = 5 : index}> : () -> index
// CHECK-NEXT:    %9 = xls.eq %arg0, %8 : (index, index) -> i1
// CHECK-NEXT:    %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:    %10 = xls.after_all  : !xls.token
// CHECK-NEXT:    %11 = xls.trace %10, %7, "a"
// CHECK-NEXT:    %c20_i32 = arith.constant 20 : i32
// CHECK-NEXT:    %12 = xls.after_all  : !xls.token
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %13 = xls.and %9, %false : i1
// CHECK-NEXT:    %14 = xls.trace %12, %13, "a"
// CHECK-NEXT:    %15 = "xls.constant_scalar"() <{value = 2 : index}> : () -> index
// CHECK-NEXT:    %16 = xls.eq %arg0, %15 : (index, index) -> i1
// CHECK-NEXT:    %17 = "xls.constant_scalar"() <{value = 5 : index}> : () -> index
// CHECK-NEXT:    %18 = xls.eq %arg0, %17 : (index, index) -> i1
// CHECK-NEXT:    %19 = xls.concat %18, %16 : (i1, i1) -> i2
// CHECK-NEXT:    %20 = xls.priority_sel %19 in[%c10_i32, %c20_i32] else %c0_i32 : (i2, [i32, i32], i32) -> i32
// CHECK-NEXT:    return %20 : i32
func.func @index_switch(%index: index) -> (i32) attributes { "xls" = true } {
  %sum_0 = arith.constant 0 : i32
  %sw = scf.index_switch %index -> i32
  case 2 {
    %1 = arith.constant 10 : i32
    %tok = xls.after_all : !xls.token
    %tok_out = xls.trace %tok, "a" verbosity 0
    scf.yield %1 : i32
  }
  case 5 {
    %2 = arith.constant 20 : i32
    %tok2 = xls.after_all : !xls.token
    %never = arith.constant 0 : i1
    %tok_out2 = xls.trace %tok2, %never, "a" verbosity 0
    scf.yield %2 : i32
  }
  default {
    scf.yield %sum_0 : i32
  }
  return %sw : i32
}
