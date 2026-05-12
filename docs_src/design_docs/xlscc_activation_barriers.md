# XLS[cc] Scoped Conditional Activation Barriers.

[TOC]

## Basic Principle

XLS[cc] has the concept of “activation barriers”, which solve mutual exclusion
errors by putting IO ops into a new activation after the
barrier. In order to avoid cycles, the condition of ops after the barrier must
not depend directly on the data received from ops before it. This seems
infeasible in the general case:


```c++
int x = in.read();
if (x > 10) {
	__xlscc_activation_barrier</*conditional=*/true>();
}
// ...
if (x == 20) {
	int y = in.read();
	// ...
}
```

## Solution: Scoped barriers

It can be made feasible for many useful cases by taking advantage of C++ scopes:

```c++
int x = in.read();
if (x > 10) {
	__xlscc_activation_barrier</*conditional=*/true>();
	int y = in.read();
	// ...
}
```

In this situation, we know that y = in.read() cannot be active in the same
activation as x = in.read(), because the barrier appears before y = in.read() in
the same scope.

However, “!(x > 10)” cannot be directly added to the condition for y =
in.read(), because there would still be a data dependency.


## Implementation of scoped mutual exclusion

There are two areas to address:

1.  How does the FSM determine which slices are active in a given activation
    without referencing any side-effecting values across the conditional 
    barrier?
    - For unconditional barriers, this is trivial: nothing after the next
    barrier can be active, and this is known just from this activation’s
    starting slice index

2. How does the FSM ensure that the function slices do not reference any
    side-effecting values across the conditional barrier?
    - For unconditional barriers, this is trivial: every continuation input can
    be reset to directly reference state after the barrier

### Slice activation calculation

Determining when the scoped slices are active is feasible based on the principle
that slices in the conditional scope cannot be active in the same activation as
the barrier (start) slice. Either the condition is:

- True: In which case they will be active in the next activation
- False: In which they will never be active this Run() iteration

If the activation starts within the scoped slices, then they can be active.

Slices after the scope must be inactive if the barrier is active this iteration.
This is achieved using the direct, side-effecting value referencing condition of
the barrier. Each barrier resets this after-barrier condition to literal 0, so
adding another barrier can remove this direct dependency.

This rule works with nested scopes because in order to reach a barrier in an
inner scope, all the outer scopes’ barriers’ conditions must also have been
true. Therefore, if the outer most scope’s condition was:

- True: Then it must be true that slices after the scope are after an active barrier
- False: Then it must be false that slices after the scope are after an active barrier

### Example A, data_in receives values 1, 1, 4:
```c++
//   Inactive this activation
//** Inactive this activation via direct reference to received value(s)
```

```c++
    const int a = data_in.read();
    if (a == 1) {
      __xlscc_activation_barrier</*conditional=*/true>();
//    const int b = data_in.read();
//    if (b == 1) {
//      __xlscc_activation_barrier</*conditional=*/true>();
//      const int c = data_in.read();
//      data_out.write(a + b + c);
//    }
//    ctrl_a.write(100);
    }
    ctrl_b.write(2);
```


```c++
//  const int a = data_in.read();
//  if (a == 1) {
//    __xlscc_activation_barrier</*conditional=*/true>();
      const int b = data_in.read();
      if (b == 1) {
        __xlscc_activation_barrier</*conditional=*/true>();
//      const int c = data_in.read();
//      data_out.write(a + b + c);
//    }
//**  ctrl_a.write(100);
    }
//**ctrl_b.write(2);
```


```c++
//  const int a = data_in.read();
//  if (a == 1) {
//    __xlscc_activation_barrier</*conditional=*/true>();
//    const int b = data_in.read();
//    if (b == 1) {
//      __xlscc_activation_barrier</*conditional=*/true>();
        const int c = data_in.read();
        data_out.write(a + b + c);
      }
      ctrl_a.write(100);
  }
  ctrl_b.write(2);
```

### Example A, data_in receives values 1, 4:
```c++
//   Inactive this activation
//** Inactive this activation via direct reference to received value(s)
```
Activation 0:

```c++
    const int a = data_in.read();
    if (a == 1) {
      __xlscc_activation_barrier</*conditional=*/true>();
//    const int b = data_in.read();
//    if (b == 1) {
//      __xlscc_activation_barrier</*conditional=*/true>();
//      const int c = data_in.read();
//      data_out.write(a + b + c);
//    }
//    ctrl_a.write(100);
    }
//**ctrl_b.write(2);
```

Activation 1:


```c++
//  const int a = data_in.read();
//  if (a == 1) {
//    __xlscc_activation_barrier</*conditional=*/true>();
      const int b = data_in.read();
      if (b == 1) {
        __xlscc_activation_barrier</*conditional=*/true>();
        const int c = data_in.read();
        data_out.write(a + b + c);
      }
      ctrl_a.write(100);
    }
    ctrl_b.write(2);
```
### Example A, data_in receives value 4:
Activation 0:


```c++
    const int a = data_in.read();
    if (a == 1) {
      __xlscc_activation_barrier</*conditional=*/true>();
      const int b = data_in.read();
      if (b == 1) {
        __xlscc_activation_barrier</*conditional=*/true>();
        const int c = data_in.read();
        data_out.write(a + b + c);
      }
      ctrl_a.write(100);
    }
    ctrl_b.write(2);
```

### Example B, data_in receives values 11, 10, 1:
```c++
//   Inactive this activation
//** Inactive this activation via direct reference to received value(s)
```

Activation 0:

```c++
    const int a = data_in.read();
    int b = 0;
    if (a > 10) {
      __xlscc_activation_barrier</*conditional=*/true>();
//    b = data_in.read();
//  }

//**__xlscc_activation_barrier</*conditional=*/true>();
//**int c = data_in.read();
//**data_out.write(a + b + c);
```

Activation 1:

```c++
//  const int a = data_in.read();
//  int b = 0;
//  if (a > 10) {
//    __xlscc_activation_barrier</*conditional=*/true>();
      b = data_in.read();
    }

    __xlscc_activation_barrier</*conditional=*/false>();  // conditional=true is equivalent
//  int c = data_in.read();
//  data_out.write(a + b + c);
```

Activation 2:

```c++
//  const int a = data_in.read();
//  int b = 0;
//  if (a > 10) {
//    __xlscc_activation_barrier</*conditional=*/true>();
//    b = data_in.read();
//  }

//  __xlscc_activation_barrier</*conditional=*/false>();  // conditional=true is equivalent
    int c = data_in.read();
    data_out.write(a + b + c);
```


## Continuation value handling

Another source of cycles can be direct references from after the barrier to
side-effecting values from before it. Section 1 only handles the implicit slice
activity condition. Although operations before and after the barrier are
mutually exclusive, according to the slice activity, XLS will still see a cycle
and be unable to merge the operations.

For example:

```c++
const int a = data_in.read();
if (a <= 5) {
  __xlscc_activation_barrier</*conditional=*/true>();
  int b = 0;
  // Introduces data dependency
  if (a == 1) {
    b = data_in.read();
  }
  data_out.write(a + b);
}
```

With unconditional barriers, it is safe to simply remap every continuation value
input* after the barrier to reference the continuation value’s state element:
that is, to reference the value from the last activation. This is safe because
an unconditional barrier will always proceed to the next activation, and so
nothing after it can be active to use the value before it is loaded into the
state element. For conditional barriers, it is possible that the condition will
be false, in which case the FSM will fall through the barrier within the same
activation, and the state elements will not have been updated. The solution to
this is, again, scoping, this time of continuation input references.

Each conditional barrier’s scope maintains its own map of continuation value
input references, much like the variable maps in the TranslationContext. When a
new scope is started, the parent’s map is copied, and all applicable
continuation inputs are reset to use state elements. When the scope ends,
propagation down to the parent scope is done, so that direct references are
referenced for any continuation values produced by slices inside the scope. This
is because it is possible to fall out of the scope within a single activation
(see section 1).

When the conditional barrier’s condition is: True: Continuation values produced
by the scoped slices are fully valid, being computed from state elements that
were updated during the activation transition. False: Continuation values
produced by the scoped slices are invalid, referencing state values from a
previous activation. However, they should never be used, since the select phis
generated in the slice after the end of the scope will always ignore them.

* With exceptions like direct-ins
