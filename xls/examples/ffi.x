// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Example how to instantiate an existing Verilog module, with the DSLX
// foreign function interface.


// A function that we want to interface with some existing verilog module
// we implement the function itself in DSLX and provide a text template that
// will be used to create an instantiation of the module with the correct
// parameters; this template is provided in the extern_verilog attribute.
//
// Here a simple sample function that illustrates parameter inputs and
// a tuple output.
// Input: "a", "b", output is a tuple (a / b, a % b, error_division_by_zero)
// If b is zero, error_division_by_zero == true and tuple contains original
// values
#[extern_verilog("external_divmod #(
     .dividend_width($bits({a})),  // Refer to DSLX symbols in curly braces.
     .divisor_width($bits({b}))
    ) {fn} (                  // fn will be replace w/ generated instance name.
     .dividend({a}),          // function parameters are passed to as-is
     .divisor({b}),
     .quotient({return.0}),   // Similar return. Here we return a tuple ...
     .remainder({return.1}),  // ... values referenced as return.0 and return.1
     .by_zero({return.2})
    )")]
fn divmod<A_WIDTH:u32, B_WIDTH:u32>(a:bits[A_WIDTH], b:bits[B_WIDTH])
                                    -> (bits[A_WIDTH], bits[B_WIDTH], bool) {
    if (b == u32:0) {
        (a, b, true)
    } else {
        (a / b, a - b * (a / b), false)   // ... no mod % operation in dslx yet
    }
}


fn main(dividend:u32, divisor:u32) -> (u32, u32, bool) {
  divmod(dividend, divisor)
}

// In regular testing and bytecode interpretation, the DSLX function is used.
// Only in the finaly code-generation, the function invocation is replaced with
// the user-chosen module.
#[test]
fn divmod_test() {
  assert_eq(divmod(u32:42, u32:5), (u32:8, u32:2, false));
  assert_eq(divmod(u32:42, u32:0), (u32:42, u32:0, true));
}
