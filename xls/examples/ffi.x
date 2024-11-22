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
// a nested tuple output.
//
// The template references {A_WIDTH} and {B_WIDTH} are replaced by evaluated
// dslx parametric values.
//
// Parameters to the function an return values are accessed by name
// in the text template. The otherwise unnamed return value can be accessed
// with the {return} name in the template.
// If the {return} value would be simply a scalar, that would be the
// chosen name. Since we have a tuple as return value, the individual elements
// are referenced via the index-dotted path such as {return.0.1}.
//
// Input: "a", "b", output is a tuple ((a / b, a % b), error_division_by_zero)
// If b is zero, error_division_by_zero == true and tuple contains original
// values.
#[extern_verilog("external_divmod #(
     .dividend_width({A_WIDTH}),  // Refer to local DSLX symbols in curly braces,
     .divisor_width({B_WIDTH})    // such as template parameters
    ) {fn} (                      // fn will be replace w/ instance name.
     .dividend({a}),              // Function parameter referenced in template
     .divisor({b}),
     .quotient({return.0.0}),   // Similar return. Here we return a tuple ...
     .remainder({return.0.1}),  // ... values referenced as return.0 and return.1
     .by_zero({return.1})
    )")]
fn divmod<A_WIDTH: u32, B_WIDTH: u32>
    (a: bits[A_WIDTH], b: bits[B_WIDTH]) -> ((bits[A_WIDTH], bits[B_WIDTH]), bool) {
    if b == u32:0 { ((a, b), true) } else { ((a / b, a % b), false) }
}

fn main(dividend: u32, divisor: u32) -> ((u32, u32), bool) { divmod(dividend, divisor) }

// In regular testing and bytecode interpretation, the DSLX function is used.
// Only in the finaly code-generation, the function invocation is replaced with
// the user-chosen module.
#[test]
fn divmod_test() {
    assert_eq(divmod(u32:42, u32:5), ((u32:8, u32:2), false));
    assert_eq(divmod(u32:42, u32:0), ((u32:42, u32:0), true));
}
