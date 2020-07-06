import textwrap
import functools
from xls.common import runfiles
from absl import logging, app
from xls.dslx import fakefs_util
from xls.dslx import parser_helpers
from xls.dslx import typecheck
from xls.dslx import ir_converter
from xls.dslx import import_routines
from xls.dslx import extract_conversion_order
from absl import logging

def parse_dsl_text(program):
    program = textwrap.dedent(program)
    filename = '/fake/test_module.x'
    with fakefs_util.scoped_fakefs(filename, program):
      m = parser_helpers.parse_text(
          program, name='test_module', print_on_error=True, filename=filename)
      return m

def main(argv):
    m = parse_dsl_text("""\
        fn [N: u32 = u32:10, M: u32 = N * u32:2] add_two(x: bits[N]) -> bits[M] {
           x as bits[M] + bits[M]: 2
        }
        fn get_bool() -> bool { true }

        fn foo() -> bool {
          let a = add_two(u7:5) in
          let b = add_two(u10:10) in
          get_bool()
        }""")
    m2 = parse_dsl_text("""fn [X: u32, Y: u32, Z: u32 = X + Y] p(x: bits[X], z: bits[Z]) -> bits[X] { u10: 5 }\
    fn f() -> u5 { p(u10:3, u5:3) }""")
    m3 = parse_dsl_text("""\
        fn times_two(v: u32) -> u32 { v * u32: 2 }
          fn [X: u32, Y: u32 = times_two(X)] double (x: bits[X]) -> uN[Y] {
             let y: bits[Y] = x++x in
             y
          }
          fn [A: u32, B: u32 = A + A] foo (x: bits[A]) -> uN[B] {
             double(x)
          }
          fn y () -> uN[10] {
            foo(u5: 5)
          }""")
    m4 = parse_dsl_text("""\
        fn [X: u32] f(x: uN[X]) -> uN[X+X] { x++x }
        fn [A: u32] g(a: uN[A]) -> uN[A+A] { f(a) }
        fn main() -> u6 { g(u3:0) }""")
    m5 = parse_dsl_text("""\
        import std

const A = u32[3]:[10, 20, 30];
const B = u16[5]:[1, 2, 3, 4, 5];

fn main(i: u32) -> (bool, u32) {
  std::find_index(A, i)
}

fn fazz(j: u16) -> (bool, u32) {
  std::find_index(B, j)
}
""")
    m6 = parse_dsl_text("""\
    fn [X: u32] foo(x: bits[X]) -> u1 {
        let non_polymorphic  =  x + u5: 1 in
        u1: 0
    }
    fn main() -> bits[1] {
        foo(u5: 5) | foo(u10: 5)
    }
    """)
    m7 = parse_dsl_text("""\
    fn [Y: u32] h(y: bits[Y]) -> bits[Y] { h(y) }
    fn g() -> u32[3] {
        let x0 = u32[3]:[0, 1, 2] in
        map(x0, h)
    }
    """)
    import_cache = {}
    f_import = functools.partial(import_routines.do_import, cache=import_cache)
    node_to_type = typecheck.check_module(m5, f_import=f_import)
    print("typecheck ok")
    #order = extract_conversion_order.get_order(m5, node_to_type.get_imports())
    converted = ir_converter.convert_module_nodump(m5, node_to_type)
    #print(m.get_function("foo"))
    print(converted.dump_ir())

if __name__ == '__main__':
    app.run(main)

"""

m arg or X: u32 = u32: 10
derived: X: u32 = X + Y
symbols:

fn [X: u32, Y: u32 = X + X] double (x: bits[X]) -> uN[Y] {
   x++x
}
fn [A: u32, B: u32 = A + A ] foo (x: bits[A]) -> uN[B] {
   double(x)
}
fn y () {
  foo(u5: 5)
}
"""



"""
py_binary(
    name = "ir_converter_interpreter",
    srcs = ["ir_converter_interpreter.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":import_routines",
        ":ir_converter",
        ":fakefs_util",
        ":parser_helpers",
        ":span",
        ":typecheck",
	"//xls/dslx/interpreter:interpreter_helpers",
        "@com_google_absl_py//absl:app",
        "@com_google_absl_py//absl/flags",
    ],
)
"""
