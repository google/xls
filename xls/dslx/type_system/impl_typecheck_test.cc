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

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TypecheckTest, StaticConstantOnStruct) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

fn point_dims() -> u32 {
    Point::NUM_DIMS
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, MissingFunctionOnStruct) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

fn point_dims() -> u32 {
    Point::num_dims()
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Function with name 'num_dims' is not defined "
                                 "by the impl for struct 'Point'")));
}

TEST(TypecheckErrorTest, MissingImplOnStruct) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

fn point_dims() -> u32 {
    Point::num_dims()
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct 'Point' has no impl defining 'num_dims'")));
}

TEST(TypecheckErrorTest, ImplWithConstCalledAsFunc) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const num_dims = u32:4;
}

fn point_dims() -> u32 {
    Point::num_dims()
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Function with name 'num_dims' is not defined "
                                 "by the impl for struct 'Point'")));
}

TEST(TypecheckTest, StaticFunctionOnStruct) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn num_dims() -> u32 {
        u32:2
    }
}

fn point_dims() -> u32 {
    Point::num_dims()
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplFunctionUsingStructMembers) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

fn point_dims() -> u8 {
    let p = Point{x: u32:4, y:u32:2};
    let y = p.area();
    uN[y]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// TODO: Support imported impl methods.
TEST(TypecheckTest, DISABLED_ImportedImplUsingStructMembers) {
  constexpr std::string_view kImported = R"(
pub struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[8] {
    let p = imported::Point{x: u32:4, y:u32:2};
    uN[p.area()]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, ImplsForDifferentStructs) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

struct Line { a: Point, b: Point }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

impl Line {
    fn height(self) -> u32 {
        self.b.y - self.a.y
    }
}

fn point_dims() -> u10 {
    let p = Point{x: u32:4, y:u32:2};
    let y = p.area(); // 8
    let l = Line{a: p, b: Point{x: u32:4, y: u32:4}};
    let h = l.height(); // 2

    uN[y + h]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplFunctionUsingStructMembersOnConst) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

fn point_dims() -> u8 {
    const p = Point{x: u32:4, y:u32:2};
    let y = p.area();
    uN[y]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplFunctionUsingStructMembersIndirect) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

fn point_dims() -> u8 {
    const p = Point{x: u32:4, y:u32:2};
    let y = p;
    uN[y.area()]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplFunctionUsingStructMembersMultIndirect) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

fn point_dims() -> u8 {
    const p = Point{x: u32:4, y:u32:2};
    let y = p;
    let x = y;
    let w = x;
    uN[w.area()]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, ImplMethodCalledStaticallyNoParams) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self) -> u32 {
        self.x * self.y
    }
}

fn point_dims() -> u16 {
    let y = Point::area();
    uN[y]:0
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected 1 parameter(s) but got 0 arguments.")));
}

TEST(TypecheckTest, ImplFunctionUsingStructMembersAndArg) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self, a: u32, b: u32) -> u32 {
        self.x * self.y * a * b
    }
}

fn point_dims() -> u16 {
    let p = Point{x: u32:4, y:u32:2};
    let y = p.area(u32:2, u32:1);
    uN[y]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplFunctionUsingStructMembersExplicitSelfType) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn area(self: Self) -> u32 {
        self.x * self.y
    }
}

fn point_dims() -> u8 {
    let p = Point{x: u32:4, y:u32:2};
    let y = p.area();
    uN[y]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, StaticFunctionUsingConst) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const DIMS = u32:2;

    fn num_dims() -> u32 {
        DIMS
    }
}

fn point_dims() -> u2 {
    uN[Point::num_dims()]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, StaticConstUsingFunction) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    fn num_dims() -> u32 {
        u32:2
    }

    const DIMS = num_dims();
}

fn point_dims() -> u2 {
    uN[Point::DIMS]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, ImplConstantOutsideScope) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

const GLOBAL_DIMS = NUM_DIMS;
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot find a definition")));
}

TEST(TypecheckTest, ImplConstantExtracted) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

const GLOBAL_DIMS = Point::NUM_DIMS;
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, ConstantExtractionWithoutImpl) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

const GLOBAL_DIMS = Point::NUM_DIMS;
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct 'Point' has no impl defining 'NUM_DIMS'")));
}

TEST(TypecheckErrorTest, ConstantAccessWithoutImplDef) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

fn point_dims() -> u32 {
    Point::NUM_DIMS
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct 'Point' has no impl defining 'NUM_DIMS'")));
}

TEST(TypecheckErrorTest, ImplWithMissingConstant) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

fn point_dims() -> u32 {
    Point::DIMENSIONS
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "'DIMENSIONS' is not defined by the impl for struct 'Point'")));
}

TEST(TypecheckTest, ImplWithTypeAlias) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

type ThisPoint = Point;

fn use_point() -> u2 {
    let size = ThisPoint::NUM_DIMS;
    uN[size]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, ImplWithTypeAliasWrongType) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

impl Point {
    const NUM_DIMS = u32:2;
}

type ThisPoint = Point;

fn use_point() -> u4 {
    let size = ThisPoint::NUM_DIMS;
    uN[size]:0
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("did not match the annotated return type")));
}

TEST(TypecheckErrorTest, TypeAliasConstantAccessWithoutImplDef) {
  constexpr std::string_view kProgram = R"(
struct Point { x: u32, y: u32 }

type ThisPoint = Point;

fn point_dims() -> u32 {
    ThisPoint::NUM_DIMS
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Struct 'Point' has no impl defining 'NUM_DIMS'")));
}

TEST(TypecheckTest, ImplWithMultipleStructInstance) {
  constexpr std::string_view kProgram = R"(
struct Point<X: u32> {}

impl Point<X> {
    const DOUBLE = u32:2 * X;
    const ZERO = uN[DOUBLE]:0;
}

fn use_points() {
     type Point2 = Point<u32:2>;
     type Point4 = Point<u32:4>;
     assert_eq(Point2::ZERO, u4:0);
     assert_eq(Point4::ZERO, u8:0);
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

// TODO(google/xls#1277): Allow parsing colonrefs with TypeAnnotation as the
// subject.
TEST(TypecheckTest, DISABLED_ImplWithParametricStruct) {
  constexpr std::string_view kProgram = R"(
struct Point<MAX_X: u32> { x: u32, y: u32 }

impl Point<MAX_X> {
    const NUM_DIMS = u32:2;
    const MAX_WIDTH = u32:2 * MAX_X;
}

fn use_point() -> u10 {
    uN[Point<u32:5>::MAX_WIDTH]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, DISABLED_ImplWithParametricStructFromFn) {
  constexpr std::string_view kProgram = R"(
struct Point<MAX_X: u32> { x: u32, y: u32 }

impl Point<MAX_X> {
    const NUM_DIMS = u32:2;
    const MAX_WIDTH = u32:2 * MAX_X;
}

fn use_point<MAX_X: u32>() -> uN[u32:2 * MAX_X] {
    uN[Point<MAX_X>::MAX_WIDTH]:0
}

fn main() -> u10 {
    use_point<u32:5>()
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplWithParametricTypeAlias) {
  constexpr std::string_view kProgram = R"(
struct APFloat<EXP_SZ: u32, FRACTION_SZ: u32> {
    sign: bits[1],
    bexp: bits[EXP_SZ],
    fraction: bits[FRACTION_SZ],
}

impl APFloat<EXP_SZ, FRACTION_SZ> {
    const EXP = EXP_SZ;
}

type BF16 = APFloat<u32:8, u32:7>;

fn bf_ezp() -> u8 {
    uN[BF16::EXP]:0
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImplUsingDerivedParametric) {
  constexpr std::string_view kProgram = R"(
struct Point<MAX_X: u32, MAX_Y: u32 = { MAX_X * u32:2 }> { x: u32, y: u32 }

impl Point<MAX_X, MAX_Y> {
    const NUM_DIMS = u32:2;
    const MAX_WIDTH = u32:2 * MAX_X;
    const MAX_HEIGHT = u32:2 * MAX_Y;
}

fn use_point() -> u8 {
    type MyPoint = Point<u32: 2>;
    uN[MyPoint::MAX_HEIGHT]:0
}

)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ImportedImpl) {
  constexpr std::string_view kImported = R"(
pub struct Empty { }

impl Empty {
   const IMPORTED = u32:6;
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[6] {
    uN[imported::Empty::IMPORTED]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, ImportedImplWithFunction) {
  constexpr std::string_view kImported = R"(
pub struct Empty { }

impl Empty {
   fn imported_func() -> u32 {
       u32:6
   }
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[6] {
    uN[imported::Empty::imported_func()]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, ImportedImplParametric) {
  constexpr std::string_view kImported = R"(
pub struct Empty<X: u32> { }

impl Empty<X> {
   const IMPORTED = u32:2 * X;
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[6] {
    type MyEmpty = imported::Empty<u32: 3>;
    uN[MyEmpty::IMPORTED]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, ParametricImplInstantiatedByGlobal) {
  constexpr std::string_view program = R"(
struct MyStruct<WIDTH: u32> {
  f: bits[WIDTH]
}

impl MyStruct<WIDTH> {
  const EXP = WIDTH;
}

const GLOBAL = u32:15;
fn f() -> u15 {
   type GlobalStruct = MyStruct<GLOBAL>;
   uN[GlobalStruct::EXP]:0
}

)";
  XLS_EXPECT_OK(Typecheck(program));
}

TEST(TypecheckTest, ImportedImplTypeAlias) {
  constexpr std::string_view kImported = R"(
pub struct Empty<X: u32> { }

impl Empty<X> {
   const IMPORTED = u32:2 * X;
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

type MyEmpty = imported::Empty<u32:5>;

fn main() -> uN[10] {
    uN[MyEmpty::IMPORTED]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, ImportedImplTypeAliasWithFunction) {
  constexpr std::string_view kImported = R"(
pub struct Empty { }

impl Empty {
   fn some_val() -> u32 {
       u32:4
   }
}

)";
  constexpr std::string_view kProgram = R"(
import imported;

type MyEmpty = imported::Empty;

fn main() -> uN[4] {
    uN[MyEmpty::some_val()]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

TEST(TypecheckTest, ImportedTypeAlias) {
  constexpr std::string_view kImported = R"(
pub struct Empty<X: u32> { }

impl Empty<X> {
   const IMPORTED = u32:2 * X;
}

pub type MyEmpty = Empty<u32:5>;
)";
  constexpr std::string_view kProgram = R"(
import imported;

fn main() -> uN[10] {
    uN[imported::MyEmpty::IMPORTED]:0
})";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule module,
      ParseAndTypecheck(kImported, "imported.x", "imported", &import_data));
  XLS_EXPECT_OK(
      ParseAndTypecheck(kProgram, "fake_main_path.x", "main", &import_data));
}

}  // namespace
}  // namespace xls::dslx
