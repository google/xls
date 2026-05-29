// Copyright 2026 The XLS Authors
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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/ir_convert/ir_converter_test_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/run_comparator.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/proc.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST_F(IrConverterTest, ExplicitStateAccessU32) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc Counter {
  init { 0 }
  config() { }
  next(state: u32) {
    let x = read(state);
    let y = x + 1;
    write(state, y);
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(PackageConversionData conv,
                           ConvertModuleToPackage(tm.module, &import_data,
                                                  kProcScopedChannelOptions));
  ExpectIr(conv.DumpIr());
}

TEST_F(IrConverterTest, ExplicitStateAccessS32) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc Counter {
  init { -5 }
  config() { }
  next(state: s32) {
    let current = read(state);
    write(state, current + 1);
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(PackageConversionData conv,
                           ConvertModuleToPackage(tm.module, &import_data,
                                                  kProcScopedChannelOptions));
  ExpectIr(conv.DumpIr());
}

TEST_F(IrConverterTest, ExplicitStateAccessString) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc String {
  init { "hello" }
  config() { }
  next(state: u8[5]) {
    let current = read(state);
    write(state, "world");
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(PackageConversionData conv,
                           ConvertModuleToPackage(tm.module, &import_data,
                                                  kProcScopedChannelOptions));
  ExpectIr(conv.DumpIr());
}

TEST_F(IrConverterTest, ExplicitStateAccessBool) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc Bool {
  init { false }
  config() { }
  next(state: bool) {
    let false_val = read(state);
    write(state, true);
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(PackageConversionData conv,
                           ConvertModuleToPackage(tm.module, &import_data,
                                                  kProcScopedChannelOptions));
  ExpectIr(conv.DumpIr());
}

TEST_F(IrConverterTest, ExplicitStateAccessStruct) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
struct Point {
  x: u32,
  y: u32,
}
proc Struct {
  init { Point { x: 0, y: 0 } }
  config() { }
  next(state: Point) {
    let curr_point = read(state);
    let shift = Point { x: curr_point.x + 1, y: curr_point.y + 1 };
    write(state, shift)
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(PackageConversionData conv,
                           ConvertModuleToPackage(tm.module, &import_data,
                                                  kProcScopedChannelOptions));
  ExpectIr(conv.DumpIr());
}

TEST_F(IrConverterTest, ExplicitStateAccessArray) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc Array {
  init { [0, 1, 2, 3] }
  config() { }
  next(state: u32[4]) {
    let array = read(state);
    let new_array = for (i, arr) in 0..4 {
      update(arr, i, arr[i] + 1)
    }(array);
    write(state, new_array);
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(PackageConversionData conv,
                           ConvertModuleToPackage(tm.module, &import_data,
                                                  kProcScopedChannelOptions));
  ExpectIr(conv.DumpIr());
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleReads) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {

  config() { () }

  init { 0 }

  next(state: u32) {
      let first = read(state);
      let second = read(state);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleWrites) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  config() { () }

  init { 0 }

  next(state: u32) {
      let accum = read(state) + 1;
      write(state, accum);
      let accum = accum + 1;
      write(state, accum);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessWriteBeforeRead) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  config() { () }

  init { 0 }

  next(state: u32) {
    write(state, u32:1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessConditionalWrite) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  config() { () }

  init { (0, false) }

  next(state: (u32, bool)) {
    let even_or_odd = read(state);
    if (even_or_odd.0 % 2 == 0) {
      write(state, (even_or_odd.0 + 1, true))
    } else {
      write(state, (even_or_odd.0 + 1, false))
    }
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMatch) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  config() { () }
  init { 0 }
  next(state: u32) {
    let current = read(state);
    let even_or_odd = current % 2;
    match even_or_odd {
      0 => {
        write(state, current + 1);
      },
      1 => {
        write(state, current * 2);
      },
      _ => {
        write(state, current);
      }
    }
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMatchMultipleWrites) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  config() { () }
  init { true }
  next(state: bool) {
    let val = read(state);
    match val {
      true => {
        write(state, false);
        write(state, false);
      },
      false => {
        write(state, true);
        write(state, true);
      },
    }
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessLabeledReadAndWrite) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  init { 0 }
  config() { }
  next(state: u32) {
    let x = 'main_read:read(state);
    let y = x + 1;
    'main_write:write(state, y);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessReadWithLabeledRead) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  init { 0 }
  config() { }
  next(state: u32) {
    let curr = read(state);
    let x = 'main_read:read(state);
    let y = x + 1 + curr;
    'main_write:write(state, y);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleStates) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
struct Point {
  x: u32,
  y: u32,
}

proc main {
  init { (Point { x: 0, y: 1 }, (2, 3), 4) }
  config() { }
  next(state_0: Point, state_1: (u32, u32), state_2: u32) {
    let a = read(state_0);
    let b = read(state_1);
    let c = read(state_2);
    let new_a = Point { x: a.x + 1, y: b.1 + c };
    let new_b = (b.0 + 1, c + 1);
    write(state_0, new_a);
    write(state_1, new_b);
    write(state_2, c + 2);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleStatesMultipleReads) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  init { (0, 1) }
  config() { }
  next(state_0: u32, state_1: u32) {
    let b_0 = read(state_1);
    let b_1 = read(state_1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleStatesMultipleWrites) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  init { (0, 1) }
  config() { }
  next(state_0: u32, state_1: u32) {
    let a = read(state_0);
    let b = read(state_1);
    write(state_1, a + b);
    write(state_1, a + 1);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleStatesWriteBeforeRead) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  init { (0, 1) }
  config() { }
  next(state_0: u32, state_1: u32) {
    let a = read(state_0);
    write(state_1, a);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

TEST_F(IrConverterTest, ExplicitStateAccessMultipleBranchingLabeledReads) {
  constexpr std::string_view kModule = R"(#![feature(explicit_state_access)]
proc main {
  init { (0, true) }
  config() { }
  next(val: u32, switch: bool) {
    let switch_val = read(switch);
    if (switch_val) {
      let even = 'EvenRead:read(val);
      'EvenWrite:write(val, even + 1);
    } else {
      let odd = 'OddRead:read(val);
      'OddWrite:write(val, odd + 1);
    };
    write(switch, !switch_val);
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::string converted,
                           ConvertModuleForTest(kModule));
  ExpectIr(converted);
}

}  // namespace
}  // namespace xls::dslx
