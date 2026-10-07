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

#include "xls/dslx/fmt/type_annotation_simplifier.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

class TypeAnnotationSimplifierTest : public ::testing::Test {
 public:
  void SimplifyAndExpectEq(
      std::string_view original, std::string_view want,
      absl::flat_hash_map<std::filesystem::path, std::string> extra_files =
          {}) {
    const std::filesystem::path path = "test.x";
    auto make_import_data = [&]() {
      if (extra_files.empty()) {
        return CreateImportDataForTest();
      }
      return CreateImportDataForTest(
          std::make_unique<FakeFilesystem>(extra_files, ""));
    };

    ImportData import_data = make_import_data();
    std::vector<CommentData> comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Module> module,
        ParseModule(original, path.c_str(), "test", import_data.file_table(),
                    &comments_vec));
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule tm,
        ParseAndTypecheck(original, path.c_str(), "test", &import_data));
    Comments comments = Comments::Create(comments_vec);
    DocArena arena(import_data.file_table());
    std::unique_ptr<Formatter> simplifier =
        CreateTypeAnnotationSimplifier(comments, arena, *tm.type_info);
    UniformContentFilesystem vfs(original);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string got,
        AutoFmt(vfs, *module, *simplifier, std::string(original), 100));
    EXPECT_EQ(got, want);

    // Verify that running standard AutoFmt on the simplified output is
    // idempotent.
    UniformContentFilesystem got_vfs(got);
    FileTable got_file_table;
    std::vector<CommentData> got_comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Module> got_module,
                             ParseModule(got, path.c_str(), "test",
                                         got_file_table, &got_comments_vec));
    Comments got_comments = Comments::Create(got_comments_vec);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string reformatted,
        AutoFmt(got_vfs, *got_module, got_comments, got, 100));
    EXPECT_EQ(reformatted, got);

    ImportData got_import_data = make_import_data();
    XLS_ASSERT_OK(ParseAndTypecheck(got, path.c_str(), "test", &got_import_data)
                      .status());
  }
};

TEST_F(TypeAnnotationSimplifierTest, EnumMembersSimpleAndConcat) {
  constexpr std::string_view kOriginal = R"(pub enum Mode : u2 {
    Idle = u2:0,
    Read = u2:1,
    Write = u2:2,
}

pub enum CombinedCode : u13 {
    Zero = u8:0 ++ u5:0,
    First = u8:1 ++ u5:2,
}
)";
  constexpr std::string_view kWant = R"(pub enum Mode : u2 {
    Idle = 0,
    Read = 1,
    Write = 2,
}

pub enum CombinedCode : u13 {
    Zero = u8:0 ++ u5:0,
    First = u8:1 ++ u5:2,
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, ConstantDeclarationsTypedAndUntyped) {
  constexpr std::string_view kOriginal = R"(pub type ItemIndex = u7;

pub const WORD_WIDTH = u32:32;
pub const DEFAULT_INDEX = ItemIndex:0;
pub const TYPED_CONST: u32 = u32:32;
pub const TYPED_ALIAS: ItemIndex = ItemIndex:0;
)";
  constexpr std::string_view kWant = R"(pub type ItemIndex = u7;

pub const WORD_WIDTH = u32:32;
pub const DEFAULT_INDEX = ItemIndex:0;
pub const TYPED_CONST: u32 = 32;
pub const TYPED_ALIAS: ItemIndex = 0;
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, StructAndSplatStructInstances) {
  constexpr std::string_view kOriginal = R"(pub type TagId = u4;
pub const ADDR_W = u32:8;

pub struct ConfigParams {
    num_units: u32,
    num_entries: u32,
}

pub const DEFAULT_PARAMS = ConfigParams {
    num_units: u32:4,
    num_entries: u32:128,
};

pub struct EntryConfig {
    enable: bool,
    tag: TagId,
    address: uN[ADDR_W],
}

pub fn make_config(base: EntryConfig) -> EntryConfig {
    let default_cfg = EntryConfig {
        enable: false,
        tag: TagId:0,
        address: uN[ADDR_W]:0,
    };
    EntryConfig { tag: TagId:1, ..default_cfg }
}

pub struct SizedPair<A: u32, B: u32 = {u32:16}> {
    first: uN[A],
    second: uN[B],
}

pub type ConcretePair = SizedPair<u32:8, u32:16>;

pub fn make_pairs() -> ConcretePair {
    let inferred_all = SizedPair { first: u8:1, second: u16:2 };
    let inferred_partial = SizedPair<u32:8> { first: u8:1, second: u16:2 };
    let explicit_all = SizedPair<u32:8, u32:16> { first: u8:1, second: u16:2 };
    let via_alias = ConcretePair { first: u8:1, second: u16:2 };
    let splat_inferred = SizedPair { first: u8:3, ..inferred_all };
    let splat_explicit = SizedPair<u32:8, u32:16> { first: u8:4, ..explicit_all };
    SizedPair<u32:8, u32:16> {
        first: splat_inferred.first + splat_explicit.first + via_alias.first,
        ..inferred_partial
    }
}
)";
  constexpr std::string_view kWant = R"(pub type TagId = u4;

pub const ADDR_W = u32:8;

pub struct ConfigParams { num_units: u32, num_entries: u32 }

pub const DEFAULT_PARAMS = ConfigParams { num_units: 4, num_entries: 128 };

pub struct EntryConfig { enable: bool, tag: TagId, address: uN[ADDR_W] }

pub fn make_config(base: EntryConfig) -> EntryConfig {
    let default_cfg = EntryConfig { enable: false, tag: 0, address: 0 };
    EntryConfig { tag: 1, ..default_cfg }
}

pub struct SizedPair<A: u32, B: u32 = {16}> { first: uN[A], second: uN[B] }

pub type ConcretePair = SizedPair<8, 16>;

pub fn make_pairs() -> ConcretePair {
    let inferred_all = SizedPair { first: u8:1, second: u16:2 };
    let inferred_partial = SizedPair<8> { first: u8:1, second: u16:2 };
    let explicit_all = SizedPair<8, 16> { first: 1, second: 2 };
    let via_alias = ConcretePair { first: 1, second: 2 };
    let splat_inferred = SizedPair { first: u8:3, ..inferred_all };
    let splat_explicit = SizedPair<8, 16> { first: 4, ..explicit_all };
    SizedPair<8, 16> {
        first: splat_inferred.first + splat_explicit.first + via_alias.first,
        ..inferred_partial
    }
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, TypeAnnotationDimensions) {
  constexpr std::string_view kOriginal = R"(pub type SmallBits = bits[u32:2];
pub type WordArray = u32[u32:4];
)";
  constexpr std::string_view kWant = R"(pub type SmallBits = bits[2];
pub type WordArray = u32[4];
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, ShiftAndBinaryExpressions) {
  constexpr std::string_view kOriginal = R"(pub const COUNT_LOG2 = u32:4;
pub const MAX_COUNT = (u32:1 << COUNT_LOG2) - u32:1;
pub const ADDR_W = u32:8;
pub const OFFSET_ADDR_W = ADDR_W - u32:1;
pub const LITERAL_SUM = u32:1 + u32:2;
pub const TYPED_SUM: u32 = u32:1 + u32:2;
pub const SHIFT_BOTH_LITERALS = u32:1 << u32:2;
pub const SHIFT_IN_TYPED_DECL: u32 = u32:1 << u32:2;
pub const SHIFT_NON_LITERAL_LHS = ADDR_W << u32:2;

const_assert!((u32:1 << (ADDR_W / u32:2)) <= u32:32);
)";
  constexpr std::string_view kWant = R"(pub const COUNT_LOG2 = u32:4;
pub const MAX_COUNT = (u32:1 << COUNT_LOG2) - 1;
pub const ADDR_W = u32:8;
pub const OFFSET_ADDR_W = ADDR_W - 1;
pub const LITERAL_SUM = u32:1 + 2;
pub const TYPED_SUM: u32 = 1 + 2;
pub const SHIFT_BOTH_LITERALS = u32:1 << 2;
pub const SHIFT_IN_TYPED_DECL: u32 = u32:1 << 2;
pub const SHIFT_NON_LITERAL_LHS = ADDR_W << 2;

const_assert!((u32:1 << (ADDR_W / 2)) <= 32);
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, FunctionInvocations) {
  constexpr std::string_view kOriginal = R"(import std;

const MEBIBYTES_IN_BYTES = std::upow(u32:2, u32:20);
const LOG2_VAL = std::clog2(MEBIBYTES_IN_BYTES);
const EXTENDED = signex(u8:0xff, s32:0);

fn add_u32(a: u32, b: u32) -> u32 {
    a + b
}

fn add_parametric<N: u32>(a: uN[N], b: uN[N]) -> uN[N] {
    a + b
}

fn concat_parametric<M: u32, N: u32>(a: uN[M], b: uN[N]) -> uN[M + N] {
    a ++ b
}

const CALL_CONCRETE = add_u32(u32:10, u32:20);
const CALL_PARAMETRIC_LITS = add_parametric(u32:10, u32:20);
const CALL_PARAMETRIC_VAR = add_parametric(CALL_CONCRETE, u32:20);
const CALL_PARAMETRIC_EXPLICIT = add_parametric<u32:32>(u32:10, u32:20);
const CALL_PARTIAL_EXPLICIT = concat_parametric<u32:8>(u8:1, u16:2);
const CALL_ALL_EXPLICIT = concat_parametric<u32:8, u32:16>(u8:1, u16:2);
)";
  constexpr std::string_view kWant = R"(import std;

const MEBIBYTES_IN_BYTES = std::upow(u32:2, u32:20);
const LOG2_VAL = std::clog2(MEBIBYTES_IN_BYTES);
const EXTENDED = signex(u8:0xff, s32:0);

fn add_u32(a: u32, b: u32) -> u32 { a + b }

fn add_parametric<N: u32>(a: uN[N], b: uN[N]) -> uN[N] { a + b }

fn concat_parametric<M: u32, N: u32>(a: uN[M], b: uN[N]) -> uN[M + N] { a ++ b }

const CALL_CONCRETE = add_u32(10, 20);
const CALL_PARAMETRIC_LITS = add_parametric(u32:10, u32:20);
const CALL_PARAMETRIC_VAR = add_parametric(CALL_CONCRETE, u32:20);
const CALL_PARAMETRIC_EXPLICIT = add_parametric<32>(10, 20);
const CALL_PARTIAL_EXPLICIT = concat_parametric<8>(u8:1, u16:2);
const CALL_ALL_EXPLICIT = concat_parametric<8, 16>(1, 2);
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, ArrayLiterals) {
  constexpr std::string_view kOriginal =
      R"(type WordArray = u32[3];

const ARR_WHOLE_TYPE = u32[3]:[1, 2, 3];
const ARR_ELEM_TYPES = [u32:1, u32:2, u32:3];
const ARR_BOTH_TYPES = u32[3]:[u32:1, u32:2, u32:3];
const ARR_TYPED_LHS: u32[3] = u32[3]:[u32:1, u32:2, u32:3];
const ARR_ALIAS = WordArray:[u32:1, u32:2, u32:3];
const ARR_ALIAS_TYPED: WordArray = WordArray:[u32:1, u32:2, u32:3];
const ARR_2D = u32[2][2]:[u32[2]:[u32:1, u32:2], u32[2]:[u32:3, u32:4]];
const ARR_2D_TYPED: u32[2][2] = u32[2][2]:[u32[2]:[u32:1, u32:2], u32[2]:[u32:3, u32:4]];
const ARR_ELLIPSIS = u32[3]:[u32:0, ...];
const ARR_ELLIPSIS_TYPED: u32[3] = u32[3]:[u32:0, ...];
)";
  constexpr std::string_view kWant = R"(type WordArray = u32[3];

const ARR_WHOLE_TYPE = [u32:1, 2, 3];
const ARR_ELEM_TYPES = [u32:1, 2, 3];
const ARR_BOTH_TYPES = [u32:1, 2, 3];
const ARR_TYPED_LHS: u32[3] = [1, 2, 3];
const ARR_ALIAS = WordArray:[1, 2, 3];
const ARR_ALIAS_TYPED: WordArray = [1, 2, 3];
const ARR_2D = [[u32:1, 2], [3, 4]];
const ARR_2D_TYPED: u32[2][2] = [[1, 2], [3, 4]];
const ARR_ELLIPSIS = u32[3]:[0, ...];
const ARR_ELLIPSIS_TYPED: u32[3] = u32[3]:[0, ...];
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, TuplesAndConcatenations) {
  constexpr std::string_view kOriginal = R"(const TUPLE = (u32:1, u16:2);
const TUPLE_WITH_ARRAY = (u32[2]:[u32:1, u32:2], u8:3);
const CONCAT_BITS = u8:1 ++ u8:2;
const CONCAT_ARRAYS = u32[2]:[u32:1, u32:2] ++ u32[2]:[u32:3, u32:4];
)";
  constexpr std::string_view kWant = R"(const TUPLE = (u32:1, u16:2);
const TUPLE_WITH_ARRAY = ([u32:1, 2], u8:3);
const CONCAT_BITS = u8:1 ++ u8:2;
const CONCAT_ARRAYS = [u32:1, 2] ++ [u32:3, 4];
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, ControlFlowAndIndexing) {
  constexpr std::string_view kOriginal = R"(fn compute(x: u32) -> u32 {
    let untyped_local = u32:5;
    let typed_local: u32 = u32:6;
    let arr = u32[3]:[u32:10, u32:20, u32:30];
    let elem = arr[u32:0];
    let sliced = x[u32:0+:u16] ++ x[s32:16:s32:32];
    let loop_sum = for (i, acc): (u32, u32) in u32:0..u32:4 {
        acc + i + u32:1
    }(u32:0);
    let branched = if x == u32:0 { u32:100 } else { u32:200 };
    match x {
        u32:0 => untyped_local + typed_local + elem + (sliced as u32),
        _ => loop_sum + branched + u32:1,
    }
}
)";
  constexpr std::string_view kWant = R"(fn compute(x: u32) -> u32 {
    let untyped_local = u32:5;
    let typed_local: u32 = 6;
    let arr = [u32:10, 20, 30];
    let elem = arr[0];
    let sliced = x[0+:u16] ++ x[16:32];
    let loop_sum = for (i, acc): (u32, u32) in 0..4 {
        acc + i + 1
    }(0);
    let branched = if x == 0 { u32:100 } else { 200 };
    match x {
        0 => untyped_local + typed_local + elem + (sliced as u32),
        _ => loop_sum + branched + 1,
    }
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, MultilineStructAndLetWithArrayFields) {
  constexpr std::string_view kOriginal = R"(pub struct ItemConfig {
    valid: bool,
    has_payload: bool,
    payload: u32,
}

pub struct GroupConfig {
    items: ItemConfig[4],
    values: u32[4],
}

pub fn make_group_config() -> GroupConfig {
    let very_long_local_array_variable_name_that_nears_column_limit_with_type_leader =
        ItemConfig[4]:[
            ItemConfig { valid: true, has_payload: true, payload: u32:0x11111111 },
            ItemConfig { valid: true, has_payload: true, payload: u32:0x22222222 },
            ItemConfig { valid: false, has_payload: false, payload: u32:0 },
            ItemConfig { valid: false, has_payload: false, payload: u32:0 },
        ];
    GroupConfig {
        items: ItemConfig[4]:[
            ItemConfig { valid: true, has_payload: true, payload: u32:0x11111111 },
            ItemConfig { valid: true, has_payload: true, payload: u32:0x22222222 },
            very_long_local_array_variable_name_that_nears_column_limit_with_type_leader[u32:2],
            very_long_local_array_variable_name_that_nears_column_limit_with_type_leader[u32:3],
        ],
        values: u32[4]:[u32:0xffffffff, u32:0x11101111, u32:0x0001020a, u32:0x22222222],
    }
}
)";
  constexpr std::string_view kWant =
      R"(pub struct ItemConfig { valid: bool, has_payload: bool, payload: u32 }

pub struct GroupConfig { items: ItemConfig[4], values: u32[4] }

pub fn make_group_config() -> GroupConfig {
    let very_long_local_array_variable_name_that_nears_column_limit_with_type_leader = [
        ItemConfig { valid: true, has_payload: true, payload: 0x11111111 },
        ItemConfig { valid: true, has_payload: true, payload: 0x22222222 },
        ItemConfig { valid: false, has_payload: false, payload: 0 },
        ItemConfig { valid: false, has_payload: false, payload: 0 },
    ];
    GroupConfig {
        items:
            [
                ItemConfig { valid: true, has_payload: true, payload: 0x11111111 },
                ItemConfig { valid: true, has_payload: true, payload: 0x22222222 },
                very_long_local_array_variable_name_that_nears_column_limit_with_type_leader[2],
                very_long_local_array_variable_name_that_nears_column_limit_with_type_leader[3],
            ],
        values: [0xffffffff, 0x11101111, 0x0001020a, 0x22222222],
    }
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, ImportedEntitiesViaColonRef) {
  absl::flat_hash_map<std::filesystem::path, std::string> extra_files = {
      {"other_mod.x", R"(pub type TagId = u4;

pub struct EntryConfig {
    enable: bool,
    tag: TagId,
    address: u8,
}

impl EntryConfig {
    pub fn with_tag(tag: TagId) -> Self {
        EntryConfig { enable: true, tag, address: u8:0 }
    }
}

pub struct SizedPair<A: u32 = {8}, B: u32 = {16}> {
    first: uN[A],
    second: uN[B],
}

impl SizedPair {
    pub fn make(first: uN[A], second: uN[B]) -> Self {
        SizedPair { first, second }
    }

    pub fn from_scaled<W: u32>(val: uN[W]) -> Self {
        SizedPair { first: val as uN[A], second: val as uN[B] }
    }
}

pub type ConcretePair = SizedPair<8, 16>;

pub fn add_u32(a: u32, b: u32) -> u32 { a + b }

pub fn add_parametric<N: u32>(a: uN[N], b: uN[N]) -> uN[N] { a + b }

pub fn concat_parametric<M: u32, N: u32>(a: uN[M], b: uN[N]) -> uN[M + N] {
    a ++ b
}
)"}};

  constexpr std::string_view kOriginal = R"(import other_mod;

pub fn use_imported_entities() -> other_mod::ConcretePair {
    let cfg = other_mod::EntryConfig {
        enable: false,
        tag: other_mod::TagId:0,
        address: u8:0,
    };
    let cfg2 = other_mod::EntryConfig { tag: other_mod::TagId:1, ..cfg };
    let cfg3 = other_mod::EntryConfig::with_tag(other_mod::TagId:2);

    let sum1 = other_mod::add_u32(u32:10, u32:20);
    let sum2 = other_mod::add_parametric(u32:10, u32:20);
    let sum3 = other_mod::add_parametric<u32:32>(u32:10, u32:20);
    let cat1 = other_mod::concat_parametric<u32:8>(u8:1, u16:2);
    let cat2 = other_mod::concat_parametric<u32:8, u32:16>(u8:1, u16:2);

    let pair_inferred = other_mod::SizedPair { first: u8:1, second: u16:2 };
    let pair_partial = other_mod::SizedPair<u32:8> { first: u8:1, second: u16:2 };
    let pair_explicit = other_mod::SizedPair<u32:8, u32:16> { first: u8:1, second: u16:2 };
    let pair_alias = other_mod::ConcretePair { first: u8:1, second: u16:2 };

    let fn_inferred = other_mod::SizedPair::make(u8:1, u16:2);
    let fn_partial = other_mod::SizedPair<u32:8>::make(u8:1, u16:2);
    let fn_explicit = other_mod::SizedPair<u32:8, u32:16>::make(u8:1, u16:2);
    let fn_alias = other_mod::ConcretePair::make(u8:1, u16:2);
    let fn_scaled_inferred = other_mod::ConcretePair::from_scaled(u8:3);
    let fn_scaled_explicit = other_mod::ConcretePair::from_scaled<u32:8>(u8:4);

    other_mod::SizedPair<u32:8, u32:16> {
        first: cfg2.address + cfg3.address + pair_inferred.first + pair_partial.first +
               pair_explicit.first + pair_alias.first + fn_inferred.first + fn_partial.first +
               fn_explicit.first + fn_alias.first + fn_scaled_inferred.first +
               fn_scaled_explicit.first + ((sum1 + sum2 + sum3) as u8),
        second: (cat1 + cat2)[u32:0+:u16],
    }
}

pub fn uninstantiated_imported_caller<K: u32>(x: uN[K]) -> other_mod::ConcretePair {
    let a = other_mod::add_u32(u32:1, u32:2);
    let b = other_mod::add_parametric(u32:3, u32:4);
    let c = other_mod::add_parametric<u32:32>(u32:5, u32:6);
    let p1 = other_mod::SizedPair::make(u8:1, u16:2);
    let p2 = other_mod::SizedPair<u32:8, u32:16>::make(u8:1, u16:2);
    let p3 = other_mod::ConcretePair::make(u8:1, u16:2);
    other_mod::ConcretePair {
        first: p1.first + p2.first + p3.first + ((a + b + c) as u8),
        second: x as u16,
    }
}
)";

  constexpr std::string_view kWant = R"(import other_mod;

pub fn use_imported_entities() -> other_mod::ConcretePair {
    let cfg = other_mod::EntryConfig { enable: false, tag: 0, address: 0 };
    let cfg2 = other_mod::EntryConfig { tag: 1, ..cfg };
    let cfg3 = other_mod::EntryConfig::with_tag(2);

    let sum1 = other_mod::add_u32(10, 20);
    let sum2 = other_mod::add_parametric(u32:10, u32:20);
    let sum3 = other_mod::add_parametric<32>(10, 20);
    let cat1 = other_mod::concat_parametric<8>(u8:1, u16:2);
    let cat2 = other_mod::concat_parametric<8, 16>(1, 2);

    let pair_inferred = other_mod::SizedPair { first: u8:1, second: u16:2 };
    let pair_partial = other_mod::SizedPair<8> { first: u8:1, second: u16:2 };
    let pair_explicit = other_mod::SizedPair<8, 16> { first: 1, second: 2 };
    let pair_alias = other_mod::ConcretePair { first: 1, second: 2 };

    let fn_inferred = other_mod::SizedPair::make(u8:1, u16:2);
    let fn_partial = other_mod::SizedPair<8>::make(u8:1, u16:2);
    let fn_explicit = other_mod::SizedPair<8, 16>::make(1, 2);
    let fn_alias = other_mod::ConcretePair::make(1, 2);
    let fn_scaled_inferred = other_mod::ConcretePair::from_scaled(u8:3);
    let fn_scaled_explicit = other_mod::ConcretePair::from_scaled<8>(4);

    other_mod::SizedPair<8, 16> {
        first:
            cfg2.address + cfg3.address + pair_inferred.first + pair_partial.first +
            pair_explicit.first + pair_alias.first + fn_inferred.first + fn_partial.first +
            fn_explicit.first + fn_alias.first + fn_scaled_inferred.first + fn_scaled_explicit.first +
            ((sum1 + sum2 + sum3) as u8),
        second: (cat1 + cat2)[0+:u16],
    }
}

pub fn uninstantiated_imported_caller<K: u32>(x: uN[K]) -> other_mod::ConcretePair {
    let a = other_mod::add_u32(1, 2);
    let b = other_mod::add_parametric(u32:3, u32:4);
    let c = other_mod::add_parametric<32>(5, 6);
    let p1 = other_mod::SizedPair::make(u8:1, u16:2);
    let p2 = other_mod::SizedPair<8, 16>::make(1, 2);
    let p3 = other_mod::ConcretePair::make(1, 2);
    other_mod::ConcretePair {
        first: p1.first + p2.first + p3.first + ((a + b + c) as u8),
        second: x as u16,
    }
}
)";
  SimplifyAndExpectEq(kOriginal, kWant, extra_files);
}

TEST_F(TypeAnnotationSimplifierTest, GenericTypeColonRefsAndInstances) {
  constexpr std::string_view kOriginal = R"(#![feature(generics)]

pub struct BoxedWord {
    value: u32,
}

impl BoxedWord {
    pub fn from_u32(v: u32) -> Self {
        BoxedWord { value: v }
    }

    pub fn add_u32(self, delta: u32) -> Self {
        BoxedWord { value: self.value + delta }
    }

    pub fn add_bits<W: u32>(self, delta: uN[W]) -> Self {
        BoxedWord { value: self.value + (delta as u32) }
    }
}

pub struct SizedBox<N: u32> {
    value: uN[N],
}

impl SizedBox {
    pub fn from_val(v: uN[N]) -> Self {
        SizedBox { value: v }
    }

    pub fn from_bits<W: u32>(v: uN[W]) -> Self {
        SizedBox { value: v as uN[N] }
    }
}

fn build_generic_word<T: type>() -> T {
    let a = T::from_u32(u32:10);
    let b = a.add_u32(u32:40);
    let c = b.add_bits(u8:50);
    let d = c.add_bits<u32:8>(u8:60);
    T { value: d.value + u32:1 }
}

fn build_generic_sized<T: type>() -> T {
    let a = T::from_val(u16:10);
    let b = T::from_bits(u8:20);
    let c = T::from_bits<u32:8>(u8:30);
    T { value: a.value + b.value + c.value + u16:1 }
}

pub fn instantiate_generics() -> (BoxedWord, SizedBox<u32:16>) {
    (build_generic_word<BoxedWord>(), build_generic_sized<SizedBox<u32:16>>())
}
)";

  constexpr std::string_view kWant = R"(#![feature(generics)]

pub struct BoxedWord { value: u32 }

impl BoxedWord {
    pub fn from_u32(v: u32) -> Self { BoxedWord { value: v } }

    pub fn add_u32(self, delta: u32) -> Self { BoxedWord { value: self.value + delta } }

    pub fn add_bits<W: u32>(self, delta: uN[W]) -> Self {
        BoxedWord { value: self.value + (delta as u32) }
    }
}

pub struct SizedBox<N: u32> { value: uN[N] }

impl SizedBox {
    pub fn from_val(v: uN[N]) -> Self { SizedBox { value: v } }

    pub fn from_bits<W: u32>(v: uN[W]) -> Self { SizedBox { value: v as uN[N] } }
}

fn build_generic_word<T: type>() -> T {
    let a = T::from_u32(10);
    let b = a.add_u32(40);
    let c = b.add_bits(u8:50);
    let d = c.add_bits<8>(60);
    T { value: d.value + 1 }
}

fn build_generic_sized<T: type>() -> T {
    let a = T::from_val(10);
    let b = T::from_bits(u8:20);
    let c = T::from_bits<8>(30);
    T { value: a.value + b.value + c.value + 1 }
}

pub fn instantiate_generics() -> (BoxedWord, SizedBox<16>) {
    (build_generic_word<BoxedWord>(), build_generic_sized<SizedBox<16>>())
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, LegacyProcs) {
  constexpr std::string_view kOriginal = R"(proc Accumulator {
    in_ch: chan<u32> in;
    out_ch: chan<u32> out;

    const STEP: u32 = u32:2;
    const LIMIT = u32:100;

    config(in_ch: chan<u32> in, out_ch: chan<u32> out, bias: u32) {
        (in_ch, out_ch)
    }

    init { u32:0 }

    next(state: u32) {
        let (tok, item, valid) = recv_non_blocking(join(), in_ch, u32:0);
        let next_state = if valid { state + item + STEP } else { state + u32:1 };
        let tok = send_if(tok, out_ch, next_state >= LIMIT, u32:0);
        let tok = send(tok, out_ch, u32:42);
        if next_state >= LIMIT { u32:0 } else { next_state }
    }
}

proc ScaledWorker<W: u32 = {u32:16}> {
    out_ch: chan<uN[W]> out;

    config(out_ch: chan<uN[W]> out, offset: uN[W]) { (out_ch,) }

    init { uN[W]:0 }

    next(state: uN[W]) {
        let tok = send(join(), out_ch, state + uN[W]:1);
        state + uN[W]:1
    }
}

proc Main {
    in_s: chan<u32> out;
    out_r: chan<u32> in;
    worker_r: chan<u16>[2] in;

    config() {
        let (in_s, in_r) = chan<u32, u32:4>("acc_in");
        let (out_s, out_r) = chan<u32>("acc_out");
        let (worker_s, worker_r) = chan<u16>[u32:2]("worker_out");
        spawn Accumulator(in_r, out_s, u32:10);
        spawn ScaledWorker(worker_s[u32:0], u16:20);
        spawn ScaledWorker<u32:16>(worker_s[u32:1], u16:30);
        (in_s, out_r, worker_r)
    }

    init { () }

    next(state: ()) { () }
}
)";

  constexpr std::string_view kWant = R"(proc Accumulator {
    in_ch: chan<u32> in;
    out_ch: chan<u32> out;
    const STEP: u32 = 2;
    const LIMIT = u32:100;

    config(in_ch: chan<u32> in, out_ch: chan<u32> out, bias: u32) { (in_ch, out_ch) }

    init { 0 }

    next(state: u32) {
        let (tok, item, valid) = recv_non_blocking(join(), in_ch, 0);
        let next_state = if valid { state + item + STEP } else { state + 1 };
        let tok = send_if(tok, out_ch, next_state >= LIMIT, 0);
        let tok = send(tok, out_ch, 42);
        if next_state >= LIMIT { 0 } else { next_state }
    }
}

proc ScaledWorker<W: u32 = {16}> {
    out_ch: chan<uN[W]> out;

    config(out_ch: chan<uN[W]> out, offset: uN[W]) { (out_ch,) }

    init { 0 }

    next(state: uN[W]) {
        let tok = send(join(), out_ch, state + 1);
        state + 1
    }
}

proc Main {
    in_s: chan<u32> out;
    out_r: chan<u32> in;
    worker_r: chan<u16>[2] in;

    config() {
        let (in_s, in_r) = chan<u32, 4>("acc_in");
        let (out_s, out_r) = chan<u32>("acc_out");
        let (worker_s, worker_r) = chan<u16>[2]("worker_out");
        spawn Accumulator(in_r, out_s, 10);
        spawn ScaledWorker(worker_s[0], u16:20);
        spawn ScaledWorker<16>(worker_s[1], 30);
        (in_s, out_r, worker_r)
    }

    init { () }

    next(state: ()) { () }
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

TEST_F(TypeAnnotationSimplifierTest, ImplStyleProcs) {
  constexpr std::string_view kOriginal = R"(#![feature(explicit_state_access)]
#![feature(generics)]

proc Counter {
    in_ch: chan<u32> in,
    out_ch: chan<u32> out,
    count: u32,
}

impl Counter {
    const STEP: u32 = u32:2;
    const LIMIT = u32:100;

    fn new(in_ch: chan<u32> in, out_ch: chan<u32> out, initial: u32) -> Self {
        let adjusted = initial + u32:1;
        Self { in_ch, out_ch, count: u32:0 }
    }

    fn next(self) {
        let cur = read(self.count);
        let (tok, delta) = recv_if(join(), self.in_ch, cur == u32:0, u32:1);
        let tok = send(tok, self.out_ch, u32:42);
        write(self.count, cur + delta + STEP);
        if cur >= LIMIT { write(self.count, u32:0); };
    }
}

proc ScaledProducer<W: u32 = {u32:16}> {
    out_ch: chan<uN[W]> out,
    state: uN[W],
}

impl ScaledProducer {
    fn new(out_ch: chan<uN[W]> out, step: uN[W]) -> Self {
        Self { out_ch, state: uN[W]:0 }
    }

    fn next(self) {
        let cur = read(self.state);
        send(join(), self.out_ch, cur + uN[W]:1);
        write(self.state, uN[W]:0);
    }
}

proc Main {
    in_s: chan<u32> out,
    out_r: chan<u32> in,
    w_r: chan<u16>[2] in,
}

impl Main {
    fn new() -> Self {
        let (in_s, in_r) = chan<u32, u32:4>("in_ch");
        let (out_s, out_r) = chan<u32>("out_ch");
        let (w_s, w_r) = chan<u16>[u32:2]("w_ch");
        Counter::new(in_r, out_s, u32:10).spawn();
        ScaledProducer::new(w_s[u32:0], u16:20).spawn();
        ScaledProducer<u32:16>::new(w_s[u32:1], u16:30).spawn();
        Self { in_s, out_r, w_r }
    }
}
)";

  constexpr std::string_view kWant = R"(#![feature(explicit_state_access)]
#![feature(generics)]

proc Counter {
    in_ch: chan<u32> in,
    out_ch: chan<u32> out,
    count: u32,
}

impl Counter {
    const STEP: u32 = 2;
    const LIMIT = u32:100;

    fn new(in_ch: chan<u32> in, out_ch: chan<u32> out, initial: u32) -> Self {
        let adjusted = initial + 1;
        Self { in_ch, out_ch, count: 0 }
    }

    fn next(self) {
        let cur = read(self.count);
        let (tok, delta) = recv_if(join(), self.in_ch, cur == 0, 1);
        let tok = send(tok, self.out_ch, 42);
        write(self.count, cur + delta + STEP);
        if cur >= LIMIT { write(self.count, 0); };
    }
}

proc ScaledProducer<W: u32 = {16}> {
    out_ch: chan<uN[W]> out,
    state: uN[W],
}

impl ScaledProducer {
    fn new(out_ch: chan<uN[W]> out, step: uN[W]) -> Self {
        Self { out_ch, state: 0 }
    }

    fn next(self) {
        let cur = read(self.state);
        send(join(), self.out_ch, cur + 1);
        write(self.state, 0);
    }
}

proc Main {
    in_s: chan<u32> out,
    out_r: chan<u32> in,
    w_r: chan<u16>[2] in,
}

impl Main {
    fn new() -> Self {
        let (in_s, in_r) = chan<u32, 4>("in_ch");
        let (out_s, out_r) = chan<u32>("out_ch");
        let (w_s, w_r) = chan<u16>[2]("w_ch");
        Counter::new(in_r, out_s, 10).spawn();
        ScaledProducer::new(w_s[0], u16:20).spawn();
        ScaledProducer<16>::new(w_s[1], 30).spawn();
        Self { in_s, out_r, w_r }
    }
}
)";
  SimplifyAndExpectEq(kOriginal, kWant);
}

}  // namespace
}  // namespace xls::dslx
