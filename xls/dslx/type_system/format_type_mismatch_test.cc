// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system/format_type_mismatch.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

// Macro definitions so we can use C style string concatenation where we can
// write/see the escapes more easily.
#define ANSI_RESET "\33[0m"
#define ANSI_RED "\33[31m"
#define ANSI_BOLD "\33[1m"
#define ANSI_UNBOLD "\33[22m"

TEST(FormatTypeMismatchTest, ElementInTuple) {
  auto t0 = TupleType::Create3(BitsType::MakeU8(),
                               std::make_unique<BitsType>(false, 16),
                               BitsType::MakeU32());
  auto t1 = TupleType::Create3(BitsType::MakeU8(),
                               std::make_unique<BitsType>(true, 16),
                               BitsType::MakeU32());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(
      got,
      ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                 " type:\n"     //
                 "   uN[16]\n"  //
                 "vs sN[16]\n" ANSI_BOLD "Overall" ANSI_UNBOLD
                 " type mismatch:\n"  //
      ANSI_RESET "   (uN[8], " ANSI_RED "uN[16]" ANSI_RESET
                 ", uN[32])\n"                                           //
                 "vs (uN[8], " ANSI_RED "sN[16]" ANSI_RESET ", uN[32])"  //
  );
}

TEST(FormatTypeMismatchTest, NestedTuple) {
  std::unique_ptr<TupleType> t0 = TupleType::Create2(
      TupleType::Create2(BitsType::MakeU8(), BitsType::MakeU1()),
      TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU1()));
  std::unique_ptr<TupleType> t1 = TupleType::Create2(
      TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU1()),
      TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU1()));

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            ANSI_RESET
            "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
            " type:\n"                                                        //
            "   uN[8]\n"                                                      //
            "vs uN[1]\n" ANSI_BOLD "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   ((" ANSI_RED "uN[8]" ANSI_RESET
            ", uN[1]), (uN[1], uN[1]))\n"
            "vs ((" ANSI_RED "uN[1]" ANSI_RESET ", uN[1]), (uN[1], uN[1]))");
}

TEST(FormatTypeMismatchTest, ElementTypeInArrayInTuple) {
  auto t0 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeU32(), TypeDim::CreateU32(4)));
  auto t1 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeS32(), TypeDim::CreateU32(4)));

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                       " type:\n"                                 //
                       "   uN[32]\n"                              //
                       "vs sN[32]\n" ANSI_BOLD                    //
                       "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   (uN[1], " ANSI_RED "uN[32]" ANSI_RESET
                       "[4])\n"                                           //
                       "vs (uN[1], " ANSI_RED "sN[32]" ANSI_RESET "[4])"  //
  );
}

TEST(FormatTypeMismatchTest, MismatchedArraySizeInTuple) {
  auto t0 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeU32(), TypeDim::CreateU32(4)));
  auto t1 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeU32(), TypeDim::CreateU32(2)));

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                       " type:\n"                                 //
                       "   uN[32][4]\n"                           //
                       "vs uN[32][2]\n" ANSI_BOLD                 //
                       "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   (uN[1], " ANSI_RED "uN[32][4]" ANSI_RESET
                       ")\n"                                              //
                       "vs (uN[1], " ANSI_RED "uN[32][2]" ANSI_RESET ")"  //
  );
}

TEST(FormatTypeMismatchTest, TotallyDifferentTuples) {
  auto t0 = TupleType::Create2(BitsType::MakeU8(), BitsType::MakeU32());
  auto t1 = TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU64());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            "Type mismatch:\n"
            "   (uN[8], uN[32])\n"
            "vs (uN[1], uN[64])");
}

TEST(FormatTypeMismatchTest, TuplesWithSharedPrefixDifferentLength) {
  auto t0 = TupleType::Create3(BitsType::MakeU1(), BitsType::MakeU8(),
                               BitsType::MakeU32());
  auto t1 = TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU8());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            "Tuple is missing elements:\n"
            "   uN[32] (index 2 of (uN[1], uN[8], uN[32]))\n"
            "Type mismatch:\n"
            "   (uN[1], uN[8], uN[32])\n"
            "vs (uN[1], uN[8])");

  XLS_ASSERT_OK_AND_ASSIGN(got, FormatTypeMismatch(*t1, *t0, file_table));
  EXPECT_EQ(got,
            "Tuple has extra elements:\n"
            "   uN[32] (index 2 of (uN[1], uN[8], uN[32]))\n"
            "Type mismatch:\n"
            "   (uN[1], uN[8])\n"
            "vs (uN[1], uN[8], uN[32])");
}

TEST(FormatTypeMismatchTest, ChannelTypeMismatch) {
  std::unique_ptr<ChannelType> ch0 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kIn);
  std::unique_ptr<ChannelType> ch1 =
      std::make_unique<ChannelType>(BitsType::MakeU32(), ChannelDirection::kIn);

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*ch0, *ch1, file_table));

  EXPECT_EQ(got,
            "Type mismatch:\n"
            "   chan(uN[8], dir=in)\n"
            "vs chan(uN[32], dir=in)");
}

TEST(FormatTypeMismatchTest, ChannelTypeDirectionMismatch) {
  std::unique_ptr<ChannelType> ch0 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kIn);
  std::unique_ptr<ChannelType> ch1 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kOut);

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*ch0, *ch1, file_table));

  EXPECT_EQ(got,
            "Type mismatch:\n"
            "   chan(uN[8], dir=in)\n"
            "vs chan(uN[8], dir=out)");
}

TEST(FormatTypeMismatchTest, TupleOfChannelTypesElementMismatch) {
  std::unique_ptr<ChannelType> ch0 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kIn);
  std::unique_ptr<ChannelType> ch1 = std::make_unique<ChannelType>(
      BitsType::MakeU32(), ChannelDirection::kOut);

  std::unique_ptr<TupleType> t0 = TupleType::Create3(
      ch0->CloneToUnique(), ch0->CloneToUnique(), ch1->CloneToUnique());
  std::unique_ptr<TupleType> t1 = TupleType::Create3(
      ch0->CloneToUnique(), ch1->CloneToUnique(), ch1->CloneToUnique());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(
      got,
      ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                 " type:\n"                  //
                 "   chan(uN[8], dir=in)\n"  //
                 "vs chan(uN[32], dir=out)\n" ANSI_BOLD "Overall" ANSI_UNBOLD
                 " type mismatch:\n"  //
      ANSI_RESET "   (chan(uN[8]), " ANSI_RED "chan(uN[8], dir=in)" ANSI_RESET
                 ", chan(uN[32]))\n"  //
                 "vs (chan(uN[8]), " ANSI_RED "chan(uN[32], dir=out)" ANSI_RESET
                 ", chan(uN[32]))"  //
  );
}

}  // namespace
}  // namespace xls::dslx
