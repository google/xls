// Copyright 2020 The XLS Authors
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

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/golden_files.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {

constexpr char kTestName[] = "ir_parser_round_trip_test";
constexpr char kTestdataPath[] = "xls/ir/testdata";

static std::string TestName() {
  // If we try to run the program it can't have the '/' in its name. Remove
  // them so this pattern works.
  std::string name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  absl::StrReplaceAll(std::vector{std::pair{"/", "_"}}, &name);
  return name;
}

static std::filesystem::path TestFilePath(std::string_view test_name) {
  return absl::StrFormat("%s/%s_%s.ir", kTestdataPath, kTestName, test_name);
}

static std::filesystem::path TestFileAbsPath(std::string_view test_name) {
  return GetXlsRunfilePath(TestFilePath(test_name)).value();
}

// Parses the given string as a function, dumps the IR and compares that the
// dumped string and input string are the same modulo whitespace.
static void ParseFunctionAndCheckDump(
    std::string_view test_name,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ParseFunctionAndCheckDump failed");
  Package p("my_package");
  std::filesystem::path abs_path = TestFileAbsPath(test_name);
  std::string ir_text = GetFileContents(abs_path).value();
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(ir_text, &p));
  ExpectEqualToGoldenFile(TestFilePath(test_name), function->DumpIr(), loc);
}

// Parses the given string as a package, dumps the IR and compares that the
// dumped string and input string are the same modulo whitespace.
static void ParsePackageAndCheckDump(
    std::string_view test_name,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ParsePackageAndCheckDump failed");
  std::filesystem::path abs_path = TestFileAbsPath(test_name);
  std::string ir_text = GetFileContents(abs_path).value();
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  ExpectEqualToGoldenFile(TestFilePath(test_name), package->DumpIr(), loc);
}

TEST(IrParserRoundTripTest, ParseBitsLiteral) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTokenLiteral) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseWideLiteral) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParsePosition) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNode) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseFunction) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseFunctionWithFFI) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseFunctionWithNewlineInFFI) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseULessThan) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseSLessThan) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTwoPlusTwo) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, CountedFor) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, CountedForInvariantArgs) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, CountedForBodyBitWidthSufficient0) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, CountedForBodyBitWidthZeroIteration) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, CountedForBodyBitWidthOneIteration) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseBitSlice) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseDynamicBitSlice) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseAfterAllEmpty) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseAfterAllMany) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseMinDelay) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArray) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseReverse) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayOfTuples) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNestedBitsArrayIndex) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNestedBitsArrayUpdate) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, DifferentWidthMultiplies) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseSingleEmptyPackage) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseSingleFunctionPackage) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseMultiFunctionPackage) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseMap) { ParsePackageAndCheckDump(TestName()); }

TEST(IrParserRoundTripTest, ParseBinarySel) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTernarySelectWithDefault) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseOneHotLsbPriority) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseOneHotMsbPriority) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseOneHotSelect) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParsePrioritySelect) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseParamReturn) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInvoke) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseAssert) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseAssertWithLabel) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTrace) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTraceWithVerbosity) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseCover) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseBitSliceUpdate) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseStatelessProc) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseSimpleProc) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseProcWithExplicitNext) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseProcWithPredicatedStateRead) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNewStyleProc) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNewStyleProcNoInterfaceChannels) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiatedProc) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiatedProcWithZeroArgs) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNewStyleProcWithChannelDefinitions) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNewStyleProcWithComplexChannelTypes) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseSimpleBlock) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseBlockWithRegister) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseBlockWithRegisterWithResetValue) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseBlockWithRegisterWithLoadEnable) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseBlockWithBlockInstantiation) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiationOfDegenerateBlock) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiationOfNoInputBlock) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiationOfNoOutputBlock) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiationWithChannel) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseInstantiationWithNoBypassChannel) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayIndex) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayIndexAssumedInBounds) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArraySlice) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayUpdate) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayUpdateAssumedInBounds) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayConcat0) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayConcat1) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayConcatMixedOperands) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTupleIndex) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseIdentity) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseUnsignedInequalities) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseSignedInequalities) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseArrayLiterals) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNestedArrayLiterals) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseTupleLiteral) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNestedTupleLiteral) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseNaryXor) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseExtendOps) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseDecode) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseEncode) {
  ParseFunctionAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, Gate) { ParseFunctionAndCheckDump(TestName()); }

TEST(IrParserRoundTripTest, ParseIIFunction) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, ParseIIProc) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, Ffi) { ParsePackageAndCheckDump(TestName()); }

TEST(IrParserRoundTripTest, ChannelPortMetadata) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, BlockProvenance) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, BlockSignature) {
  ParsePackageAndCheckDump(TestName());
}

TEST(IrParserRoundTripTest, BlockWithSvTypes) {
  ParsePackageAndCheckDump(TestName());
}

}  // namespace xls
