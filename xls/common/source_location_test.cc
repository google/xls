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

#include "xls/common/source_location.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using ::testing::EndsWith;
using ::xabsl::SourceLocation;

TEST(SourceLocationTest, CopyConstructionWorks) {
  constexpr SourceLocation location = XABSL_LOC;

  EXPECT_EQ(location.line(), __LINE__ - 2);
  EXPECT_THAT(location.file_name(), EndsWith("source_location_test.cc"));
}

TEST(SourceLocationTest, CopyAssignmentWorks) {
  SourceLocation location;
  location = XABSL_LOC;

  EXPECT_EQ(location.line(), __LINE__ - 2);
  EXPECT_THAT(location.file_name(), EndsWith("source_location_test.cc"));
}

SourceLocation Echo(const SourceLocation& location) { return location; }

TEST(SourceLocationTest, ExpectedUsageWorks) {
  SourceLocation location = Echo(XABSL_LOC);

  EXPECT_EQ(location.line(), __LINE__ - 2);
  EXPECT_THAT(location.file_name(), EndsWith("source_location_test.cc"));
}

#if ABSL_HAVE_SOURCE_LOCATION_CURRENT

TEST(SourceLocationTest, CurrentWorks) {
  constexpr SourceLocation location = SourceLocation::current();

  EXPECT_EQ(location.line(), __LINE__ - 2);
  EXPECT_THAT(location.file_name(), EndsWith("source_location_test.cc"));
}

SourceLocation FuncWithDefaultParam(
    SourceLocation loc = SourceLocation::current()) {
  return loc;
}

TEST(SourceLocationTest, CurrentWorksAsDefaultParam) {
  SourceLocation location = FuncWithDefaultParam();

  EXPECT_EQ(location.line(), __LINE__ - 2);
  EXPECT_THAT(location.file_name(), EndsWith("source_location_test.cc"));
}

SourceLocation FuncWithDefaultMacro(
    SourceLocation loc XABSL_LOC_CURRENT_DEFAULT_ARG) {
  return loc;
}

TEST(SourceLocationTest, CurrentWorksAsDefaultParamMacro) {
  SourceLocation location = FuncWithDefaultMacro();

  EXPECT_EQ(location.line(), __LINE__ - 2);
  EXPECT_THAT(location.file_name(), EndsWith("source_location_test.cc"));
}

template <typename T>
bool TryPassLineAndFile(decltype(T::current(0, ""))*) {
  return true;
}
template <typename T>
bool TryPassLineAndFile(decltype(T::current({}, 0, ""))*) {
  return true;
}
template <typename T>
bool TryPassLineAndFile(decltype(T::current(typename T::Tag{}, 0, ""))*) {
  return true;
}
template <typename T>
bool TryPassLineAndFile(...) {
  return false;
}

TEST(SourceLocationTest, CantPassLineAndFile) {
  EXPECT_FALSE(TryPassLineAndFile<xabsl::SourceLocation>(nullptr));
}

#endif  // ABSL_HAVE_SOURCE_LOCATION_CURRENT

}  // namespace
