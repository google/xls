// Copyright 2020 Google LLC
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

#include <readline/readline.h>

#include <unistd.h>

#include <cstdio>
#include <string>

#include "absl/strings/substitute.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::Eq;
using ::testing::Ne;
using ::testing::StrEq;
using ::testing::Test;

namespace {

std::string HomeDirectory() {
  const char* home = getenv("HOME");
  return home == nullptr ? "" : home;
}

}  // namespace

class ReadlineTest : public Test {
 protected:
  void TestTilde(const char* input, const std::string& expected) {
    auto actual = tilde_expand(const_cast<char*>(input));
    EXPECT_THAT(actual, Eq(expected));
    free(actual);
  }
};

TEST_F(ReadlineTest, ReadLine) {
  // Create a new input for the purpose of this test
  int pipefds[2];
  ASSERT_THAT(pipe2(pipefds, 0), Eq(0));

  FILE *in = fdopen(pipefds[0], "rb");
  ASSERT_THAT(in, Ne(nullptr));
  rl_instream = in;

  ASSERT_THAT(write(pipefds[1], "foo\\n", 4), Eq(4));
  ASSERT_THAT(close(pipefds[1]), Eq(0));

  // Test 1: Read one line.
  {
    char* s = readline("test> ");
    EXPECT_THAT(s, Ne(nullptr));
    EXPECT_THAT(s, StrEq("foo"));
    free(s);
  }

  // Test 2: Since we closed the stream after one line, no more lines are read.
  {
    char* s = readline("test> ");
    EXPECT_THAT(s, Eq(nullptr));
  }

  ASSERT_THAT(fclose(in), Eq(0));
}

TEST_F(ReadlineTest, TildeMinimal) {
  // tilde_expand always appends a /.
  TestTilde("~", absl::Substitute("$0/", HomeDirectory()));
}

TEST_F(ReadlineTest, TildeSlash) {
  TestTilde("~/", absl::Substitute("$0/", HomeDirectory()));
}

TEST_F(ReadlineTest, TildeSlashPath) {
  TestTilde("~/foo/bar",
            absl::Substitute("$0/foo/bar", HomeDirectory()));
}

TEST_F(ReadlineTest, TildeUnchanged) {
  TestTilde("foo/bar", "foo/bar");
}

TEST_F(ReadlineTest, TildeInvalid) {
  auto val = "~invalid-user-that-does-not-exist/";
  TestTilde(val, val);
}
