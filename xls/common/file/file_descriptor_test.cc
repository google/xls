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

#include "xls/common/file/file_descriptor.h"

#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

// Global state is necessary to test FileDescriptor; it takes a plain function
// pointer with no state as the parameterizable close function.
int recently_closed_fd = -1;

int MockClose(int fd) {
  if (fd != -1) {
    EXPECT_EQ(recently_closed_fd, -1);
    recently_closed_fd = fd;
  }
  return 0;
}

using MockFileDescriptor = BasicFileDescriptor<MockClose>;

class FileDescriptorTest : public ::testing::Test {
 public:
  FileDescriptorTest() { recently_closed_fd = -1; }
};

TEST_F(FileDescriptorTest, DefaultConstructorInitializesToMinusOne) {
  MockFileDescriptor fd;

  EXPECT_EQ(fd.get(), -1);
}

TEST_F(FileDescriptorTest, DescriptorIsNotClosedOnCreation) {
  MockFileDescriptor fd(5);

  EXPECT_EQ(recently_closed_fd, -1);
}

TEST_F(FileDescriptorTest, MovedDescriptorIsNotClosedOnMoveConstruct) {
  MockFileDescriptor fd_1(5);

  MockFileDescriptor fd_2(std::move(fd_1));

  EXPECT_EQ(recently_closed_fd, -1);
}

TEST_F(FileDescriptorTest, MovedDescriptorIsNotClosedOnMoveAssign) {
  MockFileDescriptor fd_1(5);
  MockFileDescriptor fd_2;

  fd_2 = std::move(fd_1);

  EXPECT_EQ(recently_closed_fd, -1);
}

TEST_F(FileDescriptorTest, OverwrittenDescriptorIsNotClosedOnMoveAssign) {
  MockFileDescriptor fd_1;
  MockFileDescriptor fd_2(5);

  fd_2 = std::move(fd_1);

  EXPECT_EQ(recently_closed_fd, 5);
}

TEST_F(FileDescriptorTest, FileDescriptorIsClosedAndResetOnClose) {
  MockFileDescriptor fd(5);

  fd.Close();

  EXPECT_EQ(recently_closed_fd, 5);
  EXPECT_EQ(fd.get(), -1);
}

TEST_F(FileDescriptorTest, FileDescriptorIsClosedOnDestruction) {
  {
    MockFileDescriptor fd(6);
  }

  EXPECT_EQ(recently_closed_fd, 6);
}

}  // namespace
}  // namespace xls
