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

#include "xls/dslx/errors.h"

#include <memory>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

TEST(ErrorsTest, TypeInferenceErrorMessage) {
  FileTable file_table;
  const Pos start(Fileno(0), 0, 0);
  const Pos limit(Fileno(0), 1, 1);
  Span span(start, limit);
  std::unique_ptr<Type> type = BitsType::MakeU32();
  absl::Status status = TypeInferenceErrorStatus(
      span, type.get(), "this is the message!", file_table);
  EXPECT_EQ(
      status.ToString(),
      "INVALID_ARGUMENT: TypeInferenceError: <no-file>:1:1-2:2 uN[32] this "
      "is the message!");
}

}  // namespace
}  // namespace xls::dslx
