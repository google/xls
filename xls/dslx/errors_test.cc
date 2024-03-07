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
#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/concrete_type.h"

namespace xls::dslx {
namespace {

TEST(ErrorsTest, TypeInferenceErrorMessage) {
  const std::string kFilename = "test.x";
  const Pos start(kFilename, 0, 0);
  const Pos limit(kFilename, 1, 1);
  Span span(start, limit);
  std::unique_ptr<Type> type = BitsType::MakeU32();
  absl::Status status =
      TypeInferenceErrorStatus(span, type.get(), "this is the message!");
  EXPECT_EQ(status.ToString(),
            "INVALID_ARGUMENT: TypeInferenceError: test.x:1:1-2:2 uN[32] this "
            "is the message!");
}

}  // namespace
}  // namespace xls::dslx
