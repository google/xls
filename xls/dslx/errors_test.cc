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
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/proto_test_utils.h"
#include "xls/dslx/dslx_status_payloads.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.pb.h"

namespace xls::dslx {
namespace {

using ::testing::UnorderedElementsAre;
using ::xls::proto_testing::EqualsProto;

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

TEST(ErrorsTest, SignednessMismatchErrorAnnotationPayload) {
  FileTable file_table;
  Module module("test_module", std::nullopt, file_table);
  const Span span1(Pos(Fileno(0), 1, 0), Pos(Fileno(0), 1, 1));
  const Span span2(Pos(Fileno(0), 2, 0), Pos(Fileno(0), 2, 1));
  BuiltinNameDef* u32_def = module.GetOrCreateBuiltinNameDef("u32");
  BuiltinNameDef* s32_def = module.GetOrCreateBuiltinNameDef("s32");
  BuiltinTypeAnnotation anno1(&module, span1, BuiltinType::kU32, u32_def);
  BuiltinTypeAnnotation anno2(&module, span2, BuiltinType::kS32, s32_def);

  absl::Status status =
      SignednessMismatchErrorStatus(&anno1, &anno2, file_table);
  std::optional<StatusPayloadProto> payload = GetStatusPayload(status);
  ASSERT_TRUE(payload.has_value());
  EXPECT_THAT(payload->spans(),
              UnorderedElementsAre(EqualsProto(ToProto(span1, file_table)),
                                   EqualsProto(ToProto(span2, file_table))));
}

}  // namespace
}  // namespace xls::dslx
