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

#include "xls/dslx/type_system/zip_types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

using testing::ElementsAre;
using testing::FieldsAre;

enum class CallbackKind : uint8_t {
  kAggregateStart,
  kAggregateEnd,
  kMatchedLeaf,
  kMismatch,
};

struct CallbackData {
  CallbackKind kind;
  const Type* lhs = nullptr;
  const Type* rhs = nullptr;
  std::optional<AggregatePair> aggregates;
};

std::ostream& operator<<(std::ostream& os, CallbackKind kind) {
  std::string kind_str;
  switch (kind) {
    case CallbackKind::kAggregateStart:
      kind_str = "aggregate-start";
      break;
    case CallbackKind::kAggregateEnd:
      kind_str = "aggregate-end";
      break;
    case CallbackKind::kMatchedLeaf:
      kind_str = "matched-leaf";
      break;
    case CallbackKind::kMismatch:
      kind_str = "mismatch";
      break;
  }
  os << kind_str;
  return os;
}

// Convenience for nicer matcher output.
std::ostream& operator<<(std::ostream& os, const CallbackData& data) {
  std::string lhs_str = "(null)";
  if (data.lhs != nullptr) {
    lhs_str = data.lhs->ToString();
  }
  std::string rhs_str = "(null)";
  if (data.rhs != nullptr) {
    rhs_str = data.rhs->ToString();
  }
  os << "{.kind=" << data.kind
     << absl::StreamFormat(", .lhs=%s, .rhs=%s}", lhs_str, rhs_str);
  return os;
}

// Trivial implementation of the callbacks abstract interface that just collects
// events as they are triggered in an underlying vector.
class ZipTypesCallbacksCollector : public ZipTypesCallbacks {
 public:
  ~ZipTypesCallbacksCollector() override = default;

  absl::Status NoteAggregateStart(const AggregatePair& pair) override {
    data_.push_back(CallbackData{.kind = CallbackKind::kAggregateStart,
                                 .aggregates = pair});
    return absl::OkStatus();
  }
  absl::Status NoteAggregateEnd(const AggregatePair& pair) override {
    data_.push_back(
        CallbackData{.kind = CallbackKind::kAggregateEnd, .aggregates = pair});
    return absl::OkStatus();
  }
  absl::Status NoteMatchedLeafType(const Type& lhs, const Type& rhs) override {
    data_.push_back(CallbackData{
        .kind = CallbackKind::kMatchedLeaf, .lhs = &lhs, .rhs = &rhs});
    return absl::OkStatus();
  }
  absl::Status NoteTypeMismatch(const Type& lhs, const Type& rhs) override {
    data_.push_back(CallbackData{
        .kind = CallbackKind::kMismatch, .lhs = &lhs, .rhs = &rhs});
    return absl::OkStatus();
  }

  absl::Span<const CallbackData> data() const { return data_; }

 private:
  std::vector<CallbackData> data_;
};

TEST(ZipTypesTest, DifferentBitsTypes) {
  auto lhs = BitsType::MakeU32();
  auto rhs = BitsType::MakeS32();

  ZipTypesCallbacksCollector collector;
  XLS_ASSERT_OK(ZipTypes(*lhs, *rhs, collector));

  EXPECT_EQ(collector.data().size(), 1);
  EXPECT_EQ(collector.data()[0].kind, CallbackKind::kMismatch);
  EXPECT_EQ(collector.data()[0].lhs, lhs.get());
  EXPECT_EQ(collector.data()[0].rhs, rhs.get());
}

// This is a special case for our type system (as represented in the C++
// objects, that is) -- bits types are type-compatible-with but structurally not
// identical to bits constructors in an array.
TEST(ZipTypesTest, BitsConstructorVsBitsType) {
  auto lhs = BitsType::MakeU32();
  auto rhs =
      std::make_unique<ArrayType>(std::make_unique<BitsConstructorType>(
                                      /*is_signed=*/TypeDim::CreateBool(false)),
                                  TypeDim::CreateU32(32));

  EXPECT_TRUE(lhs->CompatibleWith(*rhs));
  EXPECT_TRUE(rhs->CompatibleWith(*lhs));

  ZipTypesCallbacksCollector collector;
  XLS_ASSERT_OK(ZipTypes(*lhs, *rhs, collector));

  EXPECT_THAT(collector.data(),
              ElementsAre(FieldsAre(CallbackKind::kMatchedLeaf, lhs.get(),
                                    rhs.get(), std::nullopt)));
}

TEST(ZipTypesTest, TupleWithOneDifferingElement) {
  std::unique_ptr<TupleType> lhs =
      TupleType::Create2(BitsType::MakeU32(), BitsType::MakeU64());
  std::unique_ptr<TupleType> rhs =
      TupleType::Create2(BitsType::MakeU32(), BitsType::MakeS32());

  ZipTypesCallbacksCollector collector;
  XLS_ASSERT_OK(ZipTypes(*lhs, *rhs, collector));

  ASSERT_EQ(collector.data().size(), 4);

  std::pair<const TupleType*, const TupleType*> aggregates =
      std::make_pair(lhs.get(), rhs.get());
  EXPECT_THAT(collector.data()[0],
              FieldsAre(CallbackKind::kAggregateStart, nullptr, nullptr,
                        AggregatePair{aggregates}));
  EXPECT_THAT(collector.data()[1],
              FieldsAre(CallbackKind::kMatchedLeaf, &lhs->GetMemberType(0),
                        &rhs->GetMemberType(0), std::nullopt));
  EXPECT_THAT(collector.data()[2],
              FieldsAre(CallbackKind::kMismatch, &lhs->GetMemberType(1),
                        &rhs->GetMemberType(1), std::nullopt));
  EXPECT_THAT(collector.data()[3],
              FieldsAre(CallbackKind::kAggregateEnd, nullptr, nullptr,
                        AggregatePair{aggregates}));
}

}  // namespace
}  // namespace xls::dslx
