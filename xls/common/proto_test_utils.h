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

#ifndef XLS_COMMON_PROTO_TEST_UTILS_H_
#define XLS_COMMON_PROTO_TEST_UTILS_H_

#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"

namespace xls::proto_testing {

// Mark the proto-matcher as allowing for extra fields to be set in the result.
// NB that extra elements of repeated fields are still considered a failure.
template <typename InnerMatcher, typename = InnerMatcher::is_proto_matcher>
inline InnerMatcher Partially(InnerMatcher i) {
  i.SetPartial();
  return i;
}

// Mark the proto matcher as allowing for repeated fields to be in any order.
template <typename InnerMatcher, typename = InnerMatcher::is_proto_matcher>
inline InnerMatcher IgnoringRepeatedFieldOrdering(InnerMatcher i) {
  i.SetUnorderedRepeatedFields();
  return i;
}

namespace internal {
// Matcher ignore-checker that makes fields not in 'gold' ignored.
template <typename RealProto>
class PartialIgnore final
    : public google::protobuf::util::MessageDifferencer::IgnoreCriteria {
  using SpecificField = google::protobuf::util::MessageDifferencer::SpecificField;

 public:
  explicit PartialIgnore(const google::protobuf::Message& msg) : message_(msg) {}

  bool IsIgnored(const google::protobuf::Message& gold, const google::protobuf::Message& test,
                 const google::protobuf::FieldDescriptor* field,
                 const std ::vector<SpecificField>& specific_field) final {
    // Ignore any field fully absent from the gold proto.
    if (field->is_repeated()) {
      return gold.GetReflection()->FieldSize(gold, field) == 0;
    }
    return !gold.GetReflection()->HasField(gold, field);
  }

  bool IsUnknownFieldIgnored(const google::protobuf::Message&, const google::protobuf::Message&,
                             const SpecificField&,
                             const std ::vector<SpecificField>&) final {
    return true;
  }

 private:
  const google::protobuf::Message& message_;
};

// Matcher that checks for proto equality. It can be modified by the Partially
// and IgnoreRepeatedFieldOrdering to adjust the values.
template <typename RealProto>
class EqualsProtoMatcher {
 public:
  static_assert(std::is_base_of_v<google::protobuf::Message, RealProto>,
                "Proto matcher only valid for ProtoBufs");
  using is_gtest_matcher = void;
  using is_proto_matcher = void;

  explicit EqualsProtoMatcher(RealProto message)
      : real_message_(std::move(message)) {}

  const google::protobuf::Message& message() const { return real_message_; }

  template <typename T>
  bool MatchAndExplain(T test_message,
                       testing::MatchResultListener* listener) const {
    static_assert(false, "bad call!");
    return false;
  }

  bool MatchAndExplain(RealProto test_message,
                       testing::MatchResultListener* listener) const {
    google::protobuf::util::MessageDifferencer diff;
    diff.set_report_ignores(false);
    if (partial_) {
      diff.AddIgnoreCriteria(
          std::make_unique<PartialIgnore<RealProto>>(message()));
    }
    std::string str_report;
    if (unordered_repeated_fields_) {
      diff.set_repeated_field_comparison(
          google::protobuf::util::MessageDifferencer::AS_SET);
    }
    diff.ReportDifferencesToString(&str_report);
    bool same_message = diff.Compare(real_message_, test_message);
    if (same_message) {
      return true;
    }
    *listener << str_report;
    return false;
  }
  // Describes this matcher to an ostream.
  void DescribeTo(std::ostream* os) const {
    *os << absl::StreamFormat(
        "Equal%s%s to %s", partial_ ? " (ignoring extra fields)" : "",
        unordered_repeated_fields_ ? " (ignoring repeated field order)" : "",
        message().DebugString());
  }

  // Describes the negation of this matcher to an ostream.
  void DescribeNegationTo(std::ostream* os) const {
    *os << "Not ";
    DescribeTo(os);
  }

  void SetPartial() { partial_ = true; }
  void SetUnorderedRepeatedFields() { unordered_repeated_fields_ = true; }

 private:
  RealProto real_message_;
  bool partial_ = false;
  bool unordered_repeated_fields_ = false;
};

// A helper to delay parsing a string-proto until we have an actual type to give
// it.
class ParseAndCompareMatcher {
 public:
  using is_gtest_matcher = void;
  using is_proto_matcher = void;

  explicit ParseAndCompareMatcher(std::string_view proto) : proto_(proto) {}

  template <typename RealProto>
  bool MatchAndExplain(RealProto test_message,
                       testing::MatchResultListener* listener) const
    requires(std::is_base_of_v<google::protobuf::Message, RealProto>)
  {
    std::remove_reference_t<RealProto> cpy = test_message;
    cpy.Clear();
    bool parsed = google::protobuf::TextFormat::ParseFromString(proto_, &cpy);
    if (!parsed) {
      *listener << "Unable to parse " << proto_ << " as " << cpy.GetTypeName();
      return false;
    }
    EqualsProtoMatcher<RealProto> matcher(std::move(cpy));
    if (partial_) {
      matcher = Partially(matcher);
    }
    if (unordered_) {
      matcher = IgnoringRepeatedFieldOrdering(matcher);
    }
    return testing::ExplainMatchResult<RealProto>(matcher, test_message,
                                                  listener);
  }
  // Describes this matcher to an ostream.
  void DescribeTo(std::ostream* os) const {
    *os << absl::StreamFormat(
        "Equal%s%s to %s", partial_ ? " (ignoring extra fields)" : "",
        unordered_ ? " (ignoring repeated field order)" : "", proto_);
  }

  // Describes the negation of this matcher to an ostream.
  void DescribeNegationTo(std::ostream* os) const {
    *os << "Not ";
    DescribeTo(os);
  }

  void SetPartial() { partial_ = true; }
  void SetUnorderedRepeatedFields() { unordered_ = true; }

 private:
  std::string proto_;
  bool partial_ = false;
  bool unordered_ = false;
};
}  // namespace internal

// Compare protos for equality. NB This relies on proto reflection. Non-lite
// protos only!
template <typename Proto>
inline auto EqualsProto(Proto proto) {
  static_assert(std::is_base_of_v<google::protobuf::Message, Proto>);
  return internal::EqualsProtoMatcher<Proto>(std::move(proto));
}

// Parse then compare the proto.
template <>
inline auto EqualsProto(std::string_view proto) {
  return internal::ParseAndCompareMatcher(proto);
}
// Parse then compare the proto.
template <>
inline auto EqualsProto(const char* proto) {
  return internal::ParseAndCompareMatcher(proto);
}

}  // namespace xls::proto_testing

#endif  // XLS_COMMON_PROTO_TEST_UTILS_H_
