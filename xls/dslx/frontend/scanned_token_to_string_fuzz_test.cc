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

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"

namespace {

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Not;
using ::xls::status_testing::IsOk;
using ::xls::status_testing::IsOkAndHolds;
using ::xls::status_testing::StatusIs;

constexpr size_t kMaxModuleLengthInBytes = 5'000;

MATCHER_P(
    TokenStreamHasPositionalErrorData, file_table,
    "Calling GetPostitionalErrorData() on Token stream returns OK status.") {
  xls::dslx::FileTable& ft = *file_table;
  return ::testing::ExplainMatchResult(
      IsOk(), xls::dslx::GetPositionalErrorData(arg.status(), std::nullopt, ft),
      result_listener);
}

// Matcher to check correctness of a stream of tokens.
//
// This matcher is intended to be used with fuzzing- we want to check that the
// tokens we get back from the scanner are the same as the input. However, we
// want to be able to handle strings that are escaped in the input where
// different escapings may be equivalent (e.g. octal vs. hex). For most tokens,
// we do basic string equality, but for string and character literals we skip
// the contents, only checking that there are delimiters where the token's span
// says they should be.
//
// You might think it would be a good idea to escape/unescape the literals and
// check their correctness. Currently, we don't match the behavior of any of
// absl::CEscape/CUnescape and friends (e.g. no octal) and such an matcher would
// either require us to try no match absl::CEscape/Unescape (not clear it's a
// good idea) or to call the scanner and have the strings match trivially.
class TokenStreamMatcher {
 public:
  using is_gtest_matcher = void;

  explicit TokenStreamMatcher(std::string_view expected,
                              xls::dslx::FileTable& file_table)
      : expected_(expected), file_table_(file_table) {}

  // Given a span in text, find the number of bytes between the start and the
  // end of the span.
  //
  // If the span is on a single line, this is just the difference between the
  // start and end column numbers. If the span is on multiple lines, we need to
  // count the number of bytes in the intervening lines.
  static absl::StatusOr<int64_t> SpanSize(const xls::dslx::Span& span,
                                          std::string_view text) {
    if (span.start().lineno() == span.limit().lineno()) {
      return span.limit().colno() - span.start().colno();
    }

    // We have a multi-line span, we need to count how long the intervening
    // lines are. The start has already been stripped.
    XLS_RET_CHECK_GT(span.limit().lineno(), span.start().lineno());
    int64_t accumulated_line_size = 0;
    for (int64_t i = 0; i < span.limit().lineno() - span.start().lineno();
         ++i) {
      size_t line_end = text.find('\n');
      if (line_end == std::string::npos) {
        return absl::InvalidArgumentError("Span had more lines than the text");
      }
      accumulated_line_size += line_end + 1;
      text.remove_prefix(line_end + 1);
    }
    if (text.size() < span.limit().colno()) {
      return absl::InvalidArgumentError(
          "Span ended on a line that was too short.");
    }
    return accumulated_line_size + span.limit().colno();
  }

  // For tokens of the form <DELIMITER><CONTENT><DELIMITER>, match the
  // delimiters. The closing delimiter is found using the span of the token.
  //
  // Returns a pair where the first element indicates whether the match was
  // successful (returning true) and the second element is the expected
  // string_view after the removing the matched token from the beginning.
  template <char Delimiter>
  static std::pair<bool, std::string_view> MatchTokenWithDelimiter(
      std::string_view expected, const xls::dslx::Token& token,
      const xls::dslx::FileTable& file_table, std::ostream* listener) {
    absl::StatusOr<int64_t> span_size = SpanSize(token.span(), expected);
    if (!span_size.ok()) {
      *listener << span_size.status().message();
      return std::make_pair(false, expected);
    }
    if (*span_size > expected.size()) {
      *listener << "Expected token to be " << *span_size << " characters, but "
                << "was " << expected.size();
      return std::make_pair(false, expected);
    }
    if (expected[0] != Delimiter) {
      *listener << "Expected token to start with " << Delimiter << ", but was "
                << expected[0];
      return {false, expected};
    }

    if (expected[span_size.value() - 1] != Delimiter) {
      *listener << "Expected token to end with " << Delimiter << ", but was "
                << expected[span_size.value() - 1];
      *listener << "\nToken span is " << token.span().ToRepr(file_table);
      return {false, expected};
    }

    expected.remove_prefix(span_size.value());
    return {true, expected};
  }

  // Checks that a token matches the beginning of the expected string.
  //
  // Returns a pair where the first element indicates whether the match was
  // successful (returning true) and the second element is the expected
  // string_view after the removing the matched token from the beginning.
  static std::pair<bool, std::string_view> MatchToken(
      std::string_view expected, const xls::dslx::Token& token,
      const xls::dslx::FileTable& file_table, std::ostream* listener) {
    switch (token.kind()) {
      case xls::dslx::TokenKind::kCharacter:
        return MatchTokenWithDelimiter<'\''>(expected, token, file_table,
                                             listener);
      case xls::dslx::TokenKind::kString:
        return MatchTokenWithDelimiter<'\"'>(expected, token, file_table,
                                             listener);
      default: {
        std::string token_str = token.ToString();
        std::string_view token_str_view = token_str;
        bool match = absl::StartsWith(expected, token_str_view);
        if (match) {
          expected.remove_prefix(token_str_view.size());
        } else {
          *listener << token_str_view << " did not match " << expected;
        }
        return std::make_pair(match, expected);
      }
    }
  }

  bool MatchAndExplain(absl::Span<xls::dslx::Token const> tokens,
                       std::ostream* listener) const {
    std::string_view expected_substr = expected_;
    for (const xls::dslx::Token& token : tokens) {
      auto [match, next_expected_substr] =
          MatchToken(expected_substr, token, file_table_, listener);
      if (!match) {
        return false;
      }
      expected_substr = next_expected_substr;
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const {
    *os << "token stream matches expected string " << expected_;
  }

  void DescribeNegationTo(std::ostream* os) const {
    *os << "token stream does not match expected string " << expected_;
  }

 private:
  std::string expected_;
  xls::dslx::FileTable& file_table_;
};

inline auto TokenStreamMatches(std::string_view expected,
                               xls::dslx::FileTable& file_table) {
  return TokenStreamMatcher(expected, file_table);
}

void ScanningGivesErrorOrConvertsToOriginal(std::string_view test_module) {
  xls::dslx::FileTable file_table;
  xls::dslx::Scanner scanner(file_table, xls::dslx::Fileno(0),
                             std::string(test_module),
                             /*include_whitespace_and_comments=*/true);
  EXPECT_THAT(
      scanner.PopAll(),
      AnyOf(IsOkAndHolds(TokenStreamMatches(test_module, file_table)),
            AllOf(Not(IsOk()), Not(StatusIs(absl::StatusCode::kInternal)),
                  TokenStreamHasPositionalErrorData(&file_table))));
}

FUZZ_TEST(ScanFuzzTest, ScanningGivesErrorOrConvertsToOriginal)
    .WithDomains(fuzztest::Arbitrary<std::string>().WithMaxSize(
        kMaxModuleLengthInBytes));

TEST(ScanFuzzTest, ScanningGivesErrorOrConvertsToOriginalRegressionBackslash) {
  ScanningGivesErrorOrConvertsToOriginal("\"\\\"\"");
}

TEST(ScanFuzzTest, ScanningGivesErrorOrConvertsToOriginalRegressionHexEscape) {
  ScanningGivesErrorOrConvertsToOriginal(std::string("\"\000\"", 3));
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionBackslashesAndQuotes) {
  ScanningGivesErrorOrConvertsToOriginal("\"\\\"\"");
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionUnclosedFunkyStrLiteral) {
  ScanningGivesErrorOrConvertsToOriginal("\"\\\t");
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionLiteralWithBackslashes) {
  ScanningGivesErrorOrConvertsToOriginal("\"\\\\\"");
}
TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionCharLiteralQuote) {
  ScanningGivesErrorOrConvertsToOriginal("\'\\\"\'");
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionCharLiteralEscaped) {
  ScanningGivesErrorOrConvertsToOriginal(std::string("\'\000\'\t", 4));
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionNullByteStringLit2Zeros) {
  ScanningGivesErrorOrConvertsToOriginal("\"\\00\"");
}
TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionNewlineInString) {
  ScanningGivesErrorOrConvertsToOriginal("\"\n\"");
}
TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionNewlineWithNull) {
  ScanningGivesErrorOrConvertsToOriginal(std::string("\"\n\000\"", 4));
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionTabQuoteNewline) {
  ScanningGivesErrorOrConvertsToOriginal("\t\"\n\"");
}

TEST(ScanFuzzTest, ScanningGivesErrorOrConvertsToOriginalRegressionNLQuoteNL) {
  ScanningGivesErrorOrConvertsToOriginal("\n\"\n\"");
}

TEST(ScanFuzzTest,
     ScanningGivesErrorOrConvertsToOriginalRegressionNewlineLitWithWhitespace) {
  ScanningGivesErrorOrConvertsToOriginal("\t\t\"\n\"");
}

}  // namespace
