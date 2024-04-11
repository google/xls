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

#include "xls/solvers/z3_extract_counterexample.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls::solvers::z3 {
namespace {

absl::StatusOr<Value> Z3ValueToXlsValue(std::string_view z3_value_text,
                                        const BitsType& bits_type) {
  if (!absl::StartsWith(z3_value_text, "#x")) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expect Z3 value to start with '#x'; got: %s",
                        absl::CEscape(z3_value_text)));
  }

  std::string_view bits_text = z3_value_text.substr(2);
  XLS_ASSIGN_OR_RETURN(
      Bits bits, ParseUnsignedNumberWithoutPrefix(
                     bits_text, FormatPreference::kHex, bits_type.bit_count()));
  return Value(bits);
}

}  // namespace

absl::StatusOr<absl::flat_hash_map<std::string, Value>> ExtractCounterexample(
    std::string_view message, absl::Span<const IrParamSpec> params) {
  // Split on the opening ``` fence.
  std::vector<std::string_view> pieces =
      absl::StrSplit(message, absl::MaxSplits("```", 1));
  if (pieces.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not find model within solver message: `%s`",
                        absl::CEscape(message)));
  }

  // Split against the closing ``` fence, the model data resides in the middle.
  pieces = absl::StrSplit(pieces[1], absl::MaxSplits("```", 1));
  if (pieces.size() != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not find model (closing fence) within solver message: `%s`",
        absl::CEscape(message)));
  }
  std::string_view model_text = pieces[0];

  // Since the parameters can be presented out of order we build up a map from
  // parameter name to the spec (which includes the type) for easy lookup in
  // arbitrary order.
  absl::flat_hash_map<std::string, absl::Nonnull<const IrParamSpec*>>
      name_to_spec;
  for (const IrParamSpec& spec : params) {
    auto [it, inserted] = name_to_spec.insert({spec.name, &spec});
    XLS_RET_CHECK(inserted);
  }

  // Accumulate values as we iterate through lines -- note that when there are
  // more than one parameter there will be more than one line specifying data.
  absl::flat_hash_map<std::string, Value> results;
  results.reserve(params.size());

  // Go through each line, skipping empty ones, to ensure we understand all the
  // data the model presents.
  std::vector<std::string_view> model_lines = absl::StrSplit(model_text, '\n');
  for (std::string_view line : model_lines) {
    if (line.empty()) {
      continue;
    }

    // On each line split the parameter name from the data so we can parse the
    // data out.
    std::vector<std::string_view> sides =
        absl::StrSplit(line, absl::MaxSplits(" -> ", 1));
    if (sides.size() != 2) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not parse line in solver model: `%s`", absl::CEscape(line)));
    }

    // Ensure the parameter name matches our understanding of what parameter
    // we're currently processing (since we're placing it in a vector directly).
    std::string_view param_name = sides[0];
    param_name = absl::StripAsciiWhitespace(param_name);

    auto it = name_to_spec.find(param_name);
    if (it == name_to_spec.end()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("ExtractCounterexample; could not find parameter "
                          "name from model in user-provided spec: `%s`",
                          param_name));
    }

    absl::Nonnull<const IrParamSpec*> spec = it->second;

    // Use the type to guide how we parse out the solver-presented data item --
    // right now it must be a bits type, we give an unimplemented error for
    // unsupported types.
    absl::Nonnull<const Type*> type = spec->type;
    auto* bits_type = dynamic_cast<const BitsType*>(type);
    if (bits_type == nullptr) {
      return absl::UnimplementedError(
          absl::StrFormat("ExtractCounterexample; only bits-typed parameters "
                          "are currently supported; got: %s",
                          type->ToString()));
    }
    XLS_ASSIGN_OR_RETURN(Value value, Z3ValueToXlsValue(sides[1], *bits_type));

    auto it2 = results.find(param_name);
    if (it2 != results.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "ExtractCounterexample; saw duplicate param value for `%s`: %s",
          param_name, absl::CEscape(model_text)));
    }
    results.emplace_hint(it2, std::string{param_name}, std::move(value));
  }

  // Validate that we populated the right number of values for the parameter
  // specification.
  if (results.size() != params.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Z3 solver model counterexample did not include all parameters: %s",
        absl::CEscape(model_text)));
  }

  return results;
}

}  // namespace xls::solvers::z3
