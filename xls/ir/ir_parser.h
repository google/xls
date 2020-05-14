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

// The IR parser allows to build an IR from reading in and
// parsing textual IR.
//
// This is convenience functionality, great for debugging and
// construction of small test cases, it can be used by other
// front-ends to target XLS without having to fully link to it.

#ifndef THIRD_PARTY_XLS_IR_IR_PARSER_H_
#define THIRD_PARTY_XLS_IR_IR_PARSER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xls {

class ArgParser;

class Parser {
 public:
  // Parses the given input string as a package.
  static xabsl::StatusOr<std::unique_ptr<Package>> ParsePackage(
      absl::string_view input_string,
      absl::optional<absl::string_view> filename = absl::nullopt);

  // As above, but sets the entry function to be the given name in the returned
  // package.
  static xabsl::StatusOr<std::unique_ptr<Package>> ParsePackageWithEntry(
      absl::string_view input_string, absl::string_view entry,
      absl::optional<absl::string_view> filename = absl::nullopt);

  // Parse the input_string as a function into the given package.
  static xabsl::StatusOr<Function*> ParseFunction(
      absl::string_view input_string, Package* package);

  // Parse the input_string as a function type into the given package.
  static xabsl::StatusOr<FunctionType*> ParseFunctionType(
      absl::string_view input_string, Package* package);

  // Parse the input_string as a type into the given package.
  static xabsl::StatusOr<Type*> ParseType(absl::string_view input_string,
                                          Package* package);

  // Parses the given input string as a package skipping verification. This
  // should only be used in tests when malformed IR is desired.
  static xabsl::StatusOr<std::unique_ptr<Package>> ParsePackageNoVerify(
      absl::string_view input_string,
      absl::optional<absl::string_view> filename = absl::nullopt,
      absl::optional<absl::string_view> entry = absl::nullopt);

  // As above but creates a package of type PackageT where PackageT must be
  // type derived from Package.
  template <typename PackageT>
  static xabsl::StatusOr<std::unique_ptr<PackageT>> ParseDerivedPackageNoVerify(
      absl::string_view input_string,
      absl::optional<absl::string_view> filename = absl::nullopt,
      absl::optional<absl::string_view> entry = absl::nullopt);

  // Parses a literal value that should be of type "expected_type" and returns
  // it.
  static xabsl::StatusOr<Value> ParseValue(absl::string_view input_string,
                                           Type* expected_type);

  // Parses a value with embedded type information, specifically 'bits[xx]:'
  // substrings indicating the width of literal values. Value::ToString emits
  // strings of this form. Examples of strings parsable with this method:
  //   bits[32]:0x42
  //   (bits[7]:0, bits[8]:1)
  //   [bits[2]:1, bits[2]:2, bits[2]:3]
  static xabsl::StatusOr<Value> ParseTypedValue(absl::string_view input_string);

 private:
  friend class ArgParser;

  explicit Parser(Scanner scanner) : scanner_(scanner) {}

  // Parse starting from a single function.
  xabsl::StatusOr<Function*> ParseFunction(Package* package);

  // Parse starting from a function type.
  xabsl::StatusOr<FunctionType*> ParseFunctionType(Package* package);

  // A thin convenience function which parses a single boolean literal.
  xabsl::StatusOr<bool> ParseBool();

  // A thin convenience function which parses a single int64 number.
  xabsl::StatusOr<int64> ParseInt64();

  // A thin convenience function which parses a single identifier string.
  xabsl::StatusOr<std::string> ParseIdentifierString(TokenPos* pos = nullptr);

  // Convenience function that parses an identifier and resolve it to a value,
  // or returns a status error if it cannot.
  xabsl::StatusOr<BValue> ParseIdentifierValue(
      const absl::flat_hash_map<std::string, BValue>& name_to_value);

  // Parses a Value. Supports bits, array, and tuple types as well as their
  // nested variants. If expected_type is not given, the input string should
  // have embedded 'bits' types indicating the width of bits values as produced
  // by Value::ToString. For example: "(bits[32]:0x23, bits[0]:0x1)". If
  // expected_type is given, the string should NOT have embedded bits types as
  // produced by Value::ToHumanString. For example: "(0x23, 0x1)".
  xabsl::StatusOr<Value> ParseValueInternal(
      absl::optional<Type*> expected_type);

  // Parses a comma-delimited list of names surrounded by brackets; e.g.
  //
  //    "[foo, bar, baz]"
  //
  // Where the foo, bar, and baz identifiers are resolved via name_to_value.
  //
  // Returns an error if the parse fails or if any of the names cannot be
  // resolved via name_to_value.
  xabsl::StatusOr<std::vector<BValue>> ParseNameList(
      const absl::flat_hash_map<std::string, BValue>& name_to_value);

  // Parses a source location.
  // TODO(meheff): Currently the source location is a sequence of three
  // comma-separated numbers. Encapsulating the numbers in braces or something
  // would make the output less ambiguous. Example:
  // "and(x,y,pos={1,2,3},foo=bar)" vs "and(x,y,pos=1,2,3,foo=bar)"
  xabsl::StatusOr<SourceLocation> ParseSourceLocation();

  // Parse type specifications.
  xabsl::StatusOr<Type*> ParseType(Package* package);

  // Parse a tuple type (which can contain nested tuples).
  xabsl::StatusOr<Type*> ParseTupleType(Package* package);

  // Parse a bits type.
  xabsl::StatusOr<Type*> ParseBitsType(Package* package);

  // Parses a bits types and returns the width.
  xabsl::StatusOr<int64> ParseBitsTypeAndReturnWidth();

  // Builds a binary or unary BValue with the given Op using the given
  // FunctionBuilder and arg parser.
  xabsl::StatusOr<BValue> BuildBinaryOrUnaryOp(
      Op op, FunctionBuilder* fb, absl::optional<SourceLocation>* loc,
      ArgParser* arg_parser);

  // Parses the line-statements in the body of a function.
  xabsl::StatusOr<BValue> ParseFunctionBody(
      FunctionBuilder* fb,
      absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a full function signature, starting after the 'fn' keyword.
  //
  // Returns the newly created function builder and the annotated return type
  // (may be nullptr) after the opening brace has been popped.
  //
  // Note: FunctionBuilder must be unique_ptr because it is referred to by
  // pointer in BValue types.
  xabsl::StatusOr<std::pair<std::unique_ptr<FunctionBuilder>, Type*>>
  ParseSignature(absl::flat_hash_map<std::string, BValue>* name_to_value,
                 Package* package);

  // Pops the package name out of the scanner, of the form:
  //
  //  "package" <name>
  //
  // And returns the name.
  xabsl::StatusOr<std::string> ParsePackageName();

  bool AtEof() const { return scanner_.AtEof(); }

  Scanner scanner_;
};

/* static */
template <typename PackageT>
xabsl::StatusOr<std::unique_ptr<PackageT>> Parser::ParseDerivedPackageNoVerify(
    absl::string_view input_string, absl::optional<absl::string_view> filename,
    absl::optional<absl::string_view> entry) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser parser(std::move(scanner));

  XLS_ASSIGN_OR_RETURN(std::string package_name, parser.ParsePackageName());

  auto package = absl::make_unique<PackageT>(package_name, entry);
  while (!parser.AtEof()) {
    XLS_RETURN_IF_ERROR(parser.ParseFunction(package.get()).status())
        << "@ " << (filename.has_value() ? filename.value() : "<unknown file>");
  }

  // Ensure that, if there were explicit node ID hints in the input IR text,
  // that the package's next ID doesn't collide with anything.
  int64 max_id_seen = -1;
  for (auto& function : package->functions()) {
    for (Node* node : function->nodes()) {
      max_id_seen = std::max(max_id_seen, node->id());
    }
  }
  package->set_next_node_id(max_id_seen + 1);

  // Verify the given entry function exists in the package.
  if (entry.has_value()) {
    XLS_RETURN_IF_ERROR(package->GetFunction(*entry).status());
  }
  return package;
}

}  // namespace xls

#endif  // THIRD_PARTY_XLS_IR_IR_PARSER_H_
