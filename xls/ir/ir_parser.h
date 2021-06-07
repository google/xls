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

// The IR parser allows to build an IR from reading in and
// parsing textual IR.
//
// This is convenience functionality, great for debugging and
// construction of small test cases, it can be used by other
// front-ends to target XLS without having to fully link to it.

#ifndef XLS_IR_IR_PARSER_H_
#define XLS_IR_IR_PARSER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"

namespace xls {

class ArgParser;

class Parser {
 public:
  // Parses the given input string as a package.
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackage(
      absl::string_view input_string,
      absl::optional<absl::string_view> filename = absl::nullopt);

  // As above, but sets the entry function to be the given name in the returned
  // package.
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackageWithEntry(
      absl::string_view input_string, absl::string_view entry,
      absl::optional<absl::string_view> filename = absl::nullopt);

  // Parse the input_string as a function into the given package.
  static absl::StatusOr<Function*> ParseFunction(absl::string_view input_string,
                                                 Package* package);

  // Parse the input_string as a proc into the given package.
  static absl::StatusOr<Proc*> ParseProc(absl::string_view input_string,
                                         Package* package);

  // Parse the input_string as a block into the given package.
  static absl::StatusOr<Block*> ParseBlock(absl::string_view input_string,
                                           Package* package);

  // Parse the input_string as a channel in the given package.
  static absl::StatusOr<Channel*> ParseChannel(absl::string_view input_string,
                                               Package* package);

  // Parse the input_string as a function type into the given package.
  static absl::StatusOr<FunctionType*> ParseFunctionType(
      absl::string_view input_string, Package* package);

  // Parse the input_string as a type into the given package.
  static absl::StatusOr<Type*> ParseType(absl::string_view input_string,
                                         Package* package);

  // Parses the given input string as a package skipping verification. This
  // should only be used in tests when malformed IR is desired.
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackageNoVerify(
      absl::string_view input_string,
      absl::optional<absl::string_view> filename = absl::nullopt,
      absl::optional<absl::string_view> entry = absl::nullopt);

  // As above but creates a package of type PackageT where PackageT must be
  // type derived from Package.
  template <typename PackageT>
  static absl::StatusOr<std::unique_ptr<PackageT>> ParseDerivedPackageNoVerify(
      absl::string_view input_string,
      absl::optional<absl::string_view> filename = absl::nullopt,
      absl::optional<absl::string_view> entry = absl::nullopt);

  // Parses a literal value that should be of type "expected_type" and returns
  // it.
  static absl::StatusOr<Value> ParseValue(absl::string_view input_string,
                                          Type* expected_type);

  // Parses a value with embedded type information, specifically 'bits[xx]:'
  // substrings indicating the width of literal values. Value::ToString emits
  // strings of this form. Examples of strings parsable with this method:
  //   bits[32]:0x42
  //   (bits[7]:0, bits[8]:1)
  //   [bits[2]:1, bits[2]:2, bits[2]:3]
  static absl::StatusOr<Value> ParseTypedValue(absl::string_view input_string);

 private:
  friend class ArgParser;

  explicit Parser(Scanner scanner) : scanner_(scanner) {}

  // Parse a function starting at the current scanner position.
  absl::StatusOr<Function*> ParseFunction(Package* package);

  // Parse a proc starting at the current scanner position.
  absl::StatusOr<Proc*> ParseProc(Package* package);

  // Parse a block starting at the current scanner position.
  absl::StatusOr<Block*> ParseBlock(Package* package);

  // Parse a proc starting at the current scanner position.
  absl::StatusOr<Channel*> ParseChannel(Package* package);

  // Parse starting from a function type.
  absl::StatusOr<FunctionType*> ParseFunctionType(Package* package);

  // A thin convenience function which parses a single boolean literal.
  absl::StatusOr<bool> ParseBool();

  // A thin convenience function which parses a single int64_t number.
  absl::StatusOr<int64_t> ParseInt64();

  // A thin convenience function which parses a single identifier string.
  absl::StatusOr<std::string> ParseIdentifier(TokenPos* pos = nullptr);

  // A thin convenience function which parses a quoted string.
  absl::StatusOr<std::string> ParseQuotedString(TokenPos* pos = nullptr);

  // Convenience function that parses an identifier and resolve it to a value,
  // or returns a status error if it cannot.
  absl::StatusOr<BValue> ParseAndResolveIdentifier(
      const absl::flat_hash_map<std::string, BValue>& name_to_value);

  // Parses a Value. Supports bits, array, and tuple types as well as their
  // nested variants. If expected_type is not given, the input string should
  // have embedded 'bits' types indicating the width of bits values as produced
  // by Value::ToString. For example: "(bits[32]:0x23, bits[0]:0x1)". If
  // expected_type is given, the string should NOT have embedded bits types as
  // produced by Value::ToHumanString. For example: "(0x23, 0x1)".
  absl::StatusOr<Value> ParseValueInternal(absl::optional<Type*> expected_type);

  // Parses a comma-separated sequence of values of the given type. Must have at
  // least one element in the sequence.
  absl::StatusOr<std::vector<Value>> ParseCommaSeparatedValues(Type* type);

  // Parses a comma-delimited list of names surrounded by brackets; e.g.
  //
  //    "[foo, bar, baz]"
  //
  // Where the foo, bar, and baz identifiers are resolved via name_to_value.
  //
  // Returns an error if the parse fails or if any of the names cannot be
  // resolved via name_to_value.
  absl::StatusOr<std::vector<BValue>> ParseNameList(
      const absl::flat_hash_map<std::string, BValue>& name_to_value);

  // Parses a source location.
  // TODO(meheff): Currently the source location is a sequence of three
  // comma-separated numbers. Encapsulating the numbers in braces or something
  // would make the output less ambiguous. Example:
  // "and(x,y,pos={1,2,3},foo=bar)" vs "and(x,y,pos=1,2,3,foo=bar)"
  absl::StatusOr<SourceLocation> ParseSourceLocation();

  // Parse type specifications.
  absl::StatusOr<Type*> ParseType(Package* package);

  // Parse a tuple type (which can contain nested tuples).
  absl::StatusOr<Type*> ParseTupleType(Package* package);

  // Parse a bits type.
  absl::StatusOr<Type*> ParseBitsType(Package* package);

  // Parses a bits types and returns the width.
  absl::StatusOr<int64_t> ParseBitsTypeAndReturnWidth();

  // Builds a binary or unary BValue with the given Op using the given
  // FunctionBuilder and arg parser.
  absl::StatusOr<BValue> BuildBinaryOrUnaryOp(
      Op op, BuilderBase* fb, absl::optional<SourceLocation>* loc,
      absl::string_view node_name, ArgParser* arg_parser);

  // Parses a node in a function/proc body. Example: "foo: bits[32] = add(x, y)"
  absl::StatusOr<BValue> ParseNode(
      BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value);

  // Parses a register declaration. Only supported in blocks.
  absl::StatusOr<Register*> ParseRegister(Block* block);

  struct ProcNext {
    BValue next_token;
    BValue next_state;
  };
  using BodyResult = absl::variant<BValue, ProcNext>;
  // Parses the line-statements in the body of a function/proc. Returns the
  // return value if the body is a function, or the next token/state pair if the
  // body is a proc.
  absl::StatusOr<BodyResult> ParseBody(
      BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a function signature, starting after the 'fn' keyword up to and
  // including the opening brace. Returns the newly created builder and the
  // annotated return type (may be nullptr) after the opening brace has been
  // popped.
  absl::StatusOr<std::pair<std::unique_ptr<FunctionBuilder>, Type*>>
  ParseFunctionSignature(
      absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a proc signature, starting after the 'proc' keyword up to and
  // including the opening brace. Returns the newly created builder.
  absl::StatusOr<std::unique_ptr<ProcBuilder>> ParseProcSignature(
      absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a block signature, starting after the 'block' keyword up to and
  // including the opening brace. Returns the newly created builder along with
  // information about the ports. The order of the returned Ports corresponds to
  // the order within the block.
  struct Port {
    std::string name;
    Type* type;
  };
  struct BlockSignature {
    std::string block_name;
    std::vector<Port> ports;
  };
  absl::StatusOr<BlockSignature> ParseBlockSignature(Package* package);

  // Pops the package name out of the scanner, of the form:
  //
  //  "package" <name>
  //
  // And returns the name.
  absl::StatusOr<std::string> ParsePackageName();

  bool AtEof() const { return scanner_.AtEof(); }

  Scanner scanner_;
};

/* static */
template <typename PackageT>
absl::StatusOr<std::unique_ptr<PackageT>> Parser::ParseDerivedPackageNoVerify(
    absl::string_view input_string, absl::optional<absl::string_view> filename,
    absl::optional<absl::string_view> entry) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser parser(std::move(scanner));

  XLS_ASSIGN_OR_RETURN(std::string package_name, parser.ParsePackageName());

  auto package = absl::make_unique<PackageT>(package_name, entry);
  std::string filename_str =
      (filename.has_value() ? std::string(filename.value()) : "<unknown file>");
  while (!parser.AtEof()) {
    XLS_ASSIGN_OR_RETURN(Token peek, parser.scanner_.PeekToken());
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "fn") {
      XLS_RETURN_IF_ERROR(parser.ParseFunction(package.get()).status())
          << "@ " << filename_str;
      continue;
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "proc") {
      XLS_RETURN_IF_ERROR(parser.ParseProc(package.get()).status())
          << "@ " << filename_str;
      continue;
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "block") {
      XLS_RETURN_IF_ERROR(parser.ParseBlock(package.get()).status())
          << "@ " << filename_str;
      continue;
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "chan") {
      XLS_RETURN_IF_ERROR(parser.ParseChannel(package.get()).status())
          << "@ " << filename_str;
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected fn, proc, or chan definition, got %s @ %s",
                        peek.value(), peek.pos().ToHumanString()));
  }

  // Verify the given entry function exists in the package.
  if (entry.has_value()) {
    XLS_RETURN_IF_ERROR(package->GetFunction(*entry).status());
  }
  return package;
}

}  // namespace xls

#endif  // XLS_IR_IR_PARSER_H_
