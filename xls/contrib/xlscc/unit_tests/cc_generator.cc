// Copyright 2023 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/contrib/xlscc/unit_tests/cc_generator.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "xls/common/random_util.h"

namespace xlscc {
namespace {

class Function {
 public:
  std::string return_type;
  std::string name;
  std::vector<std::string> parameters;

  static Function Generate(absl::BitGenRef bit_gen) {
    std::vector<Function> functions{
        {"ac_int", "leading_sign", {}}, {"ac_fixed", "bit_complement", {}},
        {"int", "to_int", {}},          {"unsigned int", "to_uint", {}},
        {"long", "to_long", {}},        {"unsigned long", "to_ulong", {}},
        {"long long", "to_int64", {}},  {"unsigned long long", "to_uint64", {}},
    };
    return xls::RandomChoice(functions, bit_gen);
  }
};

class BuiltInIntegerTypes {
 public:
  std::string name;
  bool is_signed;
  char bit_width;
};

class Variable {
 public:
  std::string name;
  std::string type;
  std::string value;
  int64_t bit_width;
  std::vector<std::string> templated_parameters;

  static Variable GenerateInt(const std::string& variable_name,
                              absl::BitGenRef bit_gen) {
    std::vector<BuiltInIntegerTypes> int_types{
        {"bool", false, 1},         {"char", false, 8},
        {"signed char", true, 8},   {"unsigned char", false, 8},
        {"signed short", true, 16}, {"unsigned short", false, 16},
        {"signed int", true, 32},   {"unsigned int", false, 32},
        {"signed long", true, 64},  {"unsigned long", false, 64},
        {"Slong", true, 64},        {"Ulong", false, 64},
    };
    auto int_metadata = xls::RandomChoice(int_types, bit_gen);
    std::string value_str;
    if (int_metadata.bit_width < 64) {
      int64_t min_value = (1 << int_metadata.bit_width) - 1;
      int64_t max_value = 1 << (int_metadata.bit_width - 1);
      value_str = std::to_string(absl::Uniform(bit_gen, min_value, max_value));
    } else {
      if (int_metadata.is_signed) {
        int64_t min_value = std::numeric_limits<int64_t>::min();
        int64_t max_value = std::numeric_limits<int64_t>::max();
        value_str =
            std::to_string(absl::Uniform(bit_gen, min_value, max_value));
      } else {
        uint64_t min_value = std::numeric_limits<uint64_t>::min();
        uint64_t max_value = std::numeric_limits<uint64_t>::max();
        value_str =
            std::to_string(absl::Uniform(bit_gen, min_value, max_value));
      }
    }
    return {variable_name,
            int_metadata.name,
            value_str,
            int_metadata.bit_width,
            {}};
  }

  static Variable GenerateXlsInt(const std::string& variable_name,
                                 absl::BitGenRef bit_gen) {
    bool is_signed = absl::Bernoulli(bit_gen, 0.5);
    int64_t bit_width = absl::Uniform(absl::IntervalClosed, bit_gen, 1, 100);
    int64_t max_value = std::numeric_limits<int64_t>::max();
    int64_t min_value = is_signed ? std::numeric_limits<int64_t>::min() : 0;
    if (bit_width + static_cast<int64_t>(is_signed) < 64) {
      int64_t range = (int64_t{1} << bit_width) + 10;
      max_value = is_signed ? range / 2 : range;
      min_value = is_signed ? -range / 2 : 0;
    }
    return {variable_name,
            "ac_int",
            std::to_string(absl::Uniform(bit_gen, min_value, max_value)),
            bit_width,
            {std::to_string(bit_width), is_signed ? "true" : "false"}};
  }

  static Variable GenerateXlsFixed(const std::string& variable_name,
                                   absl::BitGenRef bit_gen) {
    std::vector<std::string> ac_q_mode{
        "AC_TRN",     "AC_RND",         "AC_TRN_ZERO", "AC_RND_ZERO",
        "AC_RND_INF", "AC_RND_MIN_INF", "AC_RND_CONV", "AC_RND_CONV_ODD"};
    std::vector<std::string> ac_o_mode{"AC_WRAP", "AC_SAT", "AC_SAT_ZERO",
                                       "AC_SAT_SYM"};

    bool is_float = absl::Bernoulli(bit_gen, 0.5);
    bool is_signed = absl::Bernoulli(bit_gen, 0.5);
    std::string quantization = xls::RandomChoice(ac_q_mode, bit_gen);
    std::string overflow = xls::RandomChoice(ac_o_mode, bit_gen);
    int64_t bit_width = absl::Uniform(absl::IntervalClosed, bit_gen, 1, 100);
    int64_t integer_width =
        absl::Uniform(absl::IntervalClosed, bit_gen, 1, bit_width);

    int64_t max_value = std::numeric_limits<int64_t>::max();
    int64_t min_value = is_signed ? std::numeric_limits<int64_t>::min() : 0;
    if (integer_width + static_cast<int64_t>(is_signed) < 64) {
      int64_t range = (int64_t{1} << integer_width);
      max_value = is_signed ? range / 2 : range;
      min_value = is_signed ? -range / 2 : 0;
    }
    std::string value;
    if (is_float) {
      // doubles are 32 bit integers with 32 bits of decimal precision.
      max_value = std::numeric_limits<int32_t>::max();
      min_value = is_signed ? std::numeric_limits<int32_t>::min() : 0;
      value = std::to_string(
          absl::Uniform<double>(bit_gen, static_cast<double>(min_value),
                                static_cast<double>(max_value)));
    } else {
      value =
          std::to_string(absl::Uniform<int64_t>(bit_gen, min_value, max_value));
    }

    return {
        variable_name,
        "ac_fixed",
        value,
        bit_width,
        {std::to_string(bit_width), std::to_string(integer_width),
         is_signed ? "true" : "false", quantization, overflow},
    };
  }

  static Variable GenerateVariable(const std::string& variable_name,
                                   absl::BitGenRef bit_gen, VariableType type) {
    switch (type) {
      case VariableType::kAcInt:
        return GenerateXlsInt(variable_name, bit_gen);
      case VariableType::kAcFixed:
        return GenerateXlsFixed(variable_name, bit_gen);
      default:
        return GenerateInt(variable_name, bit_gen);
    }
  }

  std::string Type() {
    std::stringstream content;
    content << type;
    if (!templated_parameters.empty()) {
      content << "<" << templated_parameters[0];
      for (int i = 1; i < templated_parameters.size(); i++) {
        content << ", " << templated_parameters[i];
      }
      content << ">";
    }
    return content.str();
  }
  std::string Declaration() {
    std::stringstream content;
    content << Type() << " " << name;
    return content.str();
  }
};
}  // namespace

void GenerateFixedBinaryOp(std::stringstream& content, Variable& v1,
                           Variable& v2, std::mt19937_64& bit_gen) {
  std::vector<std::string> binary_ops = {"+", "-", "*", "/", "|", "&", "^"};
  std::vector<std::string> binary_ops_with_ac_int = {"+", "-", "*",  "/", "|",
                                                     "&", "^", "<<", ">>"};
  std::string operation1 = xls::RandomChoice(
      v2.type == "ac_int" ? binary_ops_with_ac_int : binary_ops, bit_gen);
  content << "result = " << v1.name << " " << operation1 << " " << v2.name
          << ";\n";
}

void GenerateFixedBinaryAssignOp(std::stringstream& content, Variable& v1,
                                 Variable& v2, std::mt19937_64& bit_gen) {
  std::vector<std::string> assign_ops = {
      "+=", "-=", "*=", "/=", "|=", "&=", "^="};
  std::vector<std::string> assign_ops_with_ac_int = {
      "+=", "-=", "*=", "/=", "|=", "&=", "^=", "<<=", ">>="};
  std::string operation1 = xls::RandomChoice(
      v2.type == "ac_int" ? assign_ops_with_ac_int : assign_ops, bit_gen);
  content << v1.name << " " << operation1 << " " << v2.name << ";\n";
  content << "result = " << v1.name << ";\n";
}

void GenerateUnaryOp(std::stringstream& content, Variable& output, Variable& v1,
                     std::mt19937_64& bit_gen) {
  std::vector<std::string> postfix_unary_operations = {"++", "--"};
  std::vector<std::string> prefix_unary_operations = {"!", "+",  "-",
                                                      "~", "++", "--"};
  bool use_prefix = absl::Uniform<char>(bit_gen, 0, 2) == 0;
  if (use_prefix) {
    std::string op = xls::RandomChoice(prefix_unary_operations, bit_gen);
    content << output.name << " = " << op << v1.name << ";\n";
  } else {
    std::string op = xls::RandomChoice(postfix_unary_operations, bit_gen);
    content << v1.name << op << ";\n";
    content << output.name << " = " << v1.name << ";\n";
  }
}
void GenerateComparisonOp(std::stringstream& content, Variable& v1,
                          Variable& v2, std::mt19937_64& bit_gen) {
  std::vector<std::string> operations = {">", "<", ">=", "<=", "==", "!="};
  std::string op = xls::RandomChoice(operations, bit_gen);
  content << "result = (" << v1.name << " " << op << " " << v2.name
          << ") ? 1 : 0;\n";
}

std::string GenerateTest(uint32_t seed, VariableType type) {
  std::mt19937_64 bit_gen(seed);

  Variable var = Variable::GenerateVariable("result", bit_gen, type);
  std::stringstream content;
  content << "#ifndef __SYNTHESIS__\n";
  content << "#include <iostream>\n";
  content << "#endif\n";

  content << "#include \"ac_int.h\"\n";
  content << "#include \"ac_fixed.h\"\n";
  content << "#pragma hls_top\n";
  content << var.Type() << " FuzzTestFixed" << seed << "() {\n";
  content << var.Declaration() << " = " << var.value << ";\n";
  char op_type = absl::Uniform<char>(bit_gen, 0, 5);
  Variable input = Variable::GenerateVariable("input", bit_gen, type);
  content << input.Declaration() << " = " << input.value << ";\n";
  if (op_type == 0) {
    Variable v2 = Variable::GenerateVariable("v2", bit_gen, type);
    content << v2.Declaration() << " = " << v2.value << ";\n";
    GenerateFixedBinaryOp(content, input, v2, bit_gen);
  } else if (op_type == 1) {
    Variable v2 = Variable::GenerateVariable("v2", bit_gen, type);
    content << v2.Declaration() << " = " << v2.value << ";\n";
    GenerateFixedBinaryAssignOp(content, v2, input, bit_gen);
  } else if (op_type == 2) {
    GenerateUnaryOp(content, var, input, bit_gen);
  } else if (op_type == 3) {
    GenerateComparisonOp(content, var, input, bit_gen);
  } else {
    Function selected = Function::Generate(bit_gen);
    if (input.bit_width <= 64) {
      content << var.name << " = " << var.Type() << "(" << input.name << "."
              << selected.name << "()" << ");\n";
    } else {
      content << var.name << " = " << var.Type() << "(" << input.name << ");\n";
    }
  }

  content << "#ifndef __SYNTHESIS__\n";
  content << "std::cout << " << var.name
          << ".to_string(AC_BIN, false, true);\n";
  content << "#endif\n";
  content << "return " << var.name << ";\n";
  content << "}\n";

  content << "int main() { FuzzTestFixed" << seed << "(); return 0; }\n";
  return content.str();
}

}  // namespace xlscc
