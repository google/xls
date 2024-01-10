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

#include <sys/stat.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/types/span.h"
#include "xls/common/random_util.h"

namespace xlscc {
namespace {

class Variable {
 public:
  std::string name;
  std::string type;
  std::string value;
  std::vector<std::string> templated_parameters;
  static Variable GenerateInt(char variable_number, absl::BitGenRef bit_gen) {
    bool is_signed = absl::Bernoulli(bit_gen, 0.5);
    int64_t bit_width = absl::Uniform(absl::IntervalClosed, bit_gen, 1, 100);
    int64_t max_value = std::numeric_limits<int64_t>::max();
    int64_t min_value = std::numeric_limits<int64_t>::min();
    if (bit_width + static_cast<int64_t>(is_signed) < 64) {
      int64_t range = (int64_t{1} << bit_width) + 10;
      max_value = is_signed ? range / 2 : range;
      min_value = is_signed ? -range / 2 : 0;
    }
    Variable var = {
        std::string(1, 'a' + variable_number),
        "ac_int",
        std::to_string(absl::Uniform(bit_gen, min_value, max_value)),
        {std::to_string(bit_width), is_signed ? "true" : "false"}};
    return var;
  }
  static Variable GenerateFixed(char variable_number, absl::BitGenRef bit_gen) {
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
    int64_t min_value = std::numeric_limits<int64_t>::min();
    if (bit_width + static_cast<int64_t>(is_signed) < 64) {
      int64_t range = (int64_t{1} << bit_width) + 10;
      max_value = is_signed ? range / 2 : range;
      min_value = is_signed ? -range / 2 : 0;
    }

    std::string value;
    if (is_float) {
      value = std::to_string(
          absl::Uniform<double>(bit_gen, static_cast<double>(min_value),
                                static_cast<double>(max_value)));
    } else {
      value =
          std::to_string(absl::Uniform<int64_t>(bit_gen, min_value, max_value));
    }

    Variable var = {
        std::string(1, 'a' + variable_number), "ac_fixed", value,
        {std::to_string(bit_width), std::to_string(integer_width),
         is_signed ? "true" : "false", quantization, overflow},
    };
    return var;
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

std::string GenerateIntTest(uint32_t seed) {
  char max_variables_count = 10;

  std::mt19937_64 bit_gen(seed);
  std::vector<std::string> operations = {"+", "-", "*", "/", "|", "&", "^"};
  std::vector<Variable> variables;

  char variables_count =
      absl::Uniform<char>(bit_gen, char{2}, max_variables_count + char{2});

  Variable var = Variable::GenerateInt(0, bit_gen);
  variables.reserve(variables_count);
  for (char i = 0; i < variables_count; i++) {
    variables.push_back(Variable::GenerateInt(i + 1, bit_gen));
  }
  std::stringstream content;
  content << "#ifndef __SYNTHESIS__\n";
  content << "#include <iostream>\n";
  content << "#endif\n";

  content << "#include \"ac_int.h\"\n";
  content << "#pragma hls_top\n";
  content << var.Type() << " FuzzTestInt" << seed << "() {\n";

  for (auto variable : variables) {
    content << variable.Declaration() << " = " << variable.value << ";\n";
  }

  for (int i = 0; i < variables.size(); i++) {
    std::string operation1 = xls::RandomChoice(operations, bit_gen);
    Variable var1 = xls::RandomChoice(variables, bit_gen);
    Variable var2 = xls::RandomChoice(variables, bit_gen);
    content << "auto var" << i << " = " << var1.name << " " << operation1 << " "
            << var2.name << ";\n";
    std::string operation2 = xls::RandomChoice(operations, bit_gen);
    Variable var3 = variables[i];
    Variable var4 = xls::RandomChoice(variables, bit_gen);
    content << var3.name << " = var" << i << " " << operation2 << " "
            << var4.name << ";\n";
  }
  content << var.Declaration() << " = ";
  Variable var3 = variables.back();
  std::string operation = xls::RandomChoice(operations, bit_gen);
  content << var3.name << " " << operation << " ";
  Variable final_var = xls::RandomChoice(
      absl::MakeConstSpan(variables).first(variables.size() - 1), bit_gen);
  content << final_var.name;

  content << ";\n";
  content << "#ifndef __SYNTHESIS__\n";
  content << "std::cout << " << var.name
          << ".to_string(AC_BIN, false, true);\n";
  content << "#endif\n";
  content << "return " << var.name << ";\n";
  content << "}\n";

  content << "int main() { FuzzTestInt" << seed << "(); return 0; }\n";
  return content.str();
}

std::string GenerateFixedTest(uint32_t seed) {
  char max_variables_count = 10;

  std::mt19937_64 bit_gen(seed);
  std::vector<std::string> operations = {"+", "-", "*", "/", "|", "&", "^"};
  std::vector<Variable> variables;

  char variables_count =
      absl::Uniform<char>(bit_gen, char{2}, max_variables_count + char{2});

  Variable var = Variable::GenerateFixed(0, bit_gen);
  variables.reserve(variables_count);
  for (char i = 0; i < variables_count; i++) {
    variables.push_back(Variable::GenerateFixed(i + 1, bit_gen));
  }
  std::stringstream content;
  content << "#ifndef __SYNTHESIS__\n";
  content << "#include <iostream>\n";
  content << "#endif\n";

  content << "#include \"ac_fixed.h\"\n";
  content << "#pragma hls_top\n";
  content << var.Type() << " FuzzTestFixed" << seed << "() {\n";

  for (auto variable : variables) {
    content << variable.Declaration() << " = " << variable.value << ";\n";
  }

  for (int i = 0; i < variables.size(); i++) {
    std::string operation1 = xls::RandomChoice(operations, bit_gen);
    Variable var1 = xls::RandomChoice(variables, bit_gen);
    Variable var2 = xls::RandomChoice(variables, bit_gen);
    content << "auto var" << i << " = " << var1.name << " " << operation1 << " "
            << var2.name << ";\n";
    std::string operation2 = xls::RandomChoice(operations, bit_gen);
    Variable var3 = variables[i];
    Variable var4 = xls::RandomChoice(variables, bit_gen);
    content << var3.name << " = var" << i << " " << operation2 << " "
            << var4.name << ";\n";
  }
  content << var.Declaration() << " = ";
  Variable var3 = variables.back();
  std::string operation = xls::RandomChoice(operations, bit_gen);
  content << var3.name << " " << operation << " ";
  Variable final_var = xls::RandomChoice(
      absl::MakeConstSpan(variables).first(variables.size() - 1), bit_gen);
  content << final_var.name;

  content << ";\n";
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
