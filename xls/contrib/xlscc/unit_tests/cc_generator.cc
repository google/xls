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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace xlscc {
namespace {

class Variable {
 public:
  std::string name;
  std::string type;
  std::string value;
  std::vector<std::string> templated_parameters;
  static Variable GenerateInt(char variable_number) {
    int bit_width = (std::rand() % 100) + 1;
    int max = bit_width >= 31 ? RAND_MAX
                              : static_cast<int>(std::pow(2, bit_width)) + 10;
    bool is_signed = ((std::rand() % 2) == 0);
    Variable var = {std::string(1, 'a' + variable_number),
                    "ac_int",
                    is_signed ? std::to_string((std::rand() % max) - (max / 2))
                              : std::to_string(std::rand() % max),
                    {std::to_string(bit_width), is_signed ? "true" : "false"}};
    return var;
  }
  static Variable GenerateFixed(char variable_number) {
    std::vector<std::string> ac_q_mode{
        "AC_TRN", "AC_RND", "AC_TRN_ZERO", "AC_RND_ZERO", "AC_RND_INF",
        "AC_RND_MIN_INF", "AC_RND_CONV", "AC_RND_CONV_ODD"};
    std::vector<std::string> ac_o_mode{"AC_WRAP", "AC_SAT", "AC_SAT_ZERO",
                                       "AC_SAT_SYM"};

    int bit_width = (std::rand() % 100) + 1;
    int max = bit_width >= 31 ? RAND_MAX
                              : static_cast<int>(std::pow(2, bit_width)) + 10;
    int integer_width = bit_width - (std::rand() % bit_width);
    bool is_signed = ((std::rand() % 2) == 0);
    std::string quantization = ac_q_mode[std::rand() % ac_q_mode.size()];
    std::string overflow = ac_o_mode[std::rand() % ac_o_mode.size()];
    Variable var = {
        std::string(1, 'a' + variable_number),
        "ac_fixed",
        is_signed ? std::to_string((std::rand() % max) - (max / 2))
                  : std::to_string(std::rand() % max),
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
  int max_variables_count = 10;

  std::srand(seed);
  std::vector<std::string> operations = {"+", "-", "*", "/", "|", "&", "^"};
  std::vector<Variable> variables;

  int8_t variables_count =
      static_cast<int8_t>(std::rand() % max_variables_count + 2);

  Variable var = Variable::GenerateInt(0);
  variables.reserve(variables_count);
  for (char i = 0; i < variables_count; i++) {
    variables.push_back(Variable::GenerateInt(i + 1));
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
    int operation = std::rand() % operations.size();
    Variable var1 = variables[std::rand() % variables.size()];
    Variable var2 = variables[std::rand() % variables.size()];
    content << "auto var" << i << " = " << var1.name << " "
            << operations[operation] << " " << var2.name << ";\n";
    operation = std::rand() % operations.size();
    Variable var3 = variables[i];
    Variable var4 = variables[std::rand() % variables.size()];
    content << var3.name << " = var" << i << " " << operations[operation] << " "
            << var4.name << ";\n";
  }
  content << var.Declaration() << " = ";
  Variable var3 = variables.back();
  int operation = std::rand() % operations.size();
  content << var3.name << " " << operations[operation] << " ";
  Variable final_var = variables[std::rand() % (variables.size() - 1)];
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
  int max_variables_count = 10;

  std::srand(seed);
  std::vector<std::string> operations = {"+", "-", "*", "/", "|", "&", "^"};
  std::vector<Variable> variables;

  int8_t variables_count =
      static_cast<int8_t>(std::rand() % max_variables_count + 2);

  Variable var = Variable::GenerateFixed(0);
  variables.reserve(variables_count);
  for (char i = 0; i < variables_count; i++) {
    variables.push_back(Variable::GenerateFixed(i + 1));
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
    int operation = std::rand() % operations.size();
    Variable var1 = variables[std::rand() % variables.size()];
    Variable var2 = variables[std::rand() % variables.size()];
    content << "auto var" << i << " = " << var1.name << " "
            << operations[operation] << " " << var2.name << ";\n";
    operation = std::rand() % operations.size();
    Variable var3 = variables[i];
    Variable var4 = variables[std::rand() % variables.size()];
    content << var3.name << " = var" << i << " " << operations[operation] << " "
            << var4.name << ";\n";
  }
  content << var.Declaration() << " = ";
  Variable var3 = variables.back();
  int operation = std::rand() % operations.size();
  content << var3.name << " " << operations[operation] << " ";
  Variable final_var = variables[std::rand() % (variables.size() - 1)];
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
