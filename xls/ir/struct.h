// Copyright 2026 The XLS Authors
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

#ifndef XLS_IR_STRUCT_H_
#define XLS_IR_STRUCT_H_

#include <string>
#include <vector>

#include "xls/ir/ir_annotator.h"
#include "xls/ir/type.h"

namespace xls {

class StructDef {
 public:
  StructDef(std::string name)
    : name_(std::move(name)) {}

  void AddMember(const Type* type, std::string name);

  std::string DumpIr() const;

  struct Member {
    const Type* type = nullptr;
    std::string name;
  };

  std::string name() { return name_; }
  const std::vector<Member>& members() { return members_; }

 private:
  std::string name_;
  std::vector<Member> members_;
};

}  // namespace xls

#endif  // XLS_IR_STRUCT_H_
