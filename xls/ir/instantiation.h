// Copyright 2021 The XLS Authors
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

#ifndef XLS_IR_INSTANTIATION_H_
#define XLS_IR_INSTANTIATION_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/ir/type.h"

namespace xls {

class Block;

enum class InstantiationKind {
  // Instantiation of an IR block in the same package.
  kBlock,

  // Instantiation of an abstract FIFO.
  kFifo,

  // Instantiation of an externally defined Verilog module.
  kExtern,
};

std::string InstantiationKindToString(InstantiationKind kind);
absl::StatusOr<InstantiationKind> StringToInstantiationKind(
    std::string_view str);
std::ostream& operator<<(std::ostream& os, InstantiationKind kind);

struct InstantiationPort {
  std::string name;
  Type* type;
};

// Base class for an instantiation which is a block-scoped construct that
// represents a module instantiation at the Verilog level. The instantiated
// object can be another block, a FIFO (not yet supported), or a externally
// defined Verilog module (not yet supported).
class Instantiation {
 public:
  Instantiation(std::string_view name, InstantiationKind kind)
      : name_(name), kind_(kind) {}
  virtual ~Instantiation() = default;

  const std::string& name() const { return name_; }

  InstantiationKind kind() const { return kind_; }

  virtual std::string ToString() const = 0;

  virtual absl::StatusOr<InstantiationPort> GetInputPort(
      std::string_view name) = 0;
  virtual absl::StatusOr<InstantiationPort> GetOutputPort(
      std::string_view name) = 0;

 protected:
  std::string name_;
  InstantiationKind kind_;
};

// Abstraction representing the instantiation of an IR block.
class BlockInstantiation : public Instantiation {
 public:
  BlockInstantiation(std::string_view name, Block* instantiated_block)
      : Instantiation(name, InstantiationKind::kBlock),
        instantiated_block_(instantiated_block) {}

  Block* instantiated_block() const { return instantiated_block_; }

  std::string ToString() const override;

  absl::StatusOr<InstantiationPort> GetInputPort(
      std::string_view name) override;
  absl::StatusOr<InstantiationPort> GetOutputPort(
      std::string_view name) override;

 private:
  Block* instantiated_block_;
};

}  // namespace xls

#endif  // XLS_IR_INSTANTIATION_H_
