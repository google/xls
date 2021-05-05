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

#include "Eigen/Core"

int main() {
  // Just a soundness check to see if the library is properly imported. This
  // helps because Eigen is a header-only library.
  Eigen::MatrixXd matrix(2, 2);
  matrix << 0, 1, 2, 3;
  return matrix(0, 0);
}
