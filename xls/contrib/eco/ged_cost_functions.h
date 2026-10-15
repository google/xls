// Copyright 2025 The XLS Authors
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

#ifndef XLS_ECO_GED_COST_FUNCTIONS_H_
#define XLS_ECO_GED_COST_FUNCTIONS_H_

#include "xls/contrib/eco/ged.h"
#include "xls/contrib/eco/graph.h"

int NodeSubstCost(const XLSNode& n1, const XLSNode& n2);
int NodeInsCost(const XLSNode& node);
int NodeDelCost(const XLSNode& node);
int EdgeSubstCost(const XLSEdge& e1, const XLSEdge& e2);
int EdgeInsCost(const XLSEdge& e);
int EdgeDelCost(const XLSEdge& e);
ged::GEDOptions CreateUserCosts();

#endif  // XLS_ECO_GED_COST_FUNCTIONS_H_
