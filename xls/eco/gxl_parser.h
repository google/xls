#ifndef XLS_ECO_GXL_PARSER_H_
#define XLS_ECO_GXL_PARSER_H_

#include "xls/eco/graph.h"
#include "tinyxml2.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

XLSGraph parse_gxl(const std::string &filename);
bool export_gxl(const XLSGraph &graph, const std::string &filename);

#endif  // XLS_ECO_GXL_PARSER_H_
