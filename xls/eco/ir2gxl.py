#!/usr/bin/env python3

# TODO(b/1234567890): DEPRECATED - This GXL export tool will be replaced by
# C++ GXL parser in gxl_parser.h/cc which provides:
# - Bidirectional GXL parsing (not just export)
# - Better XML handling through tinyxml2
# - Integration with C++ GED implementation
# Please migrate to using the C++ GXL tools in //xls/eco.

"""IR to GXL conversion library for XLS ECO.

This module provides functionality to convert XLS IR files to GXL format
for use with the Java GED implementation.

TODO(eco): Deprecate this Python-based IR-to-GXL converter in favor of a
C++ implementation based on xls/visualization/ir_viz/xls_ir_to_cytoscape.cc.
The C++ approach would be more robust (using actual IR APIs instead of regex
parsing), faster, and better maintained. Consider extending the C++ cytoscape
tool to support GXL output format, or create a C++ GXL exporter that shares
the same IR graph construction logic.
"""

import sys
import os
from pathlib import Path
from typing import Optional
import networkx as nx

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xls.eco import ir2nx


class IrParser:
    """This class parses XLS intermediate representation (IR) into GXL format.

    Similar interface to ir2nx.IrParser for consistency.
    """

    def __init__(self, path=None, graph=None):
        """Initializes the IrParser object.

        Args:
            path: The path to the IR file (str or Path object). If None, graph must be provided.
            graph: NetworkX graph to convert. If None, path must be provided.
        """
        if path is not None:
            self._path = Path(path)
            self._ir_parser = ir2nx.IrParser(self._path)
            self._nx_graph = self._ir_parser.graph
        elif graph is not None:
            self._path = None
            self._nx_graph = graph
        else:
            raise ValueError("Either path or graph must be provided")

        self.gxl = self._convert_to_gxl()

    def _convert_to_gxl(self) -> str:
        """Convert the NetworkX graph to GXL format."""
        wrapper = _NetworkXGraphWrapper(self._nx_graph)
        return wrapper.to_gxl_string()


class _NetworkXGraphWrapper:
    """Internal wrapper for converting NetworkX graphs to GXL format."""

    def __init__(self, nx_graph):
        self.graph = nx_graph
        self._convert_to_internal_format()

    def _convert_to_internal_format(self):
        """Convert NetworkX graph to internal format."""
        self.nodes = {}
        self.edges = []
        self.return_node = self.graph.graph.get("ret")

        for node_id, node_data in self.graph.nodes(data=True):
            self.nodes[str(node_id)] = {
                "op": node_data.get("op", "unknown"),
                "cost_attributes": node_data.get("cost_attributes", {}),
            }

        # Preserve multigraph edge keys so operand ordering survives.
        for u, v, key, edge_data in self.graph.edges(keys=True, data=True):
            self.edges.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "key": key,
                    "cost_attributes": edge_data.get("cost_attributes", {}),
                }
            )

    def _serialize_cost_attributes(self, cost_attrs):
        """Serialize cost attributes to a parseable string format."""
        if not cost_attrs:
            return ""

        parts = []
        for key, value in cost_attrs.items():
            if isinstance(value, (list, tuple)):
                # Convert lists/tuples to comma-separated strings
                value_str = ",".join(str(v) for v in value)
                parts.append(f"{key}={value_str}")
            elif isinstance(value, bool):
                # Convert booleans to "true"/"false"
                parts.append(f"{key}={str(value).lower()}")
            else:
                # Simple string/number values
                parts.append(f"{key}={value}")

        return "|".join(parts)  # Use | as separator between attributes

    def to_gxl_string(self) -> str:
        """Convert to GXL string format."""
        gxl_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<gxl>
  <graph id="graph" edgeids="true" edgemode="directed">
"""
        if self.return_node is not None:
            gxl_content += f"""    <attr name="ret">
      <string>{self.return_node}</string>
    </attr>
"""

        for node_id, node_data in self.nodes.items():
            cost_attr_str = self._serialize_cost_attributes(
                node_data["cost_attributes"]
            )
            gxl_content += f"""    <node id="{node_id}">
      <attr name="op">
        <string>{node_data['op']}</string>
      </attr>
      <attr name="cost_attributes">
        <string>{cost_attr_str}</string>
      </attr>
    </node>
"""

        # Emit edges and include operand index info when available.
        for i, edge in enumerate(self.edges):
            cost_attr_str = self._serialize_cost_attributes(edge["cost_attributes"])
            gxl_content += f"""    <edge id="e{i}" from="{edge['source']}" to="{edge['target']}">
      <attr name="cost_attributes">
        <string>{cost_attr_str}</string>
      </attr>
"""
            if edge.get("key") is not None:
                gxl_content += f"""      <attr name="index">
        <string>{edge['key']}</string>
      </attr>
"""
            gxl_content += "    </edge>\n"

        gxl_content += """  </graph>
</gxl>"""
        return gxl_content


def ir_to_gxl(ir_file_path: str) -> str:
    """Convenience function to convert IR file to GXL string.

    Args:
        ir_file_path: Path to the IR file

    Returns:
        GXL string representation
    """
    parser = IrParser(ir_file_path)
    return parser.gxl


# Removed nx_to_gxl function - use IrParser(graph=nx_graph).gxl instead


def main():
    """Command-line interface for IR to GXL conversion."""
    if len(sys.argv) != 2:
        print("Usage: python ir2gxl.py <ir_file_path>")
        sys.exit(1)

    ir_file_path = sys.argv[1]
    if not os.path.exists(ir_file_path):
        print(f"Error: IR file not found: {ir_file_path}")
        sys.exit(1)

    try:
        # Use the same pattern as ir2nx
        parser = IrParser(ir_file_path)
        print(parser.gxl)
    except Exception as e:
        print(f"Error converting IR to GXL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
