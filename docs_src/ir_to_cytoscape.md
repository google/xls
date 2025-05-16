# Cytoscape IR Export

For IRs which are too large to render using the
[ir_viz web app](ir_visualization.md) one may use [cytoscape](https://cytoscape.org) as
a graph viewer. This is a much more manual experience than the web app but can
handle IRs of arbitrary size.

Consult the
[Cytoscape user manual](https://manual.cytoscape.org/en/stable/Introduction.html)
for information on how to use cytoscape.

## Generating the Cytoscape Data

To convert the IR into a form cytoscape can understand run:

```shell
bazel run //xls/visualization/ir_viz:ir_to_cytoscape -- --delay_model=unit --output=/path/to/output.json /path/to/design.ir
```

## Loading Cytoscape

Follow the general directions at
[cytoscape.org to import a network file](https://manual.cytoscape.org/en/stable/Creating_Networks.html#import-fixed-format-network-files).
Note that by default all nodes will be placed on top of one another so you need
to use the `Layout` menu to make cytoscape place all the nodes. Use the `Styles`
tab to adjust how the graph is displayed. A basic style can be found
[here](https://github.com/google/xls/raw/main/docs_src/cytoscape-ir-style.xml).
