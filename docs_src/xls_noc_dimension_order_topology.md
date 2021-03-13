# Dimension-Order Topology Types

## Overview

The dimension-order topology is a topology where the arrangement of the routers
are described by their location in a dimensional space. The topology has a
hierarchical structure where dimension _n_ is composed of structures from
dimension _n-1_. One or more endpoint nodes are connected to a router.

## k-ary n-cube

A _k_-ary _n_-cube topology consists of _k<sup>n</sup>_ routers arranged in
an _n_-dimensional cube with _k_ routers along each dimension. Each router is
assigned an _n_-digit radix-_k_ address {a<sub>n−1</sub>, ..., a<sub>0</sub>}
and is connected to all routers with addresses that differ by ±1(mod _k_) in
exactly one address digit. Each dimension is constructed using _k_ _k_-ary
(_n-1_)-cubes. The channels between the routers can be unidirectional or
bidirectional. In practice, the channels are bidirectional.

### Ring Topology

![drawing](./ring.png)
 \
**Figure Ring\_4ary\_1cube\_example. A ring topology with four routers, also
known as a 4-ary 1-cube topology.**

The ring topology is a special instance of the _k_-ary _n_-cube, where _n_ is
equal to one. Figure Ring\_4ary\_1cube\_example shows a ring topology with four
routers, also known as a 4-ary 1-cube topology. The dotted lines show the
connectivity between routers 0 and k-1 for a dimension.

###Symmetric Torus Topology

![drawing](./symmetric_torus.png)
 \
**Figure Symmetic\_Torus\_4ary\_2cube\_example. A symmetric torus topology with
four routers along each dimension, also known as a 4-ary 2-cube topology.**

The symmetric torus topology is a special instance of the _k_-ary _n_-cube,
where n is equal to two. Figure Symmetic\_Torus\_4ary\_2cube\_example shows a
symmetric torus topology with four routers along each dimension, also known as a
4-ary 2-cube topology. Router (3,0) has address 3 in a dimension and 0 in the
other dimension. The dotted lines show the connectivity between routers 0 and
k-1 for a dimension.

![drawing](./hierarchical_structure_dimension_order.png)
 \
**Figure 4ary\_2cube\_construction\_example. The construction of a symmetric
torus (4-ary 2-cube) topology using 4-ary 1-cubes.**

Figure 4ary\_2cube\_construction\_example shows the construction of a symmetric
torus (4-ary 2-cube) topology using 4-ary 1-cubes (ring topologies).

###multi-radix n-cube

A multi-radix _n_-cube is a generalized form of a _k_-ary _n_-cube topology
where the radix (number of router nodes) for each dimension is explicitly
stated. Each router is assigned an _n_-digit where the digit at index _i_ has
the radix of dimension _i_. A 2,3-ary 2-cube describes a grid-like topology with
a radix of 2 in one dimension and radix of 3 in the other dimension.

![drawing](./multi_radix_n_cube.png)
 \
**Figure 2,3-ary\_2cube\_example. A 2,3-ary 2-cube topology example.**

Figure 2,3-ary\_2cube\_example shows a 2,3-ary 2-cube topology example. The
dotted lines show the connectivity between router 0 and router
radix<sub>i </sub>- 1 for each dimension.

##k-ary n-mesh

A _k_-ary _n_-mesh is a _k_-ary _n_-cube topology with the connection from
address a<sub>k−1</sub> to address a<sub>0</sub> omitted in each dimension.

###Line

![drawing](./line.png)
 \
**Figure Line\_4ary\_1mesh\_example. A line topology with four routers, also
known as a 4-ary 1-mesh topology.**

The line topology is a special instance of the _k_-ary _n_-mesh, where _n_ is
equal to one. Figure Line\_4ary\_1mesh\_example shows a line topology with four
routers, also known as a 4-ary 1-mesh topology. Compared to the ring topology in
Figure Ring\_4ary\_1cube\_example, the connection from router 0 to router 3 is
omitted.

###Symmetric Mesh

![drawing](./symmetric_mesh.png)
 \
**Figure Symmetic\_Mesh\_4ary\_2mesh\_example. A symmetric mesh topology with
four routers along each dimension, also known as a 4-ary 2-mesh topology.**

The symmetric torus mesh is a special instance of the _k_-ary _n_-mesh, where
_n_ is equal to two. Figure Symmetic\_Mesh\_4ary\_2mesh\_example shows a
symmetric mesh topology with four routers along each dimension, also known as a
4-ary 2-mesh topology. Router (1,2) has address 1 in a dimension and 2 in the other
dimension. Compared to the torus topology in Figure torus\_4ary\_2cube\_example,
the connections between the pair of routers [(0,0), (0,3)], [(1,0), (1,3)],
[(2,0), (2,3)], [(3,0), (3,3)], [(0,0), (3,0)], [(0,1), (3,1)], [(0,2), (3,2)],
and [(0,3), (3,3)] are omitted.

###multi-radix n-mesh

Similar to the multi-radix _n_-cube, a multi-radix _n_-mesh is a generalized form of
a _k_-ary _n_-mesh topology where the radix (number of router nodes) for each
dimension is explicitly stated. Each router is assigned an _n_-digit where the
digit at index _i_ has the radix of dimension _i_. A 2,3-ary 2-mesh describes a
grid-like topology with a radix of 2 in one dimension and radix of 3 in the
other dimension.

![drawing](./multi_radix_n_mesh.png)
 \
**Figure 2,3-ary\_2mesh\_example. A 2,3-ary 2-mesh topology example.**

Figure 2,3-ary\_2mesh\_example shows a 2,3-ary 2-mesh topology example. Compared
to the multi-radix n-cube topology in Figure 2,3-ary\_2cube\_example, the
connections from the pair of routers [(0,0), (0,2)], [(1,0), (1,2)],
[(0,0), (1,0)], [(0,1), (1,1)] and [(0,2), (1,2)] are omitted.


##Cheat Sheet
*   A _k_-ary _n_-cube topology consists of _k<sup>n</sup>_ routers arranged in
an n-dimensional cube with k router nodes along each dimension. Each router is
assigned an _n_-digit radix-_k_ address {a<sub>n−1</sub>, ..., a<sub>0</sub>}
and is connected to all routers with addresses that differ by ±1 (mod k) in
exactly one address digit. Each dimension is constructed using _k_ _k_-ary
(_n_-1)-cubes where there are _k_ routers in the first dimension. 
*   A multi-radix n-cube is a generalized form of a _k_-ary _n_-cube topology
where the radix (number of routers) for each dimension is explicitly
stated. Each router is assigned an _n_-digit where the digit at index _i_ has
the radix of dimension _i_.
*   A _k_-ary _n_-mesh is a _k_-ary _n_-cube topology with the connection from
address a<sub>k−1</sub> to address a<sub>0</sub> omitted in each dimension.
*   Similar to the multi-radix _n_-cube, a multi-radix n-mesh is a generalized
form of a _k_-ary _n_-mesh topology where the radix (number of routers) for each
dimension is explicitly stated. Each router is assigned an _n_-digit where the
digit at index _i_ has the radix of dimension _i_.
*   Channel direction can be unidirectional or bidirectional.
*   Ring: k-ary 1-cube
*   Symmetric Torus: k-ary 2-cube
*   Line: k-ary 1-mesh
*   Symmetric Mesh: k-ary 2-mesh

## References

William James Dally and Brian Patrick Towles. 2004. Principles and Practices of
Interconnection Networks. Morgan Kaufmann Publishers Inc., San Francisco, CA,
USA.