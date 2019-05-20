from igraph import *


edges = [['a', 'b'],
         ['a', 'b'],
         ['a', 'b'],
         ['b', 'a'],
         ['a', 'c'],
         ['c', 'a'],
         ['c', 'd'],
         ['c', 'd'],
         ['d', 'c'],
         ['d', 'c']]

# collect the set of vertex names and then sort them into a list
vertices = set()
for line in edges:
    vertices.update(line)
vertices = sorted(vertices)

# create an empty graph
g = Graph()

# add vertices to the graph
g.add_vertices(vertices)

# add edges to the graph
g.add_edges(edges)

# set the weight of every edge to 1
g.es["weight"] = 1

# collapse multiple edges and sum their weights
g.simplify(combine_edges={"weight": "sum"})

# plot(g)

g2 = Graph.TupleList(directed=False, edges = edges)

plot(g2)
