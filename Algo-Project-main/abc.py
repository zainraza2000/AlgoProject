import matplotlib.pyplot as plt
import numpy as np
import sys
from math import inf
import networkx as nx

n = int(input("enter number of nodes: "))
file_name = 'input'
file_name += str(n)
file_name += '.txt'

with open(file_name, 'r') as input10:
    f_contents = input10.readline()
    f_contents = input10.readline()
    f_contents = input10.readline()
    print(f_contents)
    nodes = int(f_contents)
    f_contents = input10.readline()
    nodelist = np.empty((nodes, 3))
    print("nodesss", nodelist[0])
    for i in range(nodes):
        f_contents = input10.readline()
        nodelist[i, :] = f_contents.split("\t")
    f_contents = input10.readline()
    print("nodelist", nodelist)  # nodes read
    adj_list = [[] for _ in range(nodes)]
    print(adj_list)

    f_contents = input10.readline()
    x = f_contents.split("\t")
    print(len(x))

    for node in range(nodes):
        for i in range(1, len(x), 4):
            if x[i] != '\n':
                destNode = int(x[i])
                weight = float(x[i+2])
                weight = weight/10000000
                adj_list[int(x[0])].append([destNode, weight])
        f_contents = input10.readline()
        x = f_contents.split("\t")
    f_contents = input10.readline()
    print(*adj_list, sep="\n")
    source = int(f_contents)
x_values = np.zeros(2, float)
y_values = np.zeros(2, float)
mark_values = np.zeros(2, int)
for i in range(0, len(adj_list)):
    x_values[0] = nodelist[i][1]
    y_values[0] = nodelist[i][2]
    for j in range(0, len(adj_list[i])):
        x_values[1] = nodelist[int(adj_list[i][j][0])][1]
        y_values[1] = nodelist[int(adj_list[i][j][0])][2]
        plt.plot(x_values, y_values, color='green', linewidth=2,
                 marker='o', markerfacecolor='yellow', markersize=20)
        plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                  shape='full', lw=0, length_includes_head=True, head_length=0.03, head_width=.01)
font_dict = {'family': 'serif',
             'color': 'darkred',
             'size': 8}
for node in range(nodes):
    plt.text(nodelist[node][1], nodelist[node][2], node, fontdict=font_dict)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Original Graph')

graph = [[0 for column in range(nodes)]
         for row in range(nodes)]

for i in range(0, len(adj_list)):
    for j in range(0, len(adj_list[i])):
        if graph[i][int(adj_list[i][j][0])] == 0:
            graph[i][int(adj_list[i][j][0])] = float(
                adj_list[i][j][1])
        else:
            if graph[i][int(adj_list[i][j][0])] > float(adj_list[i][j][1]):
                graph[i][int(adj_list[i][j][0])] = float(
                    adj_list[i][j][1])
print("herherhere")
print(*graph, sep="\n")

algo = int(input("1 => Prims\n2 => Kruskal\n3 => Dijkstra\n4 => Bellman Ford\n5 => Floyd Warshall Algorithm\n6 => Clustering Coefficient in Graph Theory\n7 => Boruvka Algorithm\nEnter value to execute Algorithm on Graph: "))
plt.show()
if algo == 1:
    class PrimsGraph():
        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for column in range(vertices)]
                          for row in range(vertices)]

        def PrimsprintMST(self, parent):
            font_dict2 = {'family': 'serif',
                          'color': 'blue',
                          'size': 8}
            plt.clf()
            print("parent", parent)
            # print ("Edge \tWeight")
            prims_total_cost = 0.0
            for i in range(1, self.V):
                # print (parent[i], "-", i, "\t", self.graph[i][ parent[i] ])
                x_values[0] = nodelist[parent[i]][1]
                x_values[1] = nodelist[i][1]
                y_values[0] = nodelist[parent[i]][2]
                y_values[1] = nodelist[i][2]
                prims_total_cost = prims_total_cost + \
                    float(self.graph[i][parent[i]])
                plt.plot(x_values, y_values, label=str(
                    self.graph[i][parent[i]]), linewidth=2, marker='o', markersize=20)
                plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                          shape='full', lw=0)
                plt.text((nodelist[parent[i]][1]+nodelist[i][1])/2, (nodelist[parent[i]]
                                                                     [2]+nodelist[i][2])/2, float(self.graph[i][parent[i]]), fontdict=font_dict2)
            font_dict = {'family': 'serif',
                         'color': 'black',
                         'size': 8}
            for node in range(nodes):
                plt.text(nodelist[node][1], nodelist[node]
                         [2], node, fontdict=font_dict)
            plt.xlabel(str("x-axis\nTotal Cost: " +
                       "{0:.2f}".format(prims_total_cost)))
            plt.ylabel('y - axis')
            plt.title('Prims MST Graph')
            plt.legend()
            plt.show()

        def PrimsminKey(self, key, mstSet):

            min = sys.maxsize
  # graph[u][v] is non zero only for adjacent vertices of m
            # mstSet[v] is false for vertices not yet included in MST
            # Update the key only if graph[u][v] is smaller than key[v]
            for v in range(self.V):
                if key[v] < min and mstSet[v] == False:
                    min = key[v]
                    min_index = v

            return min_index

        def primMST(self):

            key = [sys.maxsize] * self.V
            parent = [None] * self.V
            key[0] = 0
            mstSet = [False] * self.V

            parent[0] = -1

            for cout in range(self.V):

                u = self.PrimsminKey(key, mstSet)

                mstSet[u] = True

                for v in range(self.V):

                    if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u

            self.PrimsprintMST(parent)
    g = PrimsGraph(nodes)
    print(*g.graph, sep="\n")
    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            if g.graph[i][int(adj_list[i][j][0])] == 0:
                g.graph[i][int(adj_list[i][j][0])] = float(
                    adj_list[i][j][1])
                g.graph[int(adj_list[i][j][0])][i] = float(
                    adj_list[i][j][1])
            else:
                if g.graph[i][int(adj_list[i][j][0])] > float(adj_list[i][j][1]):
                    g.graph[i][int(adj_list[i][j][0])] = float(
                        adj_list[i][j][1])
                    g.graph[int(adj_list[i][j][0])][i] = float(
                        adj_list[i][j][1])
    print("herherhere")
    print(*g.graph, sep="\n")
    # for i in range(0, nodes):
    #     for j in range(0, nodes):
    #         if g.graph[i][j] != 0:
    #             g.graph[j][i] = g.graph[i][j]
    # print(*g.graph, sep="\n")
    g.primMST()
elif algo == 2:
    # Class to represent a graph
    class kruskalGraph:

        def __init__(self, vertices):
            self.V = vertices  # No. of vertices
            self.graph = []  # default dictionary
            # to store graph

        # function to add an edge to graph
        def kruskaladdEdge(self, u, v, w):
            self.graph.append([u, v, w])

        # A utility function to find set of an element i
        # (uses path compression technique)
        def kruskalfind(self, parent, i):
            if parent[i] == i:
                return i
            return self.kruskalfind(parent, parent[i])

        # A function that does union of two sets of x and y
        # (uses union by rank)
        def kruskalunion(self, parent, rank, x, y):
            # keep on adding vertices until cycle formed stop when len is V-1 of mst
            xroot = self.kruskalfind(parent, x)
            yroot = self.kruskalfind(parent, y)

            # Attach smaller rank tree under root of
            # high rank tree (Union by Rank)
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot

            # If ranks are same, then make one as root
            # and increment its rank by one
            else:
                parent[yroot] = xroot
                rank[xroot] += 1

        # The main function to construct MST using Kruskal's
        # algorithm
        def KruskalMST(self):

            result = []  # This will store the resultant MST

            i = 0  # An index variable, used for sorted edges
            e = 0  # An index variable, used for result[]

            # Step 1:  Sort all the edges in non-decreasing
            # order of their
            # weight.  If we are not allowed to change the
            # given graph, we can create a copy of graph
            self.graph = sorted(self.graph, key=lambda item: item[2])

            parent = []
            rank = []

            # Create V subsets with single elements
            for node in range(self.V):
                parent.append(node)
                rank.append(0)

            # Number of edges to be taken is equal to V-1
            while e < self.V - 1:

                # Step 2: Pick the smallest edge and increment
                # the index for next iteration
                u, v, w = self.graph[i]
                i = i + 1
                x = self.kruskalfind(parent, u)
                y = self.kruskalfind(parent, v)

                # If including this edge does't cause cycle,
                # include it in result and increment the index
                # of result for next edge
                if x != y:
                    e = e + 1
                    result.append([u, v, w])
                    self.kruskalunion(parent, rank, x, y)
                # Else discard the edge

            # print the contents of result[] to display the built MST
            # print ("Following are the edges in the constructed MST")
            plt.clf()
            kruskal_cost = 0.0
            font_dict2 = {'family': 'serif',
                          'color': 'blue',
                          'size': 8}
            for u, v, weight in result:
                # print str(u) + " -- " + str(v) + " == " + str(weight)
                # print ("%d -- %d == %d" % (u,v,weight))
                x_values[0] = nodelist[u][1]
                x_values[1] = nodelist[v][1]
                y_values[0] = nodelist[u][2]
                y_values[1] = nodelist[v][2]
                kruskal_cost = kruskal_cost + weight
                plt.plot(x_values, y_values, label=str(weight),
                         linewidth=2, marker='o', markersize=20)
                plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                          shape='full', lw=0)
                plt.text((nodelist[u][1]+nodelist[v][1])/2, (nodelist[u]
                                                                     [2]+nodelist[v][2])/2, weight, fontdict=font_dict2)
            font_dict = {'family': 'serif',
                         'color': 'black',
                         'size': 8}
            for node in range(nodes):
                plt.text(nodelist[node][1], nodelist[node]
                         [2], node, fontdict=font_dict)
            plt.xlabel(str("x-axis\nTotal Cost: " +
                       "{0:.2f}".format(kruskal_cost)))
            plt.ylabel('y - axis')
            plt.title('Kruskal MST Graph')
            plt.legend()
            plt.show()
    # Driver code
    g = kruskalGraph(nodes)
    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            g.kruskaladdEdge(
                i, int(adj_list[i][j][0]), adj_list[i][j][1])
    print(*g.graph, sep="\n")
    g.KruskalMST()
elif algo == 3:
    class dijkstraGraph():

        def __init__(self, vertices):
            self.V = vertices
            self.graph = [[0 for column in range(vertices)]
                          for row in range(vertices)]

        def dijkstraprintSolution(self, dist):
            print(*dist, sep="\n")
            # print ("Vertex \tDistance from Source")
            dijkstra_cost = 0.0
            font_dict2 = {'family': 'serif',
                          'color': 'blue',
                          'size': 8}
            for node in range(self.V):
                # print (node, "\t", "{0:.2f}".format(dist[node]))
                x_values[0] = nodelist[source][1]
                x_values[1] = nodelist[node][1]
                y_values[0] = nodelist[source][2]
                y_values[1] = nodelist[node][2]
                dijkstra_cost = dijkstra_cost + float(dist[node])
                plt.plot(x_values, y_values, label="{0:.2f}".format(
                    dist[node]), linewidth=1, marker='o', markersize=14)
                plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                          shape='full', lw=0, length_includes_head=True, head_length=0.04, head_width=.02)
                plt.text((nodelist[source][1]+nodelist[node][1])/2, (nodelist[source]
                                                                     [2]+nodelist[node][2])/2, "{:.2f}".format(float(dist[node])), fontdict=font_dict2)
            font_dict = {'family': 'serif',
                         'color': 'black',
                         'size': 8}
            for node in range(nodes):
                plt.text(nodelist[node][1], nodelist[node]
                         [2], node, fontdict=font_dict)
            plt.xlabel(str("x-axis\nTotal Cost: " +
                       "{0:.2f}".format(dijkstra_cost)))
            plt.ylabel('y - axis')
            plt.title('Dijkstra Graph')
            plt.legend()
            plt.show()
        # A utility function to find the vertex with
        # minimum distance value, from the set of vertices
        # not yet included in shortest path tree

        def dijkstraminDistance(self, dist, sptSet):

            # Initilaize minimum distance for next node
            min = sys.maxsize

            # Search not nearest vertex not in the
            # shortest path tree
            for v in range(self.V):
                if dist[v] < min and sptSet[v] == False:
                    min = dist[v]
                    min_index = v

            return min_index

        # Funtion that implements Dijkstra's single source
        # shortest path algorithm for a graph represented
        # using adjacency matrix representation
        def dijkstra(self, src):

            dist = [sys.maxsize] * self.V
            dist[src] = 0
            sptSet = [False] * self.V

            for cout in range(self.V):

                # Pick the minimum distance vertex from
                # the set of vertices not yet processed.
                # u is always equal to src in first iteration
                u = self.dijkstraminDistance(dist, sptSet)

                # Put the minimum distance vertex in the
                # shotest path tree
                sptSet[u] = True

                # Update dist value of the adjacent vertices
                # of the picked vertex only if the current
                # distance is greater than new distance and
                # the vertex in not in the shotest path tree
                for v in range(self.V):
                    if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]

            self.dijkstraprintSolution(dist)

    # Driver program
    g = dijkstraGraph(nodes)
    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            if g.graph[i][int(adj_list[i][j][0])] == 0:
                g.graph[i][int(adj_list[i][j][0])] = float(
                    adj_list[i][j][1])
                g.graph[int(adj_list[i][j][0])][i] = float(
                    adj_list[i][j][1])
            else:
                if g.graph[i][int(adj_list[i][j][0])] > float(adj_list[i][j][1]):
                    g.graph[i][int(adj_list[i][j][0])] = float(
                        adj_list[i][j][1])
                    g.graph[int(adj_list[i][j][0])][i] = float(
                        adj_list[i][j][1])
    print(*g.graph, sep="\n")
    g.dijkstra(source)
elif algo == 4:
    # Class to represent a graph
    class bellfordGraph:

        def __init__(self, vertices):
            self.V = vertices  # No. of vertices
            self.graph = []  # default dictionary to store graph

        # function to add an edge to graph
        def bellfordaddEdge(self, u, v, w):
            self.graph.append([u, v, w])

        # utility function used to print the solution
        def bellfordprintArr(self, dist):
            font_dict2 = {'family': 'serif',
                          'color': 'blue',
                          'size': 8}
            # print("Vertex   Distance from Source")
            bellford_cost = 0.0
            for i in range(self.V):
                # print("% d \t\t % d" % (i, dist[i]))
                x_values[0] = nodelist[source][1]
                x_values[1] = nodelist[i][1]
                y_values[0] = nodelist[source][2]
                y_values[1] = nodelist[i][2]
                bellford_cost = bellford_cost + float(dist[i])
                plt.plot(x_values, y_values, label="{0:.2f}".format(
                    dist[i]), linewidth=2, marker='o', markersize=20)
                plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                          shape='full', lw=0, length_includes_head=True, head_length=0.04, head_width=.02)
                plt.text((nodelist[source][1]+nodelist[i][1])/2, (nodelist[source]
                                                                  [2]+nodelist[i][2])/2, "{:.2f}".format(float(dist[i])), fontdict=font_dict2)
            font_dict = {'family': 'serif',
                         'color': 'black',
                         'size': 8}
            for node in range(nodes):
                plt.text(nodelist[node][1], nodelist[node]
                         [2], node, fontdict=font_dict)
            plt.xlabel(str("x-axis\nTotal Cost: " +
                       "{0:.2f}".format(bellford_cost)))
            plt.ylabel('y - axis')
            plt.title('Bellman Ford Graph')
            plt.legend()
            plt.show()
        # The main function that finds shortest distances from src to
        # all other vertices using Bellman-Ford algorithm.  The function
        # also detects negative weight cycle

        def BellmanFord(self, src):

            # Step 1: Initialize distances from src to all other vertices
            # as INFINITE
            dist = [float("Inf")] * self.V
            dist[src] = 0

            # Step 2: Relax all edges |V| - 1 times. A simple shortest
            # path from src to any other vertex can have at-most |V| - 1
            # edges
            for i in range(self.V - 1):
                # Update dist value and parent index of the adjacent vertices of
                # the picked vertex. Consider only those vertices which are still in
                # queue
                for u, v, w in self.graph:
                    if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w

            # Step 3: check for negative-weight cycles.  The above step
            # guarantees shortest distances if graph doesn't contain
            # negative weight cycle.  If we get a shorter path, then there
            # is a cycle.

            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    print("Graph contains negative weight cycle")
                    return

            # print all distance
            self.bellfordprintArr(dist)

    g = bellfordGraph(nodes)
    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            g.bellfordaddEdge(
                i, int(adj_list[i][j][0]), adj_list[i][j][1])
            g.bellfordaddEdge(
                int(adj_list[i][j][0]), i, adj_list[i][j][1])
    print(*g.graph, sep="\n")
    g.BellmanFord(source)
elif algo == 5:
    V = nodes
    INF = 99999

# Solves all pair shortest path
# via Floyd Warshall Algorithm

    def floydWarshall(graph):
        """ dist[][] will be the output 
        matrix that will finally
            have the shortest distances 
            between every pair of vertices """
        """ initializing the solution matrix 
        same as input graph matrix
        OR we can say that the initial 
        values of shortest distances
        are based on shortest paths considering no 
        intermediate vertices """

        dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

        """ Add all vertices one by one 
        to the set of intermediate
        vertices.
        ---> Before start of an iteration, 
        we have shortest distances
        between all pairs of vertices 
        such that the shortest
        distances consider only the 
        vertices in the set 
        {0, 1, 2, .. k-1} as intermediate vertices.
        ----> After the end of a 
        iteration, vertex no. k is
        added to the set of intermediate 
        vertices and the 
        set becomes {0, 1, 2, .. k}
        """
        for k in range(V):

            # pick all vertices as source one by one
            for i in range(V):

                # Pick all vertices as destination for the
                # above picked source
                for j in range(V):

                    # If vertex k is on the shortest path from
                    # i to j, then update the value of dist[i][j]
                    dist[i][j] = min(dist[i][j],
                                     dist[i][k] + dist[k][j]
                                     )
        floydprintArr(dist)
        printSolution(dist)

    # A utility function to print the solution
    def floydprintArr(dist):
        font_dict2 = {'family': 'serif',
                      'color': 'blue',
                      'size': 8}
        for i in range(0, nodes):
            for j in range(0, nodes):
                if(dist[i][j] != INF and dist[i][j] != 0):
                    print(dist[i][j])
                    x_values[0] = nodelist[i][1]
                    x_values[1] = nodelist[j][1]
                    y_values[0] = nodelist[i][2]
                    y_values[1] = nodelist[j][2]
                    print("xvalue:", nodelist[i][1], nodelist[j][1])
                    # print("x_values", x_values)
                    # print("y_values", y_values)
                    plt.plot(x_values, y_values, label=str(dist[i][j]),
                             linewidth=1, marker='o', markersize=20)
                    plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                              shape='full', lw=0, length_includes_head=True, head_length=0.02, head_width=.01)
                    plt.text((nodelist[i][1]+nodelist[j][1])/2, (nodelist[i]
                             [2]+nodelist[j][2])/2, dist[i][j], fontdict=font_dict2)
        font_dict = {'family': 'serif',
                     'color': 'black',
                     'size': 8}
        for node in range(nodes):
            plt.text(nodelist[node][1], nodelist[node]
                     [2], node, fontdict=font_dict)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('Floyd Warshall Graph')
        plt.show()

    def afloydprintArr(dist):
        # print("Vertex   Distance from Source")
        bellford_cost = 0.0
        for i in range(self.V):
            # print("% d \t\t % d" % (i, dist[i]))
            x_values[0] = nodelist[source][1]
            x_values[1] = nodelist[i][1]
            y_values[0] = nodelist[source][2]
            y_values[1] = nodelist[i][2]
            bellford_cost = bellford_cost + float(dist[i])
            plt.plot(x_values, y_values, label="{0:.2f}".format(
                dist[i]), linewidth=2, marker='o', markersize=20)
            plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                      shape='full', lw=0, length_includes_head=True, head_length=0.04, head_width=.02)
        font_dict = {'family': 'serif',
                     'color': 'black',
                     'size': 8}
        for node in range(nodes):
            plt.text(nodelist[node][1], nodelist[node]
                     [2], node, fontdict=font_dict)
        plt.xlabel(str("x-axis\nTotal Cost: " +
                   "{0:.2f}".format(bellford_cost)))
        plt.ylabel('y - axis')
        plt.title('Bellman Ford Graph')
        plt.legend()
        plt.show()

    def printSolution(dist):
        print("Following matrix shows the shortest distances\
        between every pair of vertices")
        print(*dist, sep="\n")
        # for i in range(V):
        #     for j in range(V):
        #         if(dist[i][j] == INF):
        #             print("%7s" % ("INF")),
        #         else:
        #             print("%7d\t" % (dist[i][j])),
        #         if j == V-1:
        #             print("")

    graph = [[0 for column in range(nodes)]
             for row in range(nodes)]
    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            if graph[i][int(adj_list[i][j][0])] == 0:
                graph[i][int(adj_list[i][j][0])] = float(
                    adj_list[i][j][1])
            else:
                if graph[i][int(adj_list[i][j][0])] > float(adj_list[i][j][1]):
                    graph[i][int(adj_list[i][j][0])] = float(
                        adj_list[i][j][1])
    for i in range(0, nodes):
        for j in range(0, nodes):
            if i == j:
                graph[i][j] = 0
            elif graph[i][j] == 0:
                graph[i][j] = INF
    print(*graph, sep="\n")
    floydWarshall(graph)
elif algo == 6:
    G = nx.Graph()
    for i in range(nodes):
        G.add_node(nodelist[i][0], pos=(nodelist[i][1], nodelist[i][2]))

    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            G.add_edge(i, int(adj_list[i][j][0]), weight=adj_list[i][j][1])
    weight = nx.get_edge_attributes(G, 'weight')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weight, horizontalalignment='center', bbox=dict(
        alpha=0), font_color='red', font_weight='heavy')
    nx.draw(G, pos, with_labels=1, font_color='yellow')
    plt.show()
    num = nx.average_clustering(G)
    print(num)
elif algo == 7:
    class Graph:
        # These are the four small functions used in main Boruvkas function
        # It does union of two sets of x and y with the help of rank
        def union(self, parent, rank, x, y):
            xroot = self.find(parent, x)
            yroot = self.find(parent, y)
            if rank[xroot] < rank[yroot]:
                parent[xroot] = yroot
            elif rank[xroot] > rank[yroot]:
                parent[yroot] = xroot
            else:
                parent[yroot] = xroot  # Make one as root and increment.
                rank[xroot] += 1

        def __init__(self, vertices):
            self.V = vertices
            self.graph = []  # default dictionary
            self.result = []
        # add an edge to the graph

        def addEdge(self, u, v, w):
            self.graph.append([u, v, w])
        # find set of an element i

        def find(self, parent, i):
            if parent[i] == i:
                return i
            return self.find(parent, parent[i])

        def printBoruvkaMST(self):
            plt.clf()
            font_dict2 = {'family': 'serif',
                          'color': 'blue',
                          'size': 8}
            boruvka_cost = 0.0
            for u, v, weight in self.result:
                # print str(u) + " -- " + str(v) + " == " + str(weight)
                # print ("%d -- %d == %d" % (u,v,weight))
                x_values[0] = nodelist[u][1]
                x_values[1] = nodelist[v][1]
                y_values[0] = nodelist[u][2]
                y_values[1] = nodelist[v][2]
                boruvka_cost = boruvka_cost + weight
                plt.plot(x_values, y_values, label=str(weight),
                         linewidth=2, marker='o', markersize=20)
                plt.arrow(x_values[0], y_values[0], x_values[1]-x_values[0], y_values[1]-y_values[0],
                          shape='full', lw=0)
                plt.text((nodelist[u][1]+nodelist[v][1])/2, (nodelist[u]
                                                             [2]+nodelist[v][2])/2, weight, fontdict=font_dict2)
            font_dict = {'family': 'serif',
                         'color': 'black',
                         'size': 8}
            for node in range(nodes):
                plt.text(nodelist[node][1], nodelist[node]
                         [2], node, fontdict=font_dict)
            plt.xlabel(str("x-axis\nTotal Cost: " +
                       "{0:.2f}".format(boruvka_cost)))
            plt.ylabel('y - axis')
            plt.title('Boruvka MST Graph')
            plt.legend()
            plt.show()
    # ***********************************************************************
        # constructing MST

        def boruvkaMST(self):
            parent = []
            rank = []
            cheapest = []
            numTrees = self.V
            MSTweight = 0
            for node in range(self.V):
                parent.append(node)
                rank.append(0)
                cheapest = [-1] * self.V
            # Keep combining components (or sets) until all
            # compnentes are not combined into single MST
            while numTrees > 1:
                for i in range(len(self.graph)):
                    u, v, w = self.graph[i]
                    set1 = self.find(parent, u)
                    set2 = self.find(parent, v)

                    if set1 != set2:
                        if cheapest[set1] == -1 or cheapest[set1][2] > w:
                            cheapest[set1] = [u, v, w]
                        if cheapest[set2] == -1 or cheapest[set2][2] > w:
                            cheapest[set2] = [u, v, w]
                # Consider the above picked cheapest edges and add them to MST
                for node in range(self.V):
                    if cheapest[node] != -1:
                        u, v, w = cheapest[node]
                        set1 = self.find(parent, u)
                        set2 = self.find(parent, v)
                        if set1 != set2:
                            MSTweight += w
                            self.union(parent, rank, set1, set2)
                            print(
                                "Edge %d-%d has weight %d is included in MST" % (u, v, w))
                            self.result.append([u, v, w])
                            numTrees = numTrees - 1

                cheapest = [-1] * self.V
            print("Weight of MST is %d" % MSTweight)
            self.printBoruvkaMST()

    g = Graph(nodes)
    for i in range(0, len(adj_list)):
        for j in range(0, len(adj_list[i])):
            g.addEdge(
                i, int(adj_list[i][j][0]), adj_list[i][j][1])
    print(*g.graph, sep="\n")
    g.boruvkaMST()
    print(*g.result, sep="\n")
else:
    print("Option Not Valid")
