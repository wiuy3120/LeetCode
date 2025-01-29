class DisjointSet:
    def __init__(self, N):
        # Initialize DSU class, size of each component will be one and each node
        # will be representative of its own.
        self.N = N
        self.size = [1] * N
        self.representative = list(range(N))

    def find(self, node):
        # Returns the ultimate representative of the node.
        if self.representative[node] == node:
            return node
        self.representative[node] = self.find(self.representative[node])
        return self.representative[node]

    def union(self, nodeOne, nodeTwo):
        # Returns true if node nodeOne and nodeTwo belong to different component
        # and update the representatives accordingly, otherwise returns false.
        nodeOne = self.find(nodeOne)
        nodeTwo = self.find(nodeTwo)

        if nodeOne == nodeTwo:
            return False
        else:
            if self.size[nodeOne] > self.size[nodeTwo]:
                self.representative[nodeTwo] = nodeOne
                self.size[nodeOne] += self.size[nodeTwo]
            else:
                self.representative[nodeOne] = nodeTwo
                self.size[nodeTwo] += self.size[nodeOne]
            return True
