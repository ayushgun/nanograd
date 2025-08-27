from graphviz import Digraph


class Scalar:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0

        # computes global gradient via chain rule: dL/dx = dL/dy * dy/dx
        # where
        # - dL/dy is output's global gradient
        # - dy/dx is current node's local gradient
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        """Add nanograd.Scalar to another value"""

        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, _children=(self, other), _op="+")

        def _bw():
            # y = x1 + x2
            # dy/dx1 = 1 * x1**0 + 0 = 1
            # dy/dx2 = 0 + 1 * x2**0 = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _bw
        return out

    def __mul__(self, other):
        """Multiply nanograd.Scalar by another value"""

        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, _children=(self, other), _op="*")

        def _bw():
            # y = x1 * x2
            # dy/dx1 = 1 * x1**0 * x2 = x2
            # dy/dx2 = x1 * 1 * x2**0 = x1
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _bw
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data / other.data, _children=(self, other), _op="/")

        def _bw():
            # y = x1 / x2
            # dy/dx1 = 1/x2
            # dy/dx2 = -x1/(x2**2)
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / other.data**2) * out.grad

        out._backward = _bw
        return out

    def __pow__(self, k):
        """Raise nanograd.Scalar to the kth power"""

        assert isinstance(k, (float, int))
        out = Scalar(self.data**k, _children=(self,), _op=f"**{k}")

        def _bw():
            # y = x^k
            # dy/dx = k * x**(k - 1)
            self.grad += (k * self.data ** (k - 1)) * out.grad

        out._backward = _bw
        return out

    def relu(self):
        """Apply ReLU activation function to nanograd.Scalar"""

        out = Scalar(max(self.data, 0.0), _children=(self,), _op="ReLU")

        def _bw():
            # y = ReLU(x)
            # dy/dx = 1 if x > 0, else 0 (choose 0 at x == 0 to avoid undefined slope)
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _bw
        return out

    def backward(self):
        """Recursively compute gradients of all nodes flowing to this nodes"""

        # topological ordering of all children in network
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # recurse down all inputs to this node and apply chain rule to get gradient
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def draw(self):
        def trace(root):
            nodes, edges = set(), set()

            def build(v):
                if v not in nodes:
                    nodes.add(v)
                    for child in v._prev:
                        edges.add((child, v))
                        build(child)

            build(root)
            return nodes, edges

        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
        nodes, edges = trace(self)

        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid,
                label=f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}",
                shape="record",
            )

            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot

    def is_leaf(self):
        return len(self._prev) == 0

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self / other

    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"
