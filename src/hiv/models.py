import sympy as sym


class GenericVirusModel:
    def __init__(self: object, **args) -> None:
        for p in ["r", "p", "q", "c", "k", "b"]:
            if p not in args:
                raise Exception(f"Missing Parameter '{p}'")
        self.args = args

    def v(self: object, *args) -> float:
        x, v, z, = args
        return - v * (self.args["r"] -
                      (self.args["p"] * x) -
                      (self.args["q"] * z))

    def x(self: object, *args) -> float:
        x, v, _, _ = args
        return - self.args["c"] * v - self.args["b"] * x

    def z(self: object, *args) -> float:
        _, _, z, svi = args
        return self.args["k"] * svi - self.args["b"] * z

    @staticmethod
    def solve():
        xi, vi, z = sym.Symbol("xi"), sym.Symbol("vi"), sym.Symbol("z")
        r, p, q = sym.Symbol("r"), sym.Symbol("p"), sym.Symbol("q")
        c, b, k = sym.Symbol("c"), sym.Symbol("b"), sym.Symbol("k")
        v, u = sym.Symbol("v")

        return (sym.solve(-vi * (r - (p * xi) - (q * z))),
                sym.solve(- (c * vi) - (b * xi)),
                sym.solve((k * v) - (b * z), z))

    def __call__(self: object, *args) -> tuple[float, float, float]:
        return self.x(args), self.v(args), self.z(args)

    def __repr__(self: object) -> float:
        return f"{self.__class__.__name__}({self.args})"


class HIVModel(GenericVirusModel):
    def __init__(self: object, **args) -> None:
        super(HIVModel, self).__init__(**args)
        if "u" not in args:
            raise Exception("Missing Parameter 'u'")
        self.args = args

    def x(self: object, *args) -> float:
        x, v, _, svi = args
        return - self.args["c"] * v \
               - self.args["b"] * x \
               - self.args["u"] * svi * x

    def z(self: object, *args) -> float:
        _, _, z, svi = args
        return self.args["k"] * svi \
            - self.args["b"] * z \
            - self.args["u"] * svi * z

    @staticmethod
    def solve():
        xi, vi, z = sym.Symbol("xi"), sym.Symbol("vi"), sym.Symbol("z")
        r, p, q = sym.Symbol("r"), sym.Symbol("p"), sym.Symbol("q")
        c, b, k = sym.Symbol("c"), sym.Symbol("b"), sym.Symbol("k")
        v, u = sym.Symbol("v"), sym.Symbol("u")

        return (sym.solve(- vi * (r - (p * xi) - (q * z))),
                sym.solve(- (c * vi) - (b * xi) - (u * v * xi)),
                sym.solve((k * v) - (b * z) - (u * v * z)))
