
class GenericVirusModel:
    def __init__(self, **args):
        for p in ["r", "p", "q", "c", "k", "b"]:
            if p not in args:
                raise Exception(f"Missing Parameter '{p}'")
        self.args = args

    def v(self, *args):
        x, v, z = args
        return - v * self.args["r"] - self.args["p"] * x - self.args["q"] * z

    def x(self, *args):
        x, v, _, _ = args
        return - self.args["c"] * v - self.args["b"] * x

    def z(self, *args):
        _, _, z, svi = args
        return self.args["k"] * svi - self.args["b"] * z

    def __call__(self, *args):
        return self.x(args), self.v(args), self.z(args)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.args})"


class HIVModel(GenericVirusModel):
    def __init__(self, **args):
        super(HIVModel, self).__init__(**args)
        if "u" not in args:
            raise Exception("Missing Parameter 'u'")
        self.args = args

    def x(self, *args):
        x, v, _, svi = args
        return - self.args["c"] * v \
               - self.args["b"] * x \
               - self.args["u"] * svi * x

    def z(self, *args):
        _, _, z, svi = args
        return self.args["k"] * svi \
            - self.args["b"] * z \
            - self.args["u"] * svi * z


if __name__ == "__main__":
    gvm = GenericVirusModel(r=2, p=3, q=2, c=3, k=0, b=2)
    print(gvm)
    hiv = HIVModel(r=2, p=3, q=2, c=3, k=0, u=2, b=2)
    print(hiv)