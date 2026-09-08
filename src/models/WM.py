class WM:
    def __init__(self):
        raise NotImplementedError("Unimplemented constructor")

    def It(self):
        raise NotImplementedError("Unimplemented input equation")

    def Ot(self):
        raise NotImplementedError("Unimplemented output equation")

    def At(self):
        raise NotImplementedError("Unimplemented available-energy equation")

    def dVdt(self):
        raise NotImplementedError("Unimplemented volume equation")

    def deterministic_solver(self):
        raise NotImplementedError("Unimplemented deterministic solver")

    def fokker_planc_solver(self):
        raise NotImplementedError("Unimplemented stochastic Fokker-Planck solver")
