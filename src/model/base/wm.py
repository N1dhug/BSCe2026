class WM:
    def __init__(self):
        # Initialize variables
        raise NotImplementedError("Unimplemented constructor")

    def I(self):
        # Define interface
        raise NotImplementedError("Unimplemented input equation")

    def O(self):
        # Define interface
        raise NotImplementedError("Unimplemented output equation")

    def A(self):
        # Define interface
        raise NotImplementedError("Unimplemented available-energy equation")

    def dVdt(self):
        # Define interface
        raise NotImplementedError("Unimplemented volume equation")

    def deterministic_solver(self):
        # Define interface
        raise NotImplementedError("Unimplemented deterministic solver")

    def fokker_planc_solver(self):
        # Define interface
        raise NotImplementedError("Unimplemented stochastic Fokker-Planck solver")
