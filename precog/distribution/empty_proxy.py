
import precog.interface as interface

class EmptyProxy(interface.ProxyDistribution):
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def log_prob(self, *args, **kwargs):
        return 0.0
