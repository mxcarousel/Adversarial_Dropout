from abc import ABC, abstractmethod

class drop_out(ABC):
    def __init__(self,threshold,alpha,stat):
        self.active_clients = []
        self.deter_clients = [] 
        self.alpha = alpha
        self.stat = stat
        self.threshold = threshold
        super().__init__()

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def drop(self):
        pass

