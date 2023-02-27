from utils import *
from .__init__ import drop_out

class forever_scheme(drop_out):
    def prepare(self):
        self.deter_clients = drop_out_forever(self.alpha,self.stat)    
    
    def drop(self, active_clients, selected_clients, round):
        if round > self.threshold:
            selected_clients = drop_out_deter(active_clients, self.deter_clients)
        return selected_clients
