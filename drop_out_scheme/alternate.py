from .__init__ import drop_out
from utils import *

class alternate_scheme(drop_out):
    def __init__(self, threshold, alpha,stat,repetition,interval):
        super().__init__(threshold, alpha,stat)
        self.repetition = repetition
        self.interval = interval
        self.threshold = threshold
    
    def prepare(self):
        self.worker_slice = drop_out_alternating_slice(self.repetition, self.stat)
        self.repetition = len(self.worker_slice)

    def drop(self,active_clients, selected_clients, round):
        if round > self.threshold:
            selected_clients = drop_out_alternating(active_clients, self.interval, self.repetition, round, self.worker_slice)
        return selected_clients
        