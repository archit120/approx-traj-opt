import numpy as np

class ReplayBuffer(object):

    def __init__(self, max_size=1000000):
        self.buffer = []
        self.max_size = max_size
        self.i = 0
    
    def add_transition(self, transition):
        if len(self.buffer)<self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[i] = transition
            self.i+=1
            self.i = (self.i)%self.max_size
    
    def add_transitions(self, transitions):
        for transition in transitions:
            self.add_transition(transition)

    def sample_transitions(self, batch_size):
        rand_indices = np.random.permutation(len(self.buffer))[:batch_size]
        return self.buffer[rand_indices]
    
    def sample_transition(self):
        rand_index = np.random.randint(0, len(self.buffer))
        return self.buffer[rand_index]