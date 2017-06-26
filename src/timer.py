import time
from copy import deepcopy

class Timer:
    def __init__(self, **kwargs):
        self.events = []
        self.debug = False
        self.start = None
        if 'debug' in kwargs:
            self.debug = kwargs['debug']


    def start(self, event):
        if self.start is not None:
            self.end()
        self.event = event
        self.start = time()

        if kwargs['debug']:
            print("Start: %s"%(self.event)), flush=True)

    def end(self):
        self.end = time()
        entry = {
            "name": self.event,
            "start": self.start,
            "end": self.end,
            "duration": self.end - self.start + 1
        }

        if kwargs['debug']:
            print("End: %s"%(self.event)), flush=True)

        self.events.append(entry)

    def export(self):
        return self.events

