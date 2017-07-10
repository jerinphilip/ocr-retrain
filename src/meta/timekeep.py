from time import time
from copy import deepcopy

class Timer:
    def __init__(self, **kwargs):
        self.events = []
        self.debug = False
        self.start_time = None
        if 'debug' in kwargs:
            self.debug = kwargs['debug']

    def start(self, event):
        if self.start_time is not None:
            self.end()
        self.event = event
        self.start_time = time()

        if self.debug:
            print("Start: %s"%(self.event), flush=True)

    def end(self):
        self.end_time = time()
        entry = {
            "name": self.event,
            "start": self.start_time,
            "end": self.end_time,
            "duration": self.end_time - self.start_time + 1
        }

        if self.debug:
            print("End: %s"%(self.event), flush=True)

        self.events.append(entry)

    def export(self):
        return self.events

