class Subject():
    def __init__(self):
        self.observers = []

    def attach_observer(self, observer):
        self.observers.append(observer)

    def dettach_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self, **kwargs):
        for observer in self.observers:
            observer.update(kwargs)


        