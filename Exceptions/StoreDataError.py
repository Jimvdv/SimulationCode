class StoreDataError(Exception):
    def __init__(self, message, unbounded_particle=None):
        self.message = message
        self.unbounded_particle = unbounded_particle
        super().__init__(self.message)

