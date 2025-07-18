class SimulationNotFoundError(FileNotFoundError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)