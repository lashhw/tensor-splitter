class NameScope:
    def __init__(self, existing, counter=0):
        self.existing = existing
        self.counter = counter

    @classmethod
    def from_existing(cls, existing):
        return cls(set(existing))

    def make(self, base):
        name = f"{base}_{self.counter}"
        self.counter += 1
        while name in self.existing:
            name = f"{base}_{self.counter}"
            self.counter += 1
        self.existing.add(name)
        return name
