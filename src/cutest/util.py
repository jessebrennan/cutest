

class Stack(list):

    def top(self):
        return self[-1]

    def empty(self) -> bool:
        return len(self) == 0

    def add(self, item):
        return self.append(item)
