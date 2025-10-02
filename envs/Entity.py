
class Entity:
    def __init__(self, name: str):
        self.name = name
        self.color = None
        self.x, self.y = None, None
        self.size = 0.5

    @property
    def p_pos(self):
        return (self.x, self.y)

    def set_position(self, x, y):
        self.x, self.y = x, y