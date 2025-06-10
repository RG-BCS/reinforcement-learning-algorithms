class GridWorld:
    def __init__(self, size, goal, obstacles):
        self.grid_size = size
        self.goal = goal
        self.obstacles = set(obstacles)

    def is_valid(self, state):
        r, c = state
        if not (0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]):
            return False
        return state not in self.obstacles

    def step(self, s, move):
        next_state = (s[0] + move[0], s[1] + move[1])
        if not self.is_valid(next_state):
            return s
        return next_state
