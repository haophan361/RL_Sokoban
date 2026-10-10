class State:
    def __init__(self, matrix):
        self.matrix = matrix
        self.pathSolution = ""
        self.docks_position = self.get_docks_position()
        self.heuristic = 0

    def __lt__(self,other):
        return self.heuristic < other.heuristic

    def get_matrix(self):
        return self.matrix

    def get_matrix_element(self, y, x):
        return self.matrix[y][x]

    def set_matrix_element(self, y, x, object_map):
        self.matrix[y][x] = object_map

    def get_element_next_step(self, y, x):
        new_y, new_x = self.get_worker_position()[0] + y, self.get_worker_position()[1] + x
        return self.get_matrix_element(new_y, new_x)

    def get_worker_position(self):
        for y, row in enumerate(self.matrix):
            for x, char in enumerate(row):
                if char == '@':
                    return y, x

    def get_boxes_position(self):
        boxes_position = []
        for y, row in enumerate(self.matrix):
            for x, char in enumerate(row):
                if char in ['$','*']:
                    boxes_position.append((y, x))
        return boxes_position

    def get_docks_position(self):
        dockListPosition = []
        for y, row in enumerate(self.matrix):
            for x, char in enumerate(row):
                if char == '.':
                    dockListPosition.append((y, x))
        return dockListPosition
    
    def worker_can_move(self, y, x):
        return self.get_element_next_step(y, x) in [' ', '.']

    def worker_can_push_box(self, y, x):
        return self.get_element_next_step(y, x) in ['*', '$'] and self.get_element_next_step(y + y, x + x) in ['.', ' ']

    def is_complete(self):
        boxes = set(self.get_boxes_position())
        goals = set(self.docks_position)

        return boxes == goals
        
    def move_box(self, y_box, x_box, move_y, move_x):
        box_element = self.get_matrix_element(y_box, x_box)
        next_box_element = self.get_matrix_element(y_box + move_y, x_box + move_x)
        if box_element == '$':
            if next_box_element == ' ':
                self.set_matrix_element(y_box, x_box, ' ')
                self.set_matrix_element(y_box + move_y, x_box + move_x, '$')
            elif next_box_element == '.':
                self.set_matrix_element(y_box, x_box, ' ')
                self.set_matrix_element(y_box + move_y, x_box + move_x, '*')
        elif box_element == '*':
            if next_box_element == ' ':
                self.set_matrix_element(y_box, x_box, '.')
                self.set_matrix_element(y_box + move_y, x_box + move_x, '$')
            elif next_box_element == '.':
                self.set_matrix_element(y_box, x_box, '.')
                self.set_matrix_element(y_box + move_y, x_box + move_x, '*')

    def move(self, y, x):
        if self.worker_can_move(y, x):
            worker_position = self.get_worker_position()
            self.set_matrix_element(worker_position[0] + y, worker_position[1] + x, '@')
            self.set_matrix_element(worker_position[0], worker_position[1], ' ')
        elif self.worker_can_push_box(y, x):
            worker_position = self.get_worker_position()
            self.move_box(worker_position[0] + y, worker_position[1] + x, y, x)
            self.set_matrix_element(worker_position[0] + y, worker_position[1] + x, '@')
            self.set_matrix_element(worker_position[0], worker_position[1], ' ')

        for i, j in self.docks_position:
            if self.get_matrix_element(i, j) not in ['*', '@']:
                self.set_matrix_element(i, j, '.')

    def print_state(self):
        for row in self.matrix:
            print(" ".join(row))
        print()



