from deadlock_detector.corner_deadlock import is_corner_deadlock
from deadlock_detector.freeze_deadlock import is_freeze_deadlock
from state import State
from collections import deque
import copy
from push_action import PushAction
import numpy as np

class SokobanEnv:
    ACTIONS = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    def __init__(self, map_matrix, dead_squares, max_pushes, avoid_dead_squares=False):

        self.max_pushes = max_pushes
        self.avoid_dead_squares = avoid_dead_squares
        
        self.set_level(map_matrix, dead_squares)

    def set_level(self,map_matrix, dead_squares):
        self.map_matrix = copy.deepcopy(map_matrix)

        self.dead_squares = {tuple(pos)for pos in dead_squares}

        self.height= len(self.map_matrix)
        self.width= len(self.map_matrix[0])

        initial_state = State(self.map_matrix)

        self.goals = set(initial_state.docks_position)

        self.num_actions = self.height*self.width*4
        self.state = initial_state
        self.step_count = 0

    def is_deadlock(self):
        boxes_position=self.state.get_boxes_position()

        for box_postion in boxes_position:
            if box_postion in self.dead_squares:
                return True
        
        return is_freeze_deadlock(self.state) or is_corner_deadlock(self.state)


    def reset(self):
        self.state = State(copy.deepcopy(self.map_matrix))
        self.step_count = 0
        self._boxes_on_goal = self._count_boxes_on_goal()
        return self.get_observation()

    def _count_boxes_on_goal(self) -> int:
        boxes = self.state.get_boxes_position()
        return sum(1 for b in boxes if tuple(b) in self.goals)

    def is_solved(self):
        return self.state.is_complete()

    
    def _inside(self, y, x):
        return 0<= y < self.height and 0<= x < self.width

    def _get_cell(self, y, x):
        if not self._inside(y, x):
            return '#'

        return self.state.get_matrix_element(y, x)

    def _is_walkable(self, y, x):

        if not self._inside(y, x):
            return False

        return self._get_cell(y, x) in [' ', '.', '@']

    def get_player_reachable_paths(self):
        start= self.state.get_worker_position()
        
        boxes={tuple(pos) for pos in self.state.get_boxes_position()}

        visited={start}
        queue=deque([start])

        paths={start:[]}

        while queue:
            cur_y, cur_x=queue.popleft()

            cur_path=paths[(cur_y,cur_x)]

            for action, (dy,dx) in enumerate(self.ACTIONS):
                next_y=cur_y+dy
                next_x=cur_x+dx

                next_pos=(next_y,next_x)

                if not self._is_walkable(next_y, next_x):
                    continue

                if next_pos in visited:
                    continue

                if next_pos in boxes:
                    continue

                visited.add(next_pos)
                paths[next_pos]=cur_path+[action]
                queue.append(next_pos)

        return paths

    def find_path_to_push_box(self, box_y, box_x, direction):
        dy, dx = self.ACTIONS[direction]

        player_target= (box_y-dy, box_x-dx)

        box_destination = (box_y+dy, box_x+dx)

        if not self._inside(*box_destination):
            return None

        boxes = {tuple(pos)for pos in self.state.get_boxes_position()}

        if box_destination in boxes:
            return None

        if not self._is_walkable(*player_target):
            return None

        if player_target in boxes:
            return None

        reachable = self.get_player_reachable_paths()

        return reachable.get(player_target)

    def valid_push(self, avoid_dead_squares= False):

        valid_actions= []

        boxes= {tuple(pos) for pos in self.state.get_boxes_position()}

        reachable= self.get_player_reachable_paths()

        for box_y, box_x in boxes:
            for direction, (dy, dx) in enumerate(self.ACTIONS):

                destination= (box_y + dy, box_x +dx)

                if not self._inside(*destination):
                    continue

                if self._get_cell(*destination) not in (' ', '.'):
                    continue

                if destination in boxes:
                    continue

                if (
                    avoid_dead_squares
                    and destination in self.dead_squares
                    and destination not in self.state.docks_position
                ):
                    continue

                player_target= (box_y-dy, box_x-dx)

                if player_target not in reachable:
                    continue

                valid_actions.append(
                    PushAction(
                        box_position=(box_y, box_x),
                        direction=direction,
                        destination=destination,
                        player_target=player_target,
                    )
                )
        return valid_actions

    def push_to_action_id(self, action: PushAction):
        y, x = action.box_position

        return (y* self.width + x) * 4 + action.direction

    def action_id_position_direction(self, action_id):

        if not 0 <= action_id < self.num_actions:
            return None

        cell_idx=action_id // 4
        direction_id=action_id % 4

        y= cell_idx // self.width
        x= cell_idx % self.width

        return y, x, direction_id

    def get_valid_action_mask(self):
        mask= np.zeros(self.num_actions, dtype=np.bool_)

        for action in self.valid_push():
            action_id = (self.push_to_action_id(action))

            mask[action_id]=True

        return mask
    
    def action_id_to_push(self, action_id):
        decoded = self.action_id_position_direction(action_id)

        if decoded is None:
            return None

        box_y, box_x, direction = decoded

        for action in self.valid_push():
            if (action.box_position == (box_y, box_x)) and action.direction == direction:
                return action
            
        return None

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            push_action= self.action_id_to_push(int(action))

        elif isinstance(action, PushAction):
            push_action= action

        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        if push_action is None:
            return (
                self.get_observation(),
                -5.0,
                False,
                {
                    "invalid_action": True,
                    "solved": False,
                    "deadlock": False
                }
            )

        box_y, box_x = push_action.box_position

        path= self.find_path_to_push_box(box_y, box_x, push_action.direction)

        if path is None:
            return (
                self.get_observation(),
                -5.0,
                False,
                {
                    "invalid_action": True,
                    "solved": False,
                    "deadlock": False
                }
            )

        for move_action in path:
            dy, dx = self.ACTIONS[move_action]
            self.state.move(dy, dx)

        dy, dx = self.ACTIONS[push_action.direction]
        self.state.move(dy, dx)

        self.step_count += 1

        solved = self.is_solved()

        deadlock = False

        if not solved:
            deadlock = self.is_deadlock()

        no_valid_push = False

        if not solved and not deadlock:

            no_valid_push = len(self.valid_push()) == 0

            if no_valid_push:
                deadlock = True

        timeout = self.step_count >= self.max_pushes

        reward = -1.0

        done = False

        if solved:
            reward += 100
            done = True

        elif deadlock:
            reward -= 100
            done = True

        elif timeout:
            reward -= 10
            done = True

        if not solved:
            new_boxes_on_goal = self._count_boxes_on_goal()
            delta_placed = new_boxes_on_goal - self._boxes_on_goal
            if delta_placed != 0:
                reward += delta_placed * 10.0
            self._boxes_on_goal = new_boxes_on_goal

        info = {
            "invalid_action": False,
            "solved": solved,
            "deadlock": deadlock,
            "timeout": timeout,
            "pushes": self.step_count,
            "player_moves": len(path),
        }

        return self.get_observation(), reward, done, info

    
    def get_observation(self):
        
        observation= np.zeros((4, self.height, self.width), dtype=np.float32)

        for y in range(self.height):
            for x in range(self.width):
                if (self.map_matrix[y][x]) == '#':
                    observation[0,y,x]= 1.0


        for y,x in self.goals:
            observation[1,y,x]= 1.0

        for y,x in (self.state.get_boxes_position()):
            observation[2,y,x]=1.0

        py,px= self.state.get_worker_position()

        observation[3,py,px]=1.0

        return observation

    def render(self):

        self.state.print_state()


        


        

        

        


                


                


        

