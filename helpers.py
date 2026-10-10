from state import State
from deadlock_detector.corner_deadlock import is_corner_deadlock
from deadlock_detector.freeze_deadlock import is_freeze_deadlock
def valid_move(state:State):
    moves = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }

    valid_moves = []

    for step, (y, x) in moves.items():
        if state.worker_can_move(y, x) or state.worker_can_push_box(y, x):
            valid_moves.append(step)

    return valid_moves

def is_deadlock(state:State):
    return is_corner_deadlock(state) or is_freeze_deadlock(state)

def box_to_dock(state:State):
    sum = 0
    box_list = state.get_boxes_position()
    dock_list = state.get_docks_position()
    for box in box_list:
        min_distance = float('inf')
        for dock in dock_list:
            distance = (abs(dock[0] - box[0]) + abs(dock[1] - box[1]))
            if distance < min_distance:
                min_distance = distance
        sum += min_distance
    return sum

def worker_to_box(state:State):
    sum = 0
    box_list = state.get_boxes_position()
    worker_pos = state.get_worker_position()
    for box in box_list:
        sum += abs(box[0] - worker_pos[0]) + abs(box[1] - worker_pos[1])
    return sum