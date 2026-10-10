from state import State

def _is_box_frozen(state: State, y, x, visited: set) -> bool:

    if (y, x) in visited:
        return True

    visited.add((y, x))

    horizontal_blocked = _is_axis_block(state, y, x, "horizontal", visited)
    vertical_blocked   = _is_axis_block(state, y, x, "vertical",   visited)

    return horizontal_blocked and vertical_blocked


def _is_axis_block(state: State, y, x, axis: str, visited: set) -> bool:
    if axis == "horizontal":
        neighbors = [(y, x - 1), (y, x + 1)]
    else:
        neighbors = [(y - 1, x), (y + 1, x)]

    blocked_sides = 0

    for ny, nx in neighbors:
        cell = state.get_matrix_element(ny, nx)

        if cell == '#':
            blocked_sides += 1
        elif cell in ('$', '*'):
            if _is_box_frozen(state, ny, nx, visited):
                blocked_sides += 1

    return blocked_sides == 2


def is_freeze_deadlock(state: State) -> bool:
    for y, x in state.get_boxes_position():
        if state.get_matrix_element(y, x) == '*':
            continue

        if _is_box_frozen(state, y, x, visited=set()):
            return True

    return False
