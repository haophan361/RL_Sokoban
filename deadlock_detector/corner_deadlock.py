from state import State

def is_corner_deadlock(state: State):
    boxes_position= state.get_boxes_position()

    for y,x in boxes_position:
        if state.get_matrix_element(y,x) == "*":
            continue

        left= state.get_matrix_element(y,x-1)
        right= state.get_matrix_element(y,x+1)
        up= state.get_matrix_element(y-1,x)
        down= state.get_matrix_element(y+1,x)

        if (left == "#" and up == "#") or \
           (left == "#" and down == "#") or \
           (right == "#" and up == "#") or \
           (right == "#" and down == "#"):
            return True
    
    return False
