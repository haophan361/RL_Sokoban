import time
import copy
import random
import queue
from collections import deque
import math
from helpers import valid_move, is_deadlock, box_to_dock, worker_to_box
from state import State

def bfs(state:State):
    start = time.time()
    node_generated = 0
    start_state = copy.deepcopy(state)
    node_generated += 1

    if is_deadlock(start_state):
        print("No Solution!")
        return None

    queue = deque([start_state])
    visited = set()
    visited.add(tuple(map(tuple, start_state.get_matrix())))

    print("Processing BFS......")

    while queue:
        currState = queue.popleft()
        move = valid_move(currState)
        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1

            if step == 'U':
                newState.move(-1, 0)
            elif step == 'D':
                newState.move(1, 0)
            elif step == 'L':
                newState.move(0, -1)
            elif step == 'R':
                newState.move(0, 1)

            newState.pathSolution += step

            if newState.is_complete():
                end = time.time()
                print("Time to find solution:", round(end - start, 2), "seconds")
                print("Number of visited nodes:", node_generated)
                print("Solution:", newState.pathSolution, "Number steps:", len(newState.pathSolution))
                return newState.pathSolution

            if (tuple(map(tuple, newState.get_matrix())) not in visited) and (not is_deadlock(newState)):
                queue.append(newState)
                visited.add(tuple(map(tuple, newState.get_matrix())))

    print(node_generated)
    print("No Solution!")
    return None

def dfs(state:State):
    start = time.time()
    node_generated = 0
    start_state = copy.deepcopy(state)
    node_generated += 1

    if is_deadlock(start_state):
        print("No Solution!")
        return None

    stack = [start_state]
    visited = set()
    visited.add(tuple(map(tuple, start_state.get_matrix())))

    print("Processing DFS......")

    while stack:
        currState = stack.pop()
        move = valid_move(currState)
        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1

            if step == 'U':
                newState.move(-1, 0)
            elif step == 'D':
                newState.move(1, 0)
            elif step == 'L':
                newState.move(0, -1)
            elif step == 'R':
                newState.move(0, 1)

            newState.pathSolution += step

            if newState.is_complete():
                end = time.time()
                print("Time to find solution:", round(end - start, 2), "seconds")
                print("Number of visited nodes:", node_generated)
                print("Solution:", newState.pathSolution, "Number steps:", len(newState.pathSolution))
                return newState.pathSolution

            if (tuple(map(tuple, newState.get_matrix())) not in visited) and (not is_deadlock(newState)):
                stack.append(newState)
                visited.add(tuple(map(tuple, newState.get_matrix())))

    print(node_generated)
    print("No Solution!")
    return None


def astar(state:State):
    start = time.time()
    node_generated = 0
    start_state = copy.deepcopy(state)
    node_generated += 1
    start_state.heuristic = worker_to_box(start_state) + box_to_dock(start_state)

    if is_deadlock(start_state):
        print("No Solution!")
        return None

    open_list = queue.PriorityQueue()
    open_list.put(start_state)
    close_list = set()

    print("Processing A*......")

    while not open_list.empty():
        cur_state = open_list.get()
        move = valid_move(cur_state)
        close_list.add(tuple(map(tuple, cur_state.get_matrix())))

        for step in move:
            new_state = copy.deepcopy(cur_state)
            node_generated += 1

            if step == 'U':
                new_state.move(-1, 0)
            elif step == 'D':
                new_state.move(1, 0)
            elif step == 'L':
                new_state.move(0, -1)
            elif step == 'R':
                new_state.move(0, 1)

            new_state.pathSolution += step
            new_state.heuristic = worker_to_box(new_state) + box_to_dock(new_state)

            if new_state.is_complete():
                end = time.time()
                print("Time to find solution:", round(end - start, 2), "seconds")
                print("Number of visited nodes:", node_generated)
                print("Solution:", new_state.pathSolution, "Number steps:", len(new_state.pathSolution))
                return new_state.pathSolution

            if (tuple(map(tuple, new_state.get_matrix())) not in close_list) and not is_deadlock(new_state):
                open_list.put(new_state)

    print(node_generated)
    print("No Solution!")
    return None


def backtracking(state:State):
    start = time.time()
    node_generated = 0
    start_state = copy.deepcopy(state)
    visited = set()

    if is_deadlock(start_state):
        print("No Solution!")
        return None

    print("Processing BACKTRACKING......")

    def solve(state):
        nonlocal node_generated

        # Nếu hoàn thành, trả về đường đi
        if state.is_complete():
            return state.pathSolution

        # Lưu trạng thái vào visited
        visited.add(tuple(map(tuple, state.get_matrix())))

        # Thử từng bước đi khả dụng
        for step in valid_move(state):
            newState = copy.deepcopy(state)
            node_generated += 1

            # Thực hiện bước đi
            if step == 'U':
                newState.move(-1, 0)
            elif step == 'D':
                newState.move(1, 0)
            elif step == 'L':
                newState.move(0, -1)
            elif step == 'R':
                newState.move(0, 1)

            newState.pathSolution += step

            # Kiểm tra deadlock hoặc trạng thái đã ghé qua
            if tuple(map(tuple, newState.get_matrix())) in visited or is_deadlock(newState):
                continue

            # Đệ quy tìm giải pháp từ trạng thái mới
            solutionPath = solve(newState)
            if solutionPath:  # Nếu tìm được giải pháp, trả về
                return solutionPath

        # Không tìm thấy đường đi, quay lui
        return None

    # Bắt đầu giải bài toán
    solution = solve(start_state)
    end = time.time()

    if solution:
        print("Time to find solution:", round(end - start, 2), "seconds")
        print("Number of visited nodes:", node_generated)
        print("Solution:", solution, "Number steps:", len(solution))
        return solution
    else:
        print("No Solution!")
        return None


def simulated_annealing(state:State, initial_temperature=1000, cooling_rate=0.95, max_iterations=1000):
    start = time.time()
    node_generated = 0

    # Sao chép trạng thái ban đầu
    current_state = copy.deepcopy(state)
    node_generated += 1

    # Đặt nhiệt độ ban đầu
    temperature = initial_temperature

    # Đường đi tốt nhất và heuristic ban đầu
    best_path = ""
    best_heuristic = worker_to_box(current_state) + box_to_dock(current_state)

    print("Processing SIMULATED ANNEALING......")

    for _ in range(max_iterations):
        # Kiểm tra nếu đã hoàn thành
        if current_state.is_complete():
            end = time.time()
            print("Time to find solution:", round(end - start, 2), "seconds")
            print("Number of visited nodes:", node_generated)
            print("Solution:", best_path, "Number steps:", len(best_path))
            return best_path

        # Lấy tất cả các nước đi hợp lệ
        moves = valid_move(current_state)
        if not moves:
            break

        # Chọn ngẫu nhiên một bước đi
        step = random.choice(moves)
        next_state = copy.deepcopy(current_state)
        node_generated += 1

        # Thực hiện bước đi
        if step == 'U':
            next_state.move(-1, 0)
        elif step == 'D':
            next_state.move(1, 0)
        elif step == 'L':
            next_state.move(0, -1)
        elif step == 'R':
            next_state.move(0, 1)

        # Tính heuristic của trạng thái tiếp theo
        next_heuristic = worker_to_box(next_state) + box_to_dock(next_state)

        # Tính toán xác suất chọn trạng thái kém hơn
        delta = next_heuristic - best_heuristic
        if delta < 0 or random.uniform(0, 1) < math.exp(-delta / temperature):
            current_state = next_state
            best_path += step
            best_heuristic = next_heuristic

        # Giảm nhiệt độ
        temperature *= cooling_rate

    print("No Solution Found!")
    return None