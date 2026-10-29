#include "sokoban_env.h"
#include <algorithm>
#include <queue>
#include <set>

SokobanEnv::SokobanEnv(int max_pushes, bool avoid_dead_squares)
    : max_pushes(max_pushes), avoid_dead_squares(avoid_dead_squares), state({}) {}

void SokobanEnv::set_level(const std::vector<std::vector<std::string>>& input_matrix,
                           const std::vector<std::pair<int, int>>& input_dead_squares) {
    this->initial_state = State(input_matrix);
    this->state = this->initial_state;

    this->matrix = this->initial_state.matrix;
    this->docks_position = this->initial_state.docks_position;

    this->is_dead_squares.assign(this->state.height, std::vector<bool>(this->state.width, false));
    for (const auto& [y, x] : input_dead_squares) {
        if (y >= 0 && y < this->state.height && x >= 0 && x < this->state.width) {
            is_dead_squares[y][x] = true;
        }
    }

    this->num_actions = this->state.height * this->state.width * 4;
    this->step_count = 0;
    this->boxes_on_dock = count_boxes_on_goal();
}

bool SokobanEnv::is_dead_lock(bool avoid_dead_squares) {
    DeadlockDetector detector;
    std::vector<std::pair<int, int>> boxes_pos = this->state.get_boxes_position();

    if (avoid_dead_squares) {
        for (const auto& [y, x] : boxes_pos) {
            if (is_dead_squares[y][x]) {
                return true;
            }
        }
    }

    for (const auto& pos : boxes_pos) {
        if (state.get_matrix_element(pos.first, pos.second) == '*') {
            continue;
        }
        if (detector.is_corner_deadlock(this->state, pos) || detector.is_freeze_deadlock(this->state, pos)) {
            return true;
        }
    }
    return false;
}

void SokobanEnv::reset() {
    this->state = this->initial_state;
    this->step_count = 0;
    this->boxes_on_dock = count_boxes_on_goal();
    get_observation();
}

int SokobanEnv::count_boxes_on_goal() {
    std::vector<std::pair<int, int>> boxes_pos = this->state.get_boxes_position();
    int sum = 0;
    for (const auto& pos : boxes_pos) {
        if (is_in_list(pos, docks_position)) {
            sum += 1;
        }
    }
    return sum;
}

bool SokobanEnv::is_walkable(int y, int x) {
    char cell = this->state.get_matrix_element(y, x);
    return (cell == ' ' || cell == '.' || cell == '@');
}

std::map<std::pair<int, int>, std::vector<int>> SokobanEnv::get_player_reachable_path() {
    std::pair<int, int> start = {this->state.worker_y, this->state.worker_x};
    std::set<std::pair<int, int>> visited = {start};

    std::queue<std::pair<int, int>> queue;
    queue.push(start);

    std::map<std::pair<int, int>, std::vector<int>> paths = {{start, {}}};

    while (!queue.empty()) {
        auto [cur_y, cur_x] = queue.front();
        queue.pop();

        std::vector<int> cur_path = paths[{cur_y, cur_x}];

        for (size_t i = 0; i < ACTIONS.size(); ++i) {
            int dy = ACTIONS[i].first;
            int dx = ACTIONS[i].second;

            int next_y = cur_y + dy;
            int next_x = cur_x + dx;
            std::pair<int, int> next_pos = {next_y, next_x};

            if (!is_walkable(next_y, next_x)) {
                continue;
            }
            if (visited.count(next_pos)) {
                continue;
            }

            visited.insert(next_pos);
            std::vector<int> next_path = cur_path;
            next_path.push_back(static_cast<int>(i));
            paths[next_pos] = next_path;
            queue.push(next_pos);
        }
    }
    return paths;
}

bool SokobanEnv::is_in_list(std::pair<int, int> pos, const std::vector<std::pair<int, int>>& list_pos) {
    auto it = std::find(list_pos.begin(), list_pos.end(), pos);
    return it != list_pos.end();
}

std::optional<std::vector<int>> SokobanEnv::find_path_to_push(int box_y, int box_x, int direction) {
    auto [dy, dx] = ACTIONS[direction];
    std::pair<int, int> player_target = {box_y - dy, box_x - dx};
    std::pair<int, int> box_destination = {box_y + dy, box_x + dx};

    if (this->state.get_matrix_element(box_destination.first, box_destination.second) == '#') {
        return std::nullopt;
    }

    std::vector<std::pair<int, int>> boxes_pos = this->state.get_boxes_position();
    if (is_in_list(box_destination, boxes_pos) || is_in_list(player_target, boxes_pos)) {
        return std::nullopt;
    }

    if (!is_walkable(player_target.first, player_target.second)) {
        return std::nullopt;
    }

    auto reachable = get_player_reachable_path();
    auto it = reachable.find(player_target);
    if (it == reachable.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<int> SokobanEnv::valid_push(bool avoid_dead) {
    std::vector<int> valid_pushs;
    std::vector<std::pair<int, int>> boxes_pos = this->state.get_boxes_position();
    auto reachable = get_player_reachable_path();

    for (const auto& [box_y, box_x] : boxes_pos) {
        for (size_t i = 0; i < ACTIONS.size(); ++i) {
            auto [dy, dx] = ACTIONS[i];
            std::pair<int, int> destination = {box_y + dy, box_x + dx};

            char cell = state.get_matrix_element(destination.first, destination.second);
            if (cell != '.' && cell != ' ') {
                continue;
            }

            if (is_in_list(destination, boxes_pos)) {
                continue;
            }

            if (avoid_dead && is_dead_squares[destination.first][destination.second] && !is_in_list(destination, docks_position)) {
                continue;
            }

            std::pair<int, int> player_target = {box_y - dy, box_x - dx};
            if (reachable.count(player_target) == 0) {
                continue;
            }

            int action_id = (box_y * this->state.width + box_x) * 4 + static_cast<int>(i);
            valid_pushs.push_back(action_id);
        }
    }
    return valid_pushs;
}

std::vector<bool> SokobanEnv::get_valid_action_mask() {
    std::vector<bool> mask(num_actions, false);
    std::vector<int> actions = valid_push(this->avoid_dead_squares);
    for (int action_id : actions) {
        if (action_id >= 0 && action_id < num_actions) {
            mask[action_id] = true;
        }
    }
    return mask;
}

std::vector<float> SokobanEnv::get_observation() {
    std::vector<float> obs(4 * this->state.height * this->state.width, 0.0f);

    auto idx = [this](int c, int y, int x) {
        return (c * this->state.height + y) * this->state.width + x;
    };

    for (int y = 0; y < this->state.height; ++y) {
        for (int x = 0; x < this->state.width; ++x) {
            if (this->matrix[y][x] == '#') {
                obs[idx(0, y, x)] = 1.0f;
            }
        }
    }

    for (const auto& [y, x] : this->docks_position) {
        obs[idx(1, y, x)] = 1.0f;
    }

    for (const auto& [y, x] : this->state.get_boxes_position()) {
        obs[idx(2, y, x)] = 1.0f;
    }

    std::pair<int, int> player_pos = this->state.get_worker_position();
    if (player_pos.first >= 0 && player_pos.second >= 0) {
        obs[idx(3, player_pos.first, player_pos.second)] = 1.0f;
    }
    return obs;
}

std::optional<std::tuple<int, int, int>> SokobanEnv::action_id_to_position_direction(int action_id) {
    if (action_id < 0 || action_id >= num_actions) {
        return std::nullopt;
    }

    int cell_idx = action_id / 4;
    int direction = action_id % 4;

    int box_y = cell_idx / this->state.width;
    int box_x = cell_idx % this->state.width;

    return std::make_tuple(box_y, box_x, direction);
}

StepResult SokobanEnv::return_by_step(float reward, bool done, bool invalid_action, bool solved, bool deadlock) {
    StepInfo info;
    info.invalid_action = invalid_action;
    info.solved = solved;
    info.deadlock = deadlock;
    info.timeout = false;
    info.pushes = this->step_count;
    info.player_moves = 0;
    return StepResult{get_observation(), reward, done, info};
}

StepResult SokobanEnv::step(int action_id) {
    auto res = action_id_to_position_direction(action_id);
    if (!res.has_value()) {
        return return_by_step(-5.0f, false, true, false, false);
    }
    auto [box_y, box_x, direction] = *res;

    auto paths = find_path_to_push(box_y, box_x, direction);
    if (!paths.has_value()) {
        return return_by_step(-5.0f, false, true, false, false);
    }

    for (const auto& move_action : *paths) {
        auto [dy, dx] = ACTIONS[move_action];
        this->state.move(dy, dx);
    }

    auto [dy, dx] = ACTIONS[direction];
    this->state.move(dy, dx);

    step_count += 1;

    bool solved = this->state.is_complete();
    bool deadlock = false;

    if (!solved) {
        deadlock = is_dead_lock(this->avoid_dead_squares);
    }

    if (!solved && !deadlock) {
        bool no_valid_push = valid_push(this->avoid_dead_squares).empty();
        if (no_valid_push) {
            deadlock = true;
        }
    }

    bool timeout = step_count >= max_pushes;
    float reward = -1.0f;
    bool done = false;

    if (solved) {
        reward += 100.0f;
        done = true;
    } else if (deadlock) {
        reward -= 100.0f;
        done = true;
    } else if (timeout) {
        reward -= 10.0f;
        done = true;
    }

    if (!solved) {
        int new_boxes_on_dock = count_boxes_on_goal();
        int delta = new_boxes_on_dock - this->boxes_on_dock;
        if (delta != 0) {
            reward += delta * 10.0f;
        }
        this->boxes_on_dock = new_boxes_on_dock;
    }

    StepInfo info;
    info.invalid_action = false;
    info.solved = solved;
    info.deadlock = deadlock;
    info.timeout = timeout;
    info.pushes = this->step_count;
    info.player_moves = static_cast<int>(paths->size());

    return StepResult{get_observation(), reward, done, info};
}
