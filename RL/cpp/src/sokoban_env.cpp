#include "sokoban_env.h"
#include "reward_shaping.h"
#include <algorithm>
#include <queue>
#include <set>

SokobanEnv::SokobanEnv(int max_pushes, bool avoid_dead_squares)
    : max_pushes(max_pushes), avoid_dead_squares(avoid_dead_squares), state() {}

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

bool SokobanEnv::is_dead_lock(bool avoid_dead_squares) const {
    return is_dead_lock(this->state, avoid_dead_squares);
}

bool SokobanEnv::is_dead_lock(const State& state, bool avoid_dead_squares) const {
    DeadlockDetector detector;
    std::vector<std::pair<int, int>> boxes_pos = state.get_boxes_position();

    if (avoid_dead_squares) {
        for (const auto& [y, x] : boxes_pos) {
            if (y >= 0 && y < static_cast<int>(is_dead_squares.size()) &&
                x >= 0 && x < static_cast<int>(is_dead_squares[0].size()) &&
                is_dead_squares[y][x]) {
                if (state.get_matrix_element(y, x) != '*') {
                    return true;
                }
            }
        }
    }

    for (const auto& pos : boxes_pos) {
        if (state.get_matrix_element(pos.first, pos.second) == '*') {
            continue;
        }

        if (detector.is_corner_deadlock(state, pos) ||
            detector.is_freeze_deadlock(state, pos)) {
            return true;
        }
    }
    return false;
}

void SokobanEnv::reset() {
    this->state = this->initial_state;
    this->step_count = 0;
    this->boxes_on_dock = count_boxes_on_goal();

    if (this->enable_reward_shaping) {
        this->last_potential = RewardShaping::compute_potential(
            *this, this->state, -200.0f, this->astar_max_nodes
        );
    } else {
        this->last_potential = 0.0f;
    }

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

bool SokobanEnv::is_walkable(int y, int x) const {
    return is_walkable(this->state, y, x);
}

bool SokobanEnv::is_walkable(const State& s, int y, int x) const {
    char cell = s.get_matrix_element(y, x);
    return (cell == ' ' || cell == '.' || cell == '@');
}

std::vector<bool> SokobanEnv::get_reachable_mask(const State& s) const {
    int H = s.height;
    int W = s.width;
    std::vector<bool> reachable(H * W, false);

    if (s.worker_y < 0 || s.worker_y >= H || s.worker_x < 0 || s.worker_x >= W) {
        return reachable;
    }

    std::queue<std::pair<int, int>> q;
    q.push({s.worker_y, s.worker_x});
    reachable[s.worker_y * W + s.worker_x] = true;

    while (!q.empty()) {
        auto [cy, cx] = q.front();
        q.pop();

        for (const auto& [dy, dx] : ACTIONS) {
            int ny = cy + dy;
            int nx = cx + dx;
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            int idx = ny * W + nx;
            if (reachable[idx]) continue;

            char cell = s.get_matrix_element(ny, nx);
            if (cell == ' ' || cell == '.' || cell == '@') {
                reachable[idx] = true;
                q.push({ny, nx});
            }
        }
    }
    return reachable;
}

std::pair<int, int> SokobanEnv::get_normalized_player_pos(const State& s) const {
    int H = s.height;
    int W = s.width;
    auto mask = get_reachable_mask(s);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (mask[y * W + x]) {
                return {y, x};
            }
        }
    }
    return {s.worker_y, s.worker_x};
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

bool SokobanEnv::is_in_list(std::pair<int, int> pos, const std::vector<std::pair<int, int>>& list_pos) const {
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

std::vector<int> SokobanEnv::valid_push(bool avoid_dead) const {
    return valid_push(this->state, avoid_dead);
}

std::vector<int> SokobanEnv::valid_push(const State& state, bool avoid_dead) const {
    std::vector<int> valid_pushs;
    int H = state.height;
    int W = state.width;
    auto reachable = get_reachable_mask(state);
    auto boxes_pos = state.get_boxes_position();

    for (const auto& [box_y, box_x] : boxes_pos) {
        for (size_t i = 0; i < ACTIONS.size(); ++i) {
            auto [dy, dx] = ACTIONS[i];
            int dest_y = box_y + dy;
            int dest_x = box_x + dx;

            if (dest_y < 0 || dest_y >= H || dest_x < 0 || dest_x >= W) {
                continue;
            }

            char cell = state.get_matrix_element(dest_y, dest_x);
            if (cell != '.' && cell != ' ') {
                continue;
            }

            if (is_in_list({dest_y, dest_x}, boxes_pos)) {
                continue;
            }

            if (avoid_dead && is_dead_squares[dest_y][dest_x] && !is_in_list({dest_y, dest_x}, docks_position)) {
                continue;
            }

            int player_target_y = box_y - dy;
            int player_target_x = box_x - dx;
            if (player_target_y < 0 || player_target_y >= H || player_target_x < 0 || player_target_x >= W) {
                continue;
            }

            if (!reachable[player_target_y * W + player_target_x]) {
                continue;
            }

            int action_id = (box_y * W + box_x) * 4 + static_cast<int>(i);
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

std::optional<std::tuple<int, int, int>> SokobanEnv::action_id_to_position_direction(int action_id) const {
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

    int player_moves = static_cast<int>(paths->size());
    auto [dy, dx] = ACTIONS[direction];
    this->state.apply_push_action(box_y, box_x, dy, dx);

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

    if (this->enable_reward_shaping) {
        float potential_before = this->last_potential;
        float potential_after;

        if (solved) {
            potential_after = 0.0f;
        } else if (done) {
            potential_after = -200.0f;
        } else {
            potential_after = RewardShaping::compute_potential(
                *this, this->state, -200.0f, this->astar_max_nodes
            );
        }

        float shaping = RewardShaping::compute_shaping_reward(
            potential_before, potential_after, 0.99f
        );
        reward += shaping;
        this->last_potential = potential_after;
    }

    StepInfo info;
    info.invalid_action = false;
    info.solved = solved;
    info.deadlock = deadlock;
    info.timeout = timeout;
    info.pushes = this->step_count;
    info.player_moves = player_moves;

    return StepResult{get_observation(), reward, done, info};
}
