#pragma once
#include <vector>
#include <array>
#include <string>
#include <utility>
#include <map>
#include <optional>
#include <tuple>
#include "state.h"
#include "deadlock_detector.h"
#include "step_info.h"

class SokobanEnv {
public:
    const std::array<std::pair<int, int>, 4> ACTIONS = {{
        {-1, 0},
        {1, 0},
        {0, -1},
        {0, 1}
    }};

    int max_pushes = 0;
    bool avoid_dead_squares = false;
    int boxes_on_dock = 0;

    std::vector<std::string> matrix;
    std::vector<std::vector<bool>> is_dead_squares;
    std::vector<std::pair<int, int>> docks_position;

    int num_actions = 0;
    int step_count = 0;

    State state;
    State initial_state;

    SokobanEnv(int max_pushes = 100, bool avoid_dead_squares = false);

    void set_level(const std::vector<std::vector<std::string>>& input_matrix,
                   const std::vector<std::pair<int, int>>& input_dead_squares);

    bool is_dead_lock(bool avoid_dead_squares = false);
    void reset();
    int count_boxes_on_goal();
    bool is_walkable(int y, int x);

    std::map<std::pair<int, int>, std::vector<int>> get_player_reachable_path();
    bool is_in_list(std::pair<int, int> pos, const std::vector<std::pair<int, int>>& list_pos);
    std::optional<std::vector<int>> find_path_to_push(int box_y, int box_x, int direction);
    std::vector<int> valid_push(bool avoid_dead_squares = false);

    std::vector<bool> get_valid_action_mask();
    std::vector<float> get_observation();

    std::optional<std::tuple<int, int, int>> action_id_to_position_direction(int action_id);
    StepResult return_by_step(float reward, bool done, bool invalid_action, bool solved, bool deadlock);
    StepResult step(int action_id);
};

