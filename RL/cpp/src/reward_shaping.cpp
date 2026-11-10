#include "reward_shaping.h"
#include "sokoban_env.h"
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <sstream>

std::string RewardShaping::make_state_key(const SokobanEnv& env, const State& state) {
    auto norm_player = env.get_normalized_player_pos(state);
    auto boxes = state.get_boxes_position();
    std::sort(boxes.begin(), boxes.end());

    std::ostringstream oss;
    oss << norm_player.first << "," << norm_player.second << "|";
    for (const auto& [y, x] : boxes) {
        oss << y << "," << x << ";";
    }
    return oss.str();
}

int RewardShaping::heuristic_min_matching(
    const std::vector<std::pair<int, int>>& boxes,
    const std::vector<std::pair<int, int>>& goals
) {
    int n = static_cast<int>(boxes.size());
    int m = static_cast<int>(goals.size());
    if (n == 0 || m == 0) return 0;

    std::vector<std::pair<int, std::pair<int, int>>> all_pairs;
    all_pairs.reserve(n * m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int dist = std::abs(boxes[i].first - goals[j].first)
                     + std::abs(boxes[i].second - goals[j].second);
            all_pairs.push_back({dist, {i, j}});
        }
    }
    std::sort(all_pairs.begin(), all_pairs.end());

    std::vector<bool> used_box(n, false), used_goal(m, false);
    int total = 0, matched = 0;
    for (const auto& [c, p] : all_pairs) {
        if (matched >= n) break;
        int bi = p.first, gi = p.second;
        if (used_box[bi] || used_goal[gi]) continue;
        used_box[bi] = true;
        used_goal[gi] = true;
        total += c;
        matched++;
    }
    return total;
}

struct AStarNode {
    State state;
    int g = 0;
    int f = 0;
    std::string key;

    bool operator>(const AStarNode& other) const {
        return f > other.f;
    }
};

AStarResult RewardShaping::astar_search(
    const SokobanEnv& env,
    const State& state,
    int max_nodes
) {
    AStarResult result{false, 0};

    if (state.is_complete()) {
        result.solved = true;
        result.pushes = 0;
        return result;
    }

    if (env.is_dead_lock(state, env.avoid_dead_squares)) {
        return result;
    }

    auto boxes_init = state.get_boxes_position();
    int h0 = heuristic_min_matching(boxes_init, env.docks_position);

    AStarNode start_node;
    start_node.state = state.clone();
    start_node.g = 0;
    start_node.f = h0;
    start_node.key = make_state_key(env, state);

    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open;
    std::unordered_set<std::string> closed;

    open.push(std::move(start_node));
    int nodes_expanded = 0;

    while (!open.empty() && nodes_expanded < max_nodes) {
        AStarNode current = std::move(const_cast<AStarNode&>(open.top()));
        open.pop();

        if (closed.count(current.key)) continue;
        closed.insert(current.key);
        nodes_expanded++;

        if (current.state.is_complete()) {
            result.solved = true;
            result.pushes = current.g;
            return result;
        }

        std::vector<int> valid_actions = env.valid_push(current.state, env.avoid_dead_squares);

        for (int action_id : valid_actions) {
            auto opt = env.action_id_to_position_direction(action_id);
            if (!opt.has_value()) continue;
            auto [box_y, box_x, direction] = *opt;
            auto [dy, dx] = env.ACTIONS[direction];

            State next_state = current.state.clone();
            next_state.apply_push_action(box_y, box_x, dy, dx);

            std::string next_key = make_state_key(env, next_state);
            if (closed.count(next_key)) continue;

            if (env.is_dead_lock(next_state, env.avoid_dead_squares)) continue;

            auto next_boxes = next_state.get_boxes_position();
            int h = heuristic_min_matching(next_boxes, env.docks_position);
            int g = current.g + 1;

            AStarNode next_node;
            next_node.state = std::move(next_state);
            next_node.g = g;
            next_node.f = g + h;
            next_node.key = std::move(next_key);

            open.push(std::move(next_node));
        }
    }

    return result;
}

float RewardShaping::compute_potential(
    const SokobanEnv& env,
    const State& state,
    float fallback_potential,
    int max_nodes
) {
    AStarResult res = astar_search(env, state, max_nodes);
    if (res.solved) {
        return static_cast<float>(-res.pushes);
    }
    return fallback_potential;
}

float RewardShaping::compute_shaping_reward(
    float potential_before,
    float potential_after,
    float gamma
) {
    return gamma * potential_after - potential_before;
}
