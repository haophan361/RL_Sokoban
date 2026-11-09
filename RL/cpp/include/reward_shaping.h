#pragma once
#include "state.h"
#include <vector>
#include <utility>
#include <string>

class SokobanEnv;

struct AStarResult {
    bool solved = false;
    int  pushes = 0;
};

class RewardShaping {
public:
    static AStarResult astar_search(
        const SokobanEnv& env,
        const State& state,
        int max_nodes = 50000
    );

    static float compute_potential(
        const SokobanEnv& env,
        const State& state,
        float fallback_potential = -200.0f,
        int max_nodes = 50000
    );

    static float compute_shaping_reward(
        float potential_before,
        float potential_after,
        float gamma = 0.99f
    );

private:
    static int heuristic_min_matching(
        const std::vector<std::pair<int, int>>& boxes,
        const std::vector<std::pair<int, int>>& goals
    );

    static std::string make_state_key(
        const SokobanEnv& env,
        const State& state
    );
};
