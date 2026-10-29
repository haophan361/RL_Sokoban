#pragma once
#include "state.h"
#include <vector>
#include <utility>
#include <set>

enum class Axis { Horizontal, Vertical };

class DeadlockDetector{
public:
    bool _is_box_frozen(const State &state, int y, int x, const std::set<std::pair<int,int>> &visited);
    bool _is_axis_block(const State &state, int y, int x, Axis axis, const std::set<std::pair<int,int>> &visited);
    bool is_corner_deadlock(const State &state, std::pair<int,int> box_pos);
    bool is_freeze_deadlock(const State &state, std::pair<int,int> box_pos);
};