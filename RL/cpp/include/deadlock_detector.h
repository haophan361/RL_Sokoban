#pragma once
#include "state.h"
#include <utility>
#include <set>

enum class Axis { Horizontal, Vertical };

class DeadlockDetector{
public:
    bool is_corner_deadlock(const State &state, std::pair<int,int> box_pos);
    bool is_freeze_deadlock(const State &state, std::pair<int,int> box_pos);

private:
    bool _is_axis_block(const State &state, int y, int x, Axis axis,
                        std::set<std::pair<int, int>>& path);
};
