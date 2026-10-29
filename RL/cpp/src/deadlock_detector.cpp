#include "deadlock_detector.h"
#include <vector>
#include <utility>
#include <set>

bool DeadlockDetector::_is_axis_block(const State &state, int y, int x, Axis axis, const std::set<std::pair<int,int>> &visited) {
    std::vector<std::pair<int,int>> neighbors;

    if (axis == Axis::Horizontal) {
        neighbors = {{y, x - 1}, {y, x + 1}};
    } else {
        neighbors = {{y - 1, x}, {y + 1, x}};
    }

    for (const auto& neigh : neighbors) {
        char cell = state.get_matrix_element(neigh.first, neigh.second);

        if (cell == '#') {
            return true;
        } else if (cell == '$' || cell == '*') {
            if (visited.count({y, x}) || _is_box_frozen(state, neigh.first, neigh.second, visited)) {
                return true;
            }
        }
    }
    return false;
}

bool DeadlockDetector::_is_box_frozen(const State &state, int y, int x, const std::set<std::pair<int,int>> &visited) {
    std::set<std::pair<int,int>> cur_visited = visited;
    cur_visited.insert({y, x});

    return _is_axis_block(state, y, x, Axis::Horizontal, cur_visited) && 
           _is_axis_block(state, y, x, Axis::Vertical, cur_visited);
}

bool DeadlockDetector::is_corner_deadlock(const State &state, std::pair<int,int> box_pos) {
    bool left= state.get_matrix_element(box_pos.first, box_pos.second - 1) == '#';
    bool right= state.get_matrix_element(box_pos.first, box_pos.second + 1) == '#';
    bool up= state.get_matrix_element(box_pos.first - 1, box_pos.second) == '#';
    bool down= state.get_matrix_element(box_pos.first + 1, box_pos.second) == '#';

    if ((left || right) && (up || down)) {
        return true;
    }
    return false;
}

bool DeadlockDetector::is_freeze_deadlock(const State &state, std::pair<int,int> box_pos) {
    if (_is_box_frozen(state, box_pos.first, box_pos.second, std::set<std::pair<int,int>>{})) {
        return true;
    }
    return false;
}
