#include "deadlock_detector.h"

bool DeadlockDetector::_is_axis_block(const State &state, int y, int x, Axis axis,
                                     std::set<std::pair<int, int>>& path) {

    const int dy = axis == Axis::Vertical ? 1 : 0;
    const int dx = axis == Axis::Horizontal ? 1 : 0;
    const char before = state.get_matrix_element(y - dy, x - dx);
    const char after = state.get_matrix_element(y + dy, x + dx);

    if (before == '#' || after == '#') {
        return true;
    }

    const bool box_before = before == '$' || before == '*';
    const bool box_after = after == '$' || after == '*';
    if (!box_before && !box_after) {
        return false;
    }

    const Axis perpendicular = axis == Axis::Horizontal ? Axis::Vertical : Axis::Horizontal;
    const auto neighbor_blocks = [&](int neighbor_y, int neighbor_x) {

        return path.count({neighbor_y, neighbor_x}) ||
               _is_axis_block(state, neighbor_y, neighbor_x, perpendicular, path);
    };

    path.insert({y, x});
    const bool blocked = (box_before && neighbor_blocks(y - dy, x - dx)) ||
                         (box_after && neighbor_blocks(y + dy, x + dx));
    path.erase({y, x});
    return blocked;
}

bool DeadlockDetector::is_freeze_deadlock(const State &state, std::pair<int,int> box_pos) {
    const auto [y, x] = box_pos;
    if (state.get_matrix_element(y, x) != '$') {
        return false;
    }

    std::set<std::pair<int, int>> path;
    return _is_axis_block(state, y, x, Axis::Horizontal, path) &&
           _is_axis_block(state, y, x, Axis::Vertical, path);
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
