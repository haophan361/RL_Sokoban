#include "state.h"
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

State::State(const std::vector<std::vector<std::string>>& input_matrix) {
    init_map(input_matrix);
    find_worker_and_docks();
}

void State::init_map(const std::vector<std::vector<std::string>>& input_matrix) {
    height = static_cast<int>(input_matrix.size());
    width = height > 0 ? static_cast<int>(input_matrix[0].size()) : 0;

    matrix.resize(height);
    for (int y = 0; y < height; ++y) {
        matrix[y].resize(width, ' ');
        for (int x = 0; x < width; ++x) {
            if (!input_matrix[y][x].empty()) {
                matrix[y][x] = input_matrix[y][x][0];
            }
        }
    }
}

std::vector<std::vector<std::string>> State::get_matrix() const {
    std::vector<std::vector<std::string>> res(height, std::vector<std::string>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            res[y][x] = std::string(1, matrix[y][x]);
        }
    }
    return res;
}

char State::get_matrix_element(int y, int x) const {
    if (y < 0 || y >= height || x < 0 || x >= width) {
        return '#';
    }
    return matrix[y][x];
}

char State::get_element_next_step(int dy, int dx) const {
    int new_y = worker_y + dy;
    int new_x = worker_x + dx;
    return get_matrix_element(new_y, new_x);
}

std::pair<int, int> State::get_worker_position() const {
    return {worker_y, worker_x};
}

std::vector<std::pair<int, int>> State::get_boxes_position() const {
    std::vector<std::pair<int, int>> boxes;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            char c = matrix[y][x];
            if (c == '$' || c == '*') {
                boxes.push_back({y, x});
            }
        }
    }
    return boxes;
}

std::vector<std::pair<int, int>> State::get_docks_position() const {
    return docks_position;
}

bool State::worker_can_move(int dy, int dx) const {
    char cell = get_element_next_step(dy, dx);
    return cell == ' ' || cell == '.';
}

bool State::worker_can_push_box(int dy, int dx) const {
    char next_cell = get_element_next_step(dy, dx);
    if (next_cell != '*' && next_cell != '$') {
        return false;
    }

    char behind_cell = get_element_next_step(dy + dy, dx + dx);
    return behind_cell == '.' || behind_cell == ' ';
}

bool State::is_complete() const {
    if (docks_position.empty()) {
        return false;
    }

    for (const auto& [dy, dx] : docks_position) {
        if (matrix[dy][dx] != '*') {
            return false;
        }
    }
    return true;
}

void State::move_box(int y_box, int x_box, int move_y, int move_x) {
    char box_element = matrix[y_box][x_box];
    int next_y = y_box + move_y;
    int next_x = x_box + move_x;
    char next_box_element = matrix[next_y][next_x];

    if (box_element == '$') {
        matrix[y_box][x_box] = ' ';
        if (next_box_element == ' ') {
            matrix[next_y][next_x] = '$';
        } else if (next_box_element == '.') {
            matrix[next_y][next_x] = '*';
        }
    } else if (box_element == '*') {
        matrix[y_box][x_box] = '.';
        if (next_box_element == ' ') {
            matrix[next_y][next_x] = '$';
        } else if (next_box_element == '.') {
            matrix[next_y][next_x] = '*';
        }
    }
}

void State::move(int dy, int dx) {
    if (worker_can_move(dy, dx)) {
        matrix[worker_y + dy][worker_x + dx] = '@';
        matrix[worker_y][worker_x] = ' ';
        worker_y += dy;
        worker_x += dx;
    } else if (worker_can_push_box(dy, dx)) {
        move_box(worker_y + dy, worker_x + dx, dy, dx);
        matrix[worker_y + dy][worker_x + dx] = '@';
        matrix[worker_y][worker_x] = ' ';
        worker_y += dy;
        worker_x += dx;
    }

    for (const auto& [i, j] : docks_position) {
        char c = matrix[i][j];
        if (c != '*' && c != '@') {
            matrix[i][j] = '.';
        }
    }
}

void State::print_state() const {
    for (const auto& row : matrix) {
        for (char c : row) {
            std::cout << c << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}

State State::clone() const {
    return State(*this);
}

void State::find_worker_and_docks() {
    docks_position.clear();
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            char c = matrix[y][x];
            if (c == '@') {
                worker_y = y;
                worker_x = x;
            }
            if (c == '.' || c == '*') {
                docks_position.push_back({y, x});
            }
        }
    }
}
