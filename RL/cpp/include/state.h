#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <utility>

class State {
public:
    int height = 0;
    int width = 0;
    int worker_y = -1;
    int worker_x = -1;

    std::vector<std::string> matrix;
    std::vector<std::pair<int, int>> docks_position;
    std::string pathSolution = "";

    State() = default;
    State(const std::vector<std::vector<std::string>>& input_matrix);

    std::vector<std::vector<std::string>> get_matrix() const;
    char get_matrix_element(int y, int x) const;
    char get_element_next_step(int dy, int dx) const;
    std::pair<int, int> get_worker_position() const;
    std::vector<std::pair<int, int>> get_boxes_position() const;
    std::vector<std::pair<int, int>> get_docks_position() const;
    bool worker_can_move(int dy, int dx) const;
    bool worker_can_push_box(int dy, int dx) const;
    bool is_complete() const;
    void move_box(int y_box, int x_box, int move_y, int move_x);
    void move(int dy, int dx);
    void print_state() const;
    State clone() const;

private:
    void init_map(const std::vector<std::vector<std::string>>& input_matrix);
    void find_worker_and_docks();
};
