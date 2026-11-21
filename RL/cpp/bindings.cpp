#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include "sokoban_env.h"
#include "state.h"
#include "deadlock_detector.h"
#include "step_info.h"

namespace py = pybind11;

class PySokobanEnv {
    
public:
    SokobanEnv env;

    PySokobanEnv(int max_pushes = 100, bool avoid_dead_squares = false)
        : env(max_pushes, avoid_dead_squares) {}

    void set_level(const std::vector<std::vector<std::string>>& map,
                   const std::vector<std::pair<int,int>>& dead_squares) {
        env.set_level(map, dead_squares);
    }

    py::array_t<float> reset() {
        env.reset();
        return obs_to_numpy();
    }

    py::tuple step(int action_id) {
        StepResult r = env.step(action_id);
        py::dict info;
        info["invalid_action"] = r.info.invalid_action;
        info["solved"] = r.info.solved;
        info["deadlock"] = r.info.deadlock;
        info["timeout"] = r.info.timeout;
        info["pushes"] = r.info.pushes;
        info["player_moves"] = r.info.player_moves;
        return py::make_tuple(obs_to_numpy(), r.reward, r.done, info);
    }

    py::array_t<bool> get_valid_action_mask() {
        std::vector<bool> mask = env.get_valid_action_mask();
        int N = env.num_actions;
        py::array_t<bool> arr({N});
        auto ptr = arr.mutable_data();
        for (int j = 0; j < N; ++j) ptr[j] = mask[j];
        return arr;
    }

    bool get_enable_reward_shaping() const { return env.enable_reward_shaping; }
    void set_enable_reward_shaping(bool val) { env.enable_reward_shaping = val; }

    int num_actions() const { return env.num_actions; }
    int height() const { return env.state.height; }
    int width() const { return env.state.width; }
    int step_count() const { return env.step_count; }

    std::optional<std::tuple<int, int, int>> action_id_to_position_direction(int action_id) {
        return env.action_id_to_position_direction(action_id);
    }

    std::optional<std::vector<int>> find_path_to_push(int box_y, int box_x, int direction) {
        return env.find_path_to_push(box_y, box_x, direction);
    }

    State get_state() const { return env.state; }
    std::vector<std::vector<std::string>> get_matrix() const { return env.state.get_matrix(); }
    std::vector<std::pair<int, int>> get_boxes_position() const { return env.state.get_boxes_position(); }
    std::vector<std::pair<int, int>> get_docks_position() const { return env.state.get_docks_position(); }
    std::pair<int, int> get_worker_position() const { return env.state.get_worker_position(); }
    bool is_complete() const { return env.state.is_complete(); }

private:
    py::array_t<float> obs_to_numpy() {
        auto vec = env.get_observation();
        int H = env.state.height, W = env.state.width;
        return py::array_t<float>({4, H, W}, vec.data());
    }
};

PYBIND11_MODULE(sokoban_cpp, m) {

    py::class_<State>(m, "State")
        .def(py::init<>())
        .def(py::init<const std::vector<std::vector<std::string>>&>(), py::arg("input_matrix"))
        .def("get_matrix", &State::get_matrix)
        .def("get_matrix_element", &State::get_matrix_element, py::arg("y"), py::arg("x"))
        .def("get_worker_position", &State::get_worker_position)
        .def("get_boxes_position", &State::get_boxes_position)
        .def("get_docks_position", &State::get_docks_position)
        .def("worker_can_move", &State::worker_can_move, py::arg("dy"), py::arg("dx"))
        .def("worker_can_push_box", &State::worker_can_push_box, py::arg("dy"), py::arg("dx"))
        .def("is_complete", &State::is_complete)
        .def("move_box", &State::move_box, py::arg("y_box"), py::arg("x_box"), py::arg("move_y"), py::arg("move_x"))
        .def("move", &State::move, py::arg("dy"), py::arg("dx"))
        .def("apply_push_action", &State::apply_push_action, py::arg("box_y"), py::arg("box_x"), py::arg("dy"), py::arg("dx"))
        .def_readonly("height", &State::height)
        .def_readonly("width", &State::width)
        .def_readonly("worker_y", &State::worker_y)
        .def_readonly("worker_x", &State::worker_x)
        .def_readonly("docks_position", &State::docks_position);

    py::class_<PySokobanEnv>(m, "SokobanEnv")
        .def(py::init<int, bool>(), py::arg("max_pushes") = 100, py::arg("avoid_dead_squares") = false)
        .def("set_level", &PySokobanEnv::set_level, py::arg("map_matrix"), py::arg("dead_squares"))
        .def("reset", &PySokobanEnv::reset)
        .def("step", &PySokobanEnv::step, py::arg("action"))
        .def("get_valid_action_mask", &PySokobanEnv::get_valid_action_mask)
        .def_property("enable_reward_shaping", &PySokobanEnv::get_enable_reward_shaping, &PySokobanEnv::set_enable_reward_shaping)
        .def_property_readonly("num_actions", &PySokobanEnv::num_actions)
        .def_property_readonly("height", &PySokobanEnv::height)
        .def_property_readonly("width", &PySokobanEnv::width)
        .def_property_readonly("step_count", &PySokobanEnv::step_count)
        .def("action_id_to_position_direction", &PySokobanEnv::action_id_to_position_direction)
        .def("find_path_to_push", &PySokobanEnv::find_path_to_push)
        .def("get_state", &PySokobanEnv::get_state)
        .def("get_matrix", &PySokobanEnv::get_matrix)
        .def("get_boxes_position", &PySokobanEnv::get_boxes_position)
        .def("get_docks_position", &PySokobanEnv::get_docks_position)
        .def("get_worker_position", &PySokobanEnv::get_worker_position)
        .def("is_complete", &PySokobanEnv::is_complete);
}
