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

private:
    py::array_t<float> obs_to_numpy() {
        auto vec = env.get_observation();
        int H = env.state.height, W = env.state.width;
        return py::array_t<float>({4, H, W}, vec.data());
    }
};

PYBIND11_MODULE(sokoban_cpp, m) {

    py::class_<PySokobanEnv>(m, "SokobanEnv").def(py::init<int, bool>(),py::arg("max_pushes") = 100,
                py::arg("avoid_dead_squares") = false)
        .def("set_level",  &PySokobanEnv::set_level,py::arg("map_matrix"), py::arg("dead_squares"))
        .def("reset", &PySokobanEnv::reset)
        .def("step", &PySokobanEnv::step, py::arg("action"))
        .def("get_valid_action_mask", &PySokobanEnv::get_valid_action_mask)
        .def_property_readonly("num_actions", &PySokobanEnv::num_actions)
        .def_property_readonly("height", &PySokobanEnv::height)
        .def_property_readonly("width", &PySokobanEnv::width)
        .def_property_readonly("step_count", &PySokobanEnv::step_count)
        .def("action_id_to_position_direction", &PySokobanEnv::action_id_to_position_direction)
        .def("find_path_to_push", &PySokobanEnv::find_path_to_push);
}
