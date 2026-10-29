#include "step_info.h"

// Không cần cài đặt gì thêm.
// StepInfo và StepResult là Plain Old Data (POD) struct,
// khởi tạo trực tiếp bằng aggregate initialization trong sokoban_env.cpp.
// Việc convert sang py::dict / py::tuple sẽ làm ở file bindings.cpp.
