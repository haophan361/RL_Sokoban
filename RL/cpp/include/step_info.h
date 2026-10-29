#pragma once
#include <vector>

struct StepInfo {
    bool invalid_action = false;
    bool solved         = false;
    bool deadlock       = false;
    bool timeout        = false;
    int  pushes         = 0;
    int  player_moves   = 0;
};

struct StepResult {
    std::vector<float> observation;
    float              reward = 0.0f;
    bool               done   = false;
    StepInfo           info;
};
