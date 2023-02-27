#pragma once
class LogLoss{
    public:
        // float* forward(float* yhat, float y[]);
        // float* backward(float* yhat, float y[]);
        float calculate(float yhat, float y);
        float calculate_r(float yhat, float y);
    };