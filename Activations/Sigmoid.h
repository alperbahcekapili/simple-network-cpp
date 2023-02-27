#pragma once

#include "Activation.h"
class Sigmoid: Activation{
    public:    
        Sigmoid();
        float* forward(float z[], int inputsize);
        float sigmoid(float z);
        float* backward(float z[], int inputsize);
        float sigmoid_r(float z);
};