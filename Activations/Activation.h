#pragma once
class Activation{
    public:
        virtual float* forward(float* z, int inputsize) = 0;
        virtual float* backward(float* z, int inputsize) = 0;
};