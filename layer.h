#pragma once
#include "neuron.h"
class Layer{
    public:
        Layer(int neurons, int prev_layer_size, bool is_prediction);
        Neuron* layer; 
        float* activations;
        int neurons;
        int prev_layer_size;
        float* forward(float* x, float y);
        float* backward(float* x, float y, float* dldab, float* old_activations);
};