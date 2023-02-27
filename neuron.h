
#pragma once
#include "Activations/Sigmoid.h"
#include "Losses/LogLoss.h"

class Neuron{
    public:    
        Neuron(int input_size, bool is_prediction, float lr);
        float forward(float* x, float y);
        float* backward(float* x, float y, float dldab, float* old_activations);
        Sigmoid* activation;
        LogLoss* loss;
        int input_size;
        bool is_prediction;
        float* weights;
        float bias;
        int nsamples;
        float dldab;
        float dlda;
        float dadz;
        float dzdw;
        float z;
        float a;
        float lr;
        int prevlayersize;

};