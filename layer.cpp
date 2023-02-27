#include "layer.h"
#include "neuron.h"

#include <stdlib.h>

#define lr  0.001

Layer::Layer(int neurons, int prev_layer_size, bool is_prediction){
    this->layer = (Neuron*)malloc(sizeof(Neuron) * neurons);
    this->neurons = neurons;
    this->prev_layer_size = prev_layer_size;
    for (int i = 0; i < neurons; i++)
    {
        layer[i] = Neuron(prev_layer_size, is_prediction, lr);
    }
}


float* Layer::forward(float* x, float y){
    float* activations_ = new float[this->neurons];
    for (int i = 0; i < this->neurons; i++)
    {
        this->layer[i].forward(x, y);
        activations_[i] = this->layer[i].a;
    }
    this->activations = activations_;
    return activations_;
}


float* Layer::backward(float* x, float y, float* dldab, float* old_activations){
    // here we should run backward on each neuron
    // and then recieve dldab' s from them

    // here should not every value coming from forward neurons be the same?
    // because each neuron try to take partial derivative respect to preceding neurons activation
    // check it here in future


    // may cause memory issues, should free in future
    float* dldab_ = (float*)malloc(sizeof(float) * this->prev_layer_size);

    for (int i = 0; i < this->neurons; i++)
    {
        float* temp = this->layer[i].backward(x, y, dldab[i], old_activations);
        for (int j = 0; j < prev_layer_size; j++)
        {
            dldab_[j] += temp[j];
        }
    }
    
    // here these values will be used for 
    return dldab_;

}