#pragma once
#include "layer.h"

class Network{
        public:
            Layer* network;
            float test();
            void LoadDataset(float* x[2], float* y);
            float* forward(float* x, float y);
            float* backward(float* x ,float y);
            Network(int* layer_sizes, int layernum);
            int layers;
            int* layer_sizes;

};