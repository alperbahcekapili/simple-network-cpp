#include <cmath>

#include "Sigmoid.h"
#define euler 2.718818

// TODO: Here change declaration to include input size
float* Sigmoid::forward(float z[], int inputsize){
    float* newz = new float[inputsize];
    for(int i = 0; i< inputsize; i++){
        newz[i] = sigmoid(z[i]);
    }
    return newz;
}

float Sigmoid::sigmoid(float z){
    return 1/ (1 + pow(euler, z));
}

float* Sigmoid::backward(float z[], int inputsize){
    float* newz = new float[inputsize];
    for(int i = 0; i< inputsize; i++){
        newz[i] = sigmoid_r(z[i]);
    }
    return newz;
}

float Sigmoid::sigmoid_r(float z){
    return sigmoid(z) * (1- sigmoid(z));
}

Sigmoid::Sigmoid(){
    
}