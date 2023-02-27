#include "neuron.h"
#include <iostream>

using namespace std;

Neuron::Neuron(int input_size, bool is_prediction, float lr){

    this->activation = new Sigmoid();
    this->loss = new LogLoss();
    this->input_size = input_size;
    this->is_prediction = is_prediction;
    this-> nsamples = 0;
    this->lr = lr;
    // initialize weights
    this->weights = new float[input_size];
    for(int i = 0; i< input_size; i++){
            this->weights[i] = 1.0;
    }

    this->bias = 1;
    this->dldab = 0;
    this->dlda = 0;
    this->dadz = 0;
    this->dzdw = 0;
    this->z = 0;
    this->a = 0;    
}


float Neuron::forward(float* x, float y){

    this->nsamples +=1;
    float z = 0;
    for(int i = 0; i< this->input_size; i++){
        z+= (this->weights[i]*x[i]);
    }

    z+=this->bias;

    float a = this->activation->sigmoid(z);


    this->a = a;
    this->z = z;

    // if this is the last neuron then calculate error
    if (this->is_prediction){
        float yhat = a;
        float loss = this->loss->calculate(yhat, y);

        // std::cout << "\nPrediction: " << yhat << ", loss: " << loss << ", nsamples: "<< this->nsamples << std::endl;

        return loss;
    }
    return a;

}

float* Neuron::backward(float* x, float y, float dldab, float* old_activations){

    if(this->is_prediction)
        this->dlda = (1-y)/(1-this->a) - y/this->a;
    else
        this->dlda = dldab;
    
    float dadz = this->activation->sigmoid(this->z)*(1-this->activation->sigmoid(z));
    int prevlayersize = this->input_size;
    //here this will be a vector
    float* dzdw = new float[prevlayersize];
    float* dldw = new float[prevlayersize];
    float* dldab_ = new float[prevlayersize];
    for (int i = 0; i <prevlayersize; i++)
    {
        // change this to previous neuron's activation
        dzdw[i] = old_activations[i];
        dldw[i] = dlda * dadz * dzdw[i];
        
        // we should return the value below
        dldab_[i] = weights[i]*dadz*dlda;

        //update parameters
        this->weights[i] += this->lr * dldw[i];
        // cout << dldw[i];
    }   
    return dldab_;
}