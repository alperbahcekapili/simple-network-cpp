#include "network.h"
#include "layer.h"
#include "Losses/LogLoss.h"
#include <iostream>
#include <fstream> 
#include <stdlib.h>

using namespace std;

Network::Network(int layer_sizes[], int layernum){

    this->network = (Layer*)malloc(sizeof(Layer) * layernum);
    this->layer_sizes = layer_sizes;
    // initialize each layer
    this->network[0] = Layer(layer_sizes[0], 0, false);

    for (int i = 1; i < layernum-1; i++)
    {
        // int neurons, int prev_layer_size, bool is_prediction)
        
        this->network[i] = Layer(layer_sizes[i], layer_sizes[i-1], false);
    }
    if(layernum>1){
        this->network[layernum-1] = Layer(layer_sizes[layernum-1], layer_sizes[layernum-2] , true);  
    }else
            this->network[0] = Layer(layer_sizes[0], 0, true);

    this->layers = layernum;
}   

void Network::LoadDataset(float* x[2], float* y){

    //make here generic


    string dataset_path = "/home/alpfischer/python_files/classification_dataset";
    
    ifstream dataset_file(dataset_path);
    string line;


    int index = 0;
    while (getline(dataset_file, line)){


        float x1 = stof(line.substr(0, line.find(" ")));
        float x2 = stof(line.substr(line.find(" "),line.find_last_of(" ")));
        float y_ = stof(line.substr(line.find_last_of(" "),line.length()));
        x[index] = (float*)malloc(2* sizeof(float));
        x[index][0] = x1;
        x[index][1] = x2;
        y[index] = y_;


        index+=1;
    }   

    dataset_file.close();
}

float Network::test(){

    // change here to be generic
    float* x[1000];
    float y[1000];
    LogLoss* l = new LogLoss();


    this->LoadDataset(x,y);

    float* lastact;
    float* temp;
    
    int total_samples = 1000;
    float total_error = 0;
    
    int epochs = 1000;
    for (int i = 0; i < epochs; i++)
    {
        total_error = 0;
            for (int j = 0; j < total_samples; j++)
        {
            lastact = this->forward(x[j],y[j]);
            temp = this->backward(x[j], y[j]);
            // cout << *lastact;
            total_error+= l->calculate(*lastact, y[j]);
        }

        cout << "After epoch: " << i+1 << " mean loss: " << total_error/1000 << endl;

    }
    
    


    return 1.0;
}

float* Network::forward(float* x, float y){
    float* old_activations = x;
    float* new_activations;
    for (int i = 0; i < this->layers; i++)
    {
        new_activations = this->network[i].forward(old_activations, y);
        old_activations = new_activations;
    }
    return old_activations;
}



float* Network::backward(float* x ,float y){
    float* old_dadbs = &y;
    float* new_dadbs;
    for (int i = this->layers-1; i > 0 ; i--)
    {
        new_dadbs = this->network[i].backward(x, y, old_dadbs, this->network[i-1].activations);
        old_dadbs = new_dadbs;
    }

    new_dadbs = this->network[0].backward(x, y, old_dadbs, x);
    old_dadbs = new_dadbs;

    return x;
}

