#include "LogLoss.h"
#include <cmath>

// float* LogLoss::forward(float yhat[], float y[]){
//     float* losses = new float[sizeof(yhat)];
//     for(size_t i = 0; i< sizeof(yhat); i++){
//         losses[i] = calculate(yhat[i], y[i]);
//     }
//     return losses;
// }
float LogLoss::calculate(float yhat, float y){
    return -( y*log(yhat) + (1-y)*log(1-yhat));
}

// float* LogLoss::backward(float yhat[], float y[]){
//     float* losses = new float[sizeof(yhat)];
//     for(size_t i = 0; i< sizeof(yhat); i++){
//         losses[i] = calculate(yhat[i], y[i]);
//     }
//     return losses;
// }

float LogLoss::calculate_r(float yhat, float y){
    return ((1-y) / (1-yhat)) - (y/yhat);
}

