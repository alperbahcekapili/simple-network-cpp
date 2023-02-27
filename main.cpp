#include "network.h"
int main(){
    int layer_sizes[4] = {2,3,2,1};
    Network n = Network(layer_sizes, 4);
    n.test();

}