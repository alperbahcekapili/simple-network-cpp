CC = g++ 
main: main.o network.o neuron.o layer.o LogLoss.o Sigmoid.o
	$(CC) -g -Wall main.o network.o neuron.o layer.o LogLoss.o Sigmoid.o -o main

network.o: network.cpp  network.h neuron.h layer.h Losses/LogLoss.h Activations/Sigmoid.h
	g++ -c -Wall -g network.cpp

main.o: main.cpp network.o network.cpp layer.h
	g++ -c -Wall -g main.cpp

layer.o: layer.cpp  neuron.h layer.h Losses/LogLoss.h Activations/Sigmoid.h
	g++ -c -Wall -g layer.cpp

neuron.o: neuron.cpp neuron.h layer.h Losses/LogLoss.h Activations/Sigmoid.h
	g++ -c -Wall -g neuron.cpp

Sigmoid.o: Activations/Sigmoid.cpp  Activations/Sigmoid.h Activations/Activation.o
	g++ -c -Wall -g Activations/Sigmoid.cpp

Activation.o: -g Activations/Activation.cpp Activations/Activation.h
	g++ -c -Wall -g Activations/Activation.cpp

LogLoss.o: Losses/LogLoss.cpp  Losses/LogLoss.h
	g++ -c -Wall -g Losses/LogLoss.cpp

clean:
	rm *.o
	rm main