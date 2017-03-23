# Introduction

This is simple neural network lib for C++,using Visual Studio 2015 C++ Programing.

Ervery neuron using sigmoid function for activity function.

Using backpropagation gradient descent for learning method.

<OR_Logic_example> this project is OR logic regresion example.



# Using

## Initial Neural Network Architecture.

neural_network *nn = new neural_network(layer_num, layer_neurons_num);

	Architecture example:  
	input_layer -->  hidden_layer1 --> hidden_layer2 --> output_layer

	input dimension is 4
	hidden_layer1 neuron dimension is 2
	hidden_layer2 neuron dimension is 2  					
	output dimension is 1

	Then, layer_num = 4 , layer_neurons_num[] ={4,2,2,1}
	It's will initial every neurons weight & bias. 

## Traning Neural Network with backpropagation gradient descent

nn->nn_train(target_output, input, learnling_factor);

	input is input layer data.

	target_output is what we expect the net output. 

	learnling_factor is grandeint descent parameter,this about net learning quality.

## Neural Network Output

### Output Layer Output

nn->nn_output(input_temp);
	
	Will return double* output

### Every Single Layer Output

nn->nn_ervery_layer_output(input_temp);

	example:
	it will return double** output(It include all layer's output,contain input layer.)

	output[0] is input layer data

	output[1] is hidden layer1 output

	output[2] is hidden_ ayer2 output

	output[3] is output layer output

 
