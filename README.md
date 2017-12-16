# 介紹

這個lib是在Visual Studio 2015開發的C++專案

每個神經元以sigmoid function作為activation function.

利用倒傳遞梯度下降法作為學習方式

<OR_Logic_example> OR logic regression example.
<XOR_logic_example>XOR logic regression example.
<NN_MNIST> MNIST example code with C#.

# 使用方式

## Initial Neural Network Architecture.

neural_network *nn = new neural_network(layer_num, layer_neurons_num);

	Architecture example:  
	input_layer |  hidden_layer1 | hidden_layer2 | output_layer

	input_dimension is 4
	hidden_layer1 neuron dimension is 2
	hidden_layer2 neuron dimension is 2  					
	output_layer dimension is 1

	Then, layer_num = 4 , layer_neurons_num[] ={4,2,2,1}
	It's will initial every neurons weight & bias. 

## Traning Neural Network with backpropagation gradient descent

nn->train(target_output, input, learnling_factor);

	input is input layer data.

	target_output is what we expect the net output. 

	learnling_factor is grandeint descent parameter,this about net learning quality.

## Neural Network Output

### Output Layer Result

nn->output(input_temp);
	
	Will return double* output

### Every Single Layer Output

nn->ervery_layer_output(input_temp);

	example:
	it will return double** output(It include all layer's output,contain input layer.)

	output[0] is input layer data

	output[1] is hidden layer1 output

	output[2] is hidden_ ayer2 output

	output[3] is output layer output

 
