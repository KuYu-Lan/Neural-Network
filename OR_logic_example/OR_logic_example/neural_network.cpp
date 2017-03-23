// neural_network.cpp by Ku Yu Lan
// 
// Description:https://github.com/KuYu-Lan/neural_network_lib

#include "neural_network.h"

neural_network::neural_network(int layer_num, int* layer_neurons_num)
{
//Initial Neural net Parameter
int weight_num = 0;
int bias_num = 0;

if (layer_num < 2) {
	return; //at least 2 layer
}
this->layer_num = layer_num;
this->layer_neurons_num = layer_neurons_num;

for (int i = 0; i < layer_num - 1; i++) {
	weight_num += layer_neurons_num[i] * layer_neurons_num[i + 1];
	bias_num += layer_neurons_num[i + 1];
}
this->weight_num = weight_num;
this->bias_num = bias_num;
nn_weight = new double[weight_num];
nn_bias = new double[bias_num];

srand(time(NULL));//init rand

//Init Weight & Bias
for (int i = 0; i < weight_num; i++) {
	nn_weight[i] = (double)(rand() % 2000) / 1000 - 1; //Init neurno = [1 ~ -1]
}
for (int i = 0; i < bias_num; i++) {
	nn_bias[i] = (double)(rand() % 2000) / 1000 - 1; //Init bias = [1 ~ -1]
}
}

double** neural_network::nn_ervery_layer_output(double* input_value) {
	int weight_offset = 0, bias_offset = 0;
	double **output = new double *[this->layer_num];

	output[0] = input_value; //ouput[0] is input_layer
	//Feedforward Network
	for (int i = 1; i < layer_num; i++) {
		double *temp = NULL;
		temp = Fully_Connect_layer(output[i-1], nn_weight + weight_offset, nn_bias + bias_offset, i); //Every Single Layer Output
		output[i] = temp; //point to ervery layer output

		weight_offset += layer_neurons_num[i - 1] * layer_neurons_num[i];
		bias_offset += layer_neurons_num[i];
	}
	return output;
}

double* neural_network::nn_output(double* input_value) {
	int weight_offset = 0, bias_offset = 0;
	double *output = input_value;
	double *temp = NULL;

	//Feedforward Network
	for (int i = 1; i < layer_num; i++) {
		temp = output;
		output = Fully_Connect_layer(temp, nn_weight + weight_offset, nn_bias + bias_offset, i); //Every Single Layer Output
		weight_offset += layer_neurons_num[i - 1] * layer_neurons_num[i];
		bias_offset += layer_neurons_num[i];

		//free memory,if is input layer then don't free memory 
		if( i != 1) 
			delete[]temp;
	}


	return output;
}

double* neural_network::Fully_Connect_layer(double* input_value, double* weight, double* bias, int layer)
{
	int output_size = layer_neurons_num[layer];
	double* output = new double[output_size];
	int weight_offset = 0;

	//Compute Ouput_i = sum(input_j * Weight_ji) + bias_i;
	//Then,   Output_i = A(Ouput_i);    (A is activity function ,in here is sigmoid function) 
	for (int i = 0; i < layer_neurons_num[layer]; i++)
	{
		output[i] = 0;
		weight_offset = i*layer_neurons_num[layer - 1];
		for (int j = 0; j < layer_neurons_num[layer - 1]; j++) {
			output[i] += input_value[j] * weight[weight_offset + j];
		}
		output[i] += output[i] + bias[i]; //Add Bias
		output[i] = 1 / (1 + exp(-output[i])); //Sigomoid function
	}


	return output;
}

void neural_network::nn_train(double* target_value, double* input_value, double learnling_factor)
{
	double** output = NULL;
	double* delta_a = NULL;
	double* delta_z = NULL;
	double* delta_w = NULL;

	double* weight;
	double* bias;

	int weight_offset = 0, bias_offset = 0, layer_weight_num;
	
	output = this->nn_ervery_layer_output(input_value);
	

	//Network Backpropagation ,  use gradient descent
	for (int i = layer_num - 1; i > 0; i--) {
		layer_weight_num = layer_neurons_num[i] * layer_neurons_num[i - 1];
		weight_offset += layer_weight_num;
		bias_offset += layer_neurons_num[i];
		

		//get sigle layer weight & bias
		weight = nn_weight + weight_num - weight_offset;
		bias = nn_bias + bias_num - bias_offset;

		if (i == layer_num - 1) {  //i is Output_Layer
			//delta_a_j = -(t_j - A(Ouput_j)) , A() is activity function
			delta_a = new double[layer_neurons_num[i]];
			for (int j = 0; j < layer_neurons_num[i]; j++)
				delta_a[j] = -(target_value[j] - *(output[i] + j));
		}

		//delta_z_j = delta_a_j * A(Ouput_j) * (1 - A(Ouput_j)) 
		delta_z = new double[layer_neurons_num[i]];
		for (int j = 0; j < layer_neurons_num[i]; j++) {
			delta_z[j] = delta_a[j] * *(output[i] + j) * (1 - *(output[i] + j));
			//update bias
			//bias_j = bias_j - learnling_factor * delta_z_j 
			bias[j] -= learnling_factor * delta_z[j];
		}

		//delta_w_kj = delta_z_j * A(Ouput_k) , A(Ouput_k) is «e¤@¼h¿é¥X  
		delta_w = new double[layer_weight_num];
		for (int j = 0; j < layer_neurons_num[i]; j++) {
			for (int k = 0; k < layer_neurons_num[i-1]; k++) {
				delta_w[layer_neurons_num[i - 1] *j + k] = *(output[i - 1] + k) * delta_z[j];
			}
		}

		delete[]delta_a; //free delta_a
		delta_a = new double[layer_neurons_num[i - 1]];

		//delta_a_j = sum(delta_z_k * weight_kj)
		for (int j = 0; j < layer_neurons_num[i - 1]; j++) {
			delta_a[j] = 0;
			for (int k = 0; k < layer_neurons_num[i]; k++) {
				delta_a[j] += delta_z[k] * weight[j + k* layer_neurons_num[i - 1]];
			}
		}

		//update weight
		//weight_j = weight_j - learnling_factor * delta_w_j 
		for (int j = 0; j < layer_weight_num; j++)
		{
			weight[j] -= learnling_factor *  delta_w[j];
		}

		delete[]output[i];//free output
		delete[]delta_w; //free delta_a
		delete[]delta_z; //free delta_a
	}

}