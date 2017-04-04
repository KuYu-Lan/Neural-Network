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
	this->layer_neurons_num = new int[layer_num];

	for (int i = 0; i < layer_num - 1; i++) {
		weight_num += layer_neurons_num[i] * layer_neurons_num[i + 1];
		bias_num += layer_neurons_num[i + 1];

		this->layer_neurons_num[i] = layer_neurons_num[i];
	}
	this->layer_neurons_num[layer_num - 1] = layer_neurons_num[layer_num - 1];

	this->weight_num = weight_num;
	this->bias_num = bias_num;
	weight = new double[weight_num];
	bias = new double[bias_num];

	srand(time(NULL));//init rand

					  //Init Weight & Bias
	for (int i = 0; i < weight_num; i++) {
		weight[i] = (double)(rand() % 2000) / 1000 - 1; //Init neurno = [1 ~ -1]
	}
	for (int i = 0; i < bias_num; i++) {
		bias[i] = (double)(rand() % 2000) / 1000 - 1; //Init bias = [1 ~ -1]
	}
}

neural_network::~neural_network()
{
	//free memory
	delete[]bias;
	delete[]layer_neurons_num;
	delete[]weight;
}

void neural_network::set_parameter(double* weight, double* bias) {
	for (int i = 0; i<this->weight_num; i++)
		this->weight[i] = weight[i];

	for (int i = 0; i < this->bias_num; i++)
		this->bias[i] = bias[i];

}




double** neural_network::ervery_layer_output(double* input_value) {
	int weight_offset = 0, bias_offset = 0;
	double **output = new double *[this->layer_num];

	output[0] = input_value; //ouput[0] is input_layer
							 //Feedforward Network
	for (int i = 1; i < layer_num; i++) {
		double *temp = NULL;
		temp = Fully_Connect_layer(output[i - 1], weight + weight_offset, bias + bias_offset, i); //Every Single Layer Output
		output[i] = temp; //point to ervery layer output

		weight_offset += layer_neurons_num[i - 1] * layer_neurons_num[i];
		bias_offset += layer_neurons_num[i];
	}

	return output;
}

double* neural_network::output(double* input_value) {
	int weight_offset = 0, bias_offset = 0;
	double *output = input_value;
	double *temp = NULL;

	//Feedforward Network
	for (int i = 1; i < layer_num; i++) {
		temp = output;
		output = Fully_Connect_layer(temp, weight + weight_offset, bias + bias_offset, i); //Every Single Layer Output
		weight_offset += layer_neurons_num[i - 1] * layer_neurons_num[i];
		bias_offset += layer_neurons_num[i];

		//free memory,if is input layer then don't free memory 
		if (i != 1)
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

void neural_network::train(double* target_value, double* input_value, double learnling_factor)
{
	double** output = NULL;

	output = this->ervery_layer_output(input_value);
	Update_parameter(output, target_value, learnling_factor);
}

bool neural_network::Classification_train(double* target_value, double* input_value, double learnling_factor, double up_bound, double lower_bound) {
	double** output = NULL;
	bool check = false;

	output = this->ervery_layer_output(input_value);

	//Check NN_output Convergence Criteria
	for (int i = 0; i < layer_neurons_num[layer_num - 1]; i++) {
		if ((*(output[layer_num - 1] + i) > lower_bound) && target_value[i] == 0)
			check = true;
		else if ((*(output[layer_num - 1] + i) < up_bound) && target_value[i] == 1)
			check = true;
	}

	//If not suffice convergence criteria then train nn
	if (check) {
		Update_parameter(output, target_value, learnling_factor);
		for (int i = 1; i < layer_num; ++i) {
			delete[] output[i];
		}
		delete[] output;
		return true;
	}
	for (int i = 1; i < layer_num; ++i) {
		delete[] output[i];
	}
	delete[] output;

	return false;
}

double * neural_network::Get_weight()
{
	return this->weight;
}

double * neural_network::Get_bias()
{
	return this->bias;
}

int neural_network::Get_bias_num()
{
	return this->bias_num;
}

int neural_network::Get_weight_num()
{
	return this->weight_num;
}



void neural_network::Update_parameter(double** output, double *target_value, double learnling_factor) {
	double* weight;
	double* bias;
	double* delta_a = NULL;
	double* delta_z = NULL;
	double* delta_w = NULL;
	int weight_offset = 0, bias_offset = 0, layer_weight_num;

	//Network Backpropagation ,  use gradient descent
	for (int i = layer_num - 1; i > 0; i--) {
		layer_weight_num = layer_neurons_num[i] * layer_neurons_num[i - 1];
		weight_offset += layer_weight_num;
		bias_offset += layer_neurons_num[i];

		//get sigle layer weight & bias
		weight = this->weight + weight_num - weight_offset;
		bias = this->bias + bias_num - bias_offset;

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
			for (int k = 0; k < layer_neurons_num[i - 1]; k++) {
				delta_w[layer_neurons_num[i - 1] * j + k] = *(output[i - 1] + k) * delta_z[j];
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

		delete[]delta_w; //free delta_w
		delete[]delta_z; //free delta_z
	}
	delete[] delta_a; //free delta_a
}




