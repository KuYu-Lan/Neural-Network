// neural_network.h by Ku Yu Lan
//
// Description:https://github.com/KuYu-Lan/neural_network_lib





#include <stdlib.h>
#include <time.h>
#include <math.h> 
#include<iostream>
using namespace std;

class neural_network
{
public:
	neural_network(int layer_num,int* layer_neurons_num);//Init Neural Net Topology
	double** nn_ervery_layer_output(double* intput_value);
	double* nn_output(double* input_value);
	void nn_train(double* target_value , double* input_value,double learnling_factor);

private:
	int* layer_neurons_num;  //every layer neuron num
	int layer_num,weight_num,bias_num;
	double* nn_weight; //neural net weight
	double* nn_bias; //neural net bias

	double* Fully_Connect_layer(double* input_value, double* weight, double* bias, int layer);//Compute Single Layer Output
};
