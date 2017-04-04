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
	neural_network(int layer_num, int* layer_neurons_num);//Init Neural Net Topology
	~neural_network();
	double** ervery_layer_output(double* intput_value);
	double* output(double* input_value);
	void train(double* target_value, double* input_value, double learnling_factor);
	bool Classification_train(double* target_value, double* input_value, double learnling_factor, double up_bound, double lower_bound);  //overload,Plus Ouput Convergence Criteria  
	double* Get_weight();
	double* Get_bias();
	int Get_bias_num();
	int Get_weight_num();
	void set_parameter(double* weight, double* bias);

private:
	int* layer_neurons_num;  //every layer neuron num
	int layer_num, weight_num, bias_num;
	double* weight; //neural net weight
	double* bias; //neural net bias

	double* Fully_Connect_layer(double* input_value, double* weight, double* bias, int layer);//Compute Single Layer Output
	void Update_parameter(double** output, double *target_value, double learnling_factor); //Use Backpropagation gradient to update weight & bias 
};
