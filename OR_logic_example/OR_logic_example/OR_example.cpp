#include "neural_network.h"
#include <iostream>

using namespace std;
int main() {
	int layer_num = 2;
	int* layer_neurons_num = new int[layer_num];
	neural_network *nn;
	
	//Neural Network Architecture Initial
	layer_neurons_num[0] = 2; 
	layer_neurons_num[1] = 1;

	//OR logic
	double input_value[4][2] = {{0,0},{0,1},{1,0},{1,1}};
	double target_output[4] = {0 , 1 ,1 ,1};

	cout << "OR logic" << endl << "input\toutput" << endl << "0 0\t0"<<endl << "0 1\t1" << endl << "1 0\t1" << endl << "1 1\t1"<<endl <<endl;
	
	double input_temp[2];
	double learnling_factor = 0.2;
	//initial network
	nn = new neural_network(layer_num, layer_neurons_num);

	//train Neural Network
	for (int iteration = 0; iteration < 200; iteration++) {
		for (int i = 0; i < 4; i++) {
			//init input
			input_temp[0] = input_value[i][0];
			input_temp[1] = input_value[i][1];
			//train network
			nn->nn_train(&target_output[i], input_temp, learnling_factor);
		}
	}

	//Test Neural Network output Data
	cout << "NN_OR logic" << endl << "input\toutput" << endl;
	for (int i = 0; i < 4; i++) {
		input_temp[0] = input_value[i][0];
		input_temp[1] = input_value[i][1];
		cout << input_value[i][0] << " " << input_value[i][1] << "\t";
		cout << *(nn->nn_output(input_temp)) << endl;
	}
	
	

	system("pause");
	return 0;
}