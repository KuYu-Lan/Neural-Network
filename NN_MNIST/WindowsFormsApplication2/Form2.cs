// Code by Ku Yu Lan
//
// Description:https://github.com/KuYu-Lan/neural_network_lib

using NN_Csharp_lib;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace WindowsFormsApplication2
{
    public partial class Form2 : Form
    {
        Form1 f1;
        Form3 f3; //Error Log Form
        neural_network nn;
        Thread t1;
        MNIST mnist = new MNIST(); 

        public Form2(Form1 f1)
        {
            InitializeComponent();
            this.f1 = f1;
        }

        private void Form2_Load(object sender, EventArgs e)
        {
           
        }


        private void Form2_FormClosing(object sender, FormClosingEventArgs e)
        {
         
        }
        public void STOP_train()
        {
            if(t1 != null)t1.Abort();
            button1.Enabled = true;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            f1.Train_start = false;
            //IsDisposed
            if(f3 != null)f3.Close();
            STOP_train();
            this.Close();
        }


        private void textBox1_KeyPress(object sender, KeyPressEventArgs e)
        {
            Number_Check(sender, e); //Input must be number

        }
        private void textBox2_KeyPress(object sender, KeyPressEventArgs e)
        {
            Number_Check(sender, e);//Input must be number
        }

        private void textBox3_KeyPress(object sender, KeyPressEventArgs e)
        {
            Number_Check(sender, e);//Input must be number
        }
        private void textBox6_KeyPress(object sender, KeyPressEventArgs e)
        {
            Number_Check(sender, e);//Input must be number
        }

        private void textBox7_KeyPress(object sender, KeyPressEventArgs e)
        {
            Number_Check(sender, e);//Input must be number
        }
        private void Number_Check(object sender, KeyPressEventArgs e)
        {
            //This function is check that the TextBox input value.
            TextBox temp = (TextBox)sender;
            if (!char.IsDigit(e.KeyChar) && !char.IsControl(e.KeyChar))
            {
                MessageBox.Show("Input Must Be Number",
                    "Important Notice");
                e.Handled = true; //already handle process,Do no more
            }
        }
   

        private void button1_Click(object sender, EventArgs e)
        {
            //Train NN


            //Check all Textbox
            float n;
            //NN output is [0,1]
            if (float.TryParse(textBox4.Text,out n))
            {
                if (n > 1 || n < 0)
                {
                    MessageBox.Show("Convergence Criteria range is 0 ~ 1",//wrong range
                        "Important Notice");
                    return;
                }
            }
            else
            {
                MessageBox.Show("Convergence Criteria Must Be Number",//wrong range
                        "Important Notice");
                return;
            }

            //TextBox can't be empty
            if(textBox1.Text == "" || textBox2.Text == "" || textBox3.Text == "" || textBox4.Text == "" || textBox5.Text == ""||textBox6.Text == "" || textBox7.Text == "")
            {
                MessageBox.Show("Some parameter not set",//wrong range
                        "Important Notice");
                return;
            }

            button1.Enabled = false;

            //NN Training
            f3 = new Form3(this);
            f3.Show();
            t1 = new Thread(NN_Training);
            t1.Start();
        }


        private unsafe void NN_Training()
        {
            int epoch = int.Parse(textBox1.Text);
            int iteration = int.Parse(textBox2.Text);
            double learning_factor = double.Parse(textBox3.Text);
            double up_bound = double.Parse(textBox5.Text);
            double lower_bound = double.Parse(textBox4.Text);
            
            int[] layer_neuron_num = new int[4];
            layer_neuron_num[0] = 784;      //Input layer Num
            layer_neuron_num[1] = Int32.Parse(textBox6.Text);// hidden_layer1 Num
            layer_neuron_num[2] = Int32.Parse(textBox7.Text);// hidden_layer2 Num
            layer_neuron_num[3] = 10;       //Output_layer, Output layer Neuron 0 ~ 9 represent number 0 ~ 9

            byte[] image_data = new byte[784]; //MNistread return image
            Int16 label_data;   //MNistread return label 
            double[] input_data = new double[784];//NN_train input data
            double[] target_data = new double[10];//NN_train target data
            bool check_data=false;//verify the nn_train function
            float []error_log = new float [iteration];
            float nn_error = 1; //compara every epoch performmance

            for (int epoch_count = 0; epoch_count < epoch; epoch_count++)
            {
                //init NN
                neural_network nn = NN_init(4, layer_neuron_num);

                //Change Form3 epoch label
                f3.Change_Label(epoch_count, false,100);

                //Training NN
                for (int iteration_count = 0; iteration_count < iteration; iteration_count++)
                {
                    //init error log 
                    error_log[iteration_count] = 0;

                    //Change Form3 iteration label
                    f3.Change_iteration_label(iteration_count);

                    //Get Test Label & Image to train NN
                    for (int Label_count = 0; Label_count < 60000; Label_count++) //MNIST Train dataset number of item is 60000 
                    {
                        //Read data
                        mnist.ReadImg("train-labels.idx1-ubyte", "train-images.idx3-ubyte", Label_count, image_data);
                        label_data = mnist.label;

                        //transform image_data to input data
                        for (int input_count = 0; input_count < 784; input_count++)
                            input_data[input_count] = image_data[input_count];

                        //Set Target Output data
                        for (int output_count = 0; output_count < 10; output_count++)
                            target_data[output_count] = 0;
                        target_data[label_data] = 1; //expect NN Output 

                        //train_nn
                        fixed (double* p1 = target_data)
                        {
                            fixed (double* p2 = input_data)
                            {
                                //if check output is wrong then train NN
                                check_data = nn.Classification_train(p1, p2, learning_factor, up_bound,lower_bound);
                            }
                        }
                        if (check_data) error_log[iteration_count]++;
                        
                    }
                    error_log[iteration_count] = error_log[iteration_count] / 60000; //compute error average

                    //Save the best performance data
                    if (nn_error > error_log[iteration_count])
                    {
                        nn_error = error_log[iteration_count];
                        this.nn = nn;
                    }
                }
                //show error log
                f3.show_to_graph(error_log,epoch_count);
            }
            //Train end
            f3.Change_Label(epoch - 1, true,nn_error);
        }

        public unsafe void Out_NN_Parameter()
        {
            Form1.nn = this.nn; //Set to Form1 nn

            //Write file to 'NN_Parameter'
            //Write Out train NN Parameter 
            FileStream fsSource = new FileStream("NN_Parameter", FileMode.Create, FileAccess.Write);

            byte[] temp_byte = new byte[20];
            //write out NN Architecture
            temp_byte = BitConverter.GetBytes(4); //layer_num
            fsSource.Write(temp_byte, 0, 4);
  
            temp_byte = BitConverter.GetBytes(784);
            fsSource.Write(temp_byte, 0, 4); //input length

            temp_byte = BitConverter.GetBytes(Int32.Parse(textBox6.Text));
            fsSource.Write(temp_byte, 0, 4); //hidden layer 1 length

            temp_byte = BitConverter.GetBytes(Int32.Parse(textBox7.Text));
            fsSource.Write(temp_byte, 0, 4); //hidden layer 2 length

            temp_byte = BitConverter.GetBytes(10);
            fsSource.Write(temp_byte, 0, 4); //output layer length

            int weight_num = this.nn.Get_weight_num();

            temp_byte = BitConverter.GetBytes(weight_num);
            fsSource.Write(temp_byte, 0, 4); //weight length

            double* p1 = nn.Get_weight();
            for(int i = 0; i < weight_num; i++)
            {
                temp_byte = BitConverter.GetBytes(p1[i]);
                fsSource.Write(temp_byte, 0, 8); //double is 8bytes
            }

            int bias_num = this.nn.Get_bias_num();
            temp_byte = BitConverter.GetBytes(bias_num);
            fsSource.Write(temp_byte, 0, 4); //bias length
            p1 = this.nn.Get_bias();
            for (int i = 0; i < bias_num; i++)
            {
                temp_byte = BitConverter.GetBytes(p1[i]);
                fsSource.Write(temp_byte, 0, 8); //double is 8bytes
            }
            fsSource.Close();
        }

        private unsafe neural_network NN_init(int layer_num , int[] layer_neuron_num)
        {
            //Init NN
            fixed (int* p = layer_neuron_num)
            {
                neural_network nn = new neural_network(layer_num, p);
                return nn;
            }
           
        }

    }
}
