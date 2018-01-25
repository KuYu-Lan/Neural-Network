// Code by Ku Yu Lan
//
// Description:https://github.com/KuYu-Lan/neural_network_lib

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using NN_Csharp_lib;
using System.IO;

namespace WindowsFormsApplication2
{
    public partial class Form1 : Form
    {
        static public neural_network nn;
        private MNIST mnist = new MNIST();
        public bool Train_start = false; //Form2 display controll

        int Test_label;//MNIST label
        Form2 form2; //train NN FORM
        byte[] image_data; //read MNIST image data

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            changeImg();
            label1.Text = "";
            init_NN();
        }

        unsafe static private void init_NN()
        {
            //Read NN parameter file
            if (!File.Exists("NN_Parameter"))
            {
                MessageBox.Show("Open NN_Parameter fail!!");
                Console.WriteLine("Open NN_Parameter fail!!");
                return;
            }
            FileStream fsSource = new FileStream("NN_Parameter", FileMode.Open, FileAccess.Read);
            byte[] temp_byte = new byte[8];
            
            fsSource.Read(temp_byte, 0, 4);
            int layer_num = BitConverter.ToInt32(temp_byte, 0);

            //init nn layer parameter
            int[] layer_neuron_num = new int[layer_num];
            for (int i = 0; i < layer_num; i++)
            {
                fsSource.Read(temp_byte, 0, 4); //int is 4bytes
                layer_neuron_num[i] = BitConverter.ToInt32(temp_byte, 0);
            }


            //init nn weight parameter
            fsSource.Read(temp_byte, 0, 4); //int is 4bytes
            int weight_num = BitConverter.ToInt32(temp_byte, 0);
            double[] weight = new double[weight_num];
            for (int i = 0; i < weight_num; i++)
            {
                fsSource.Read(temp_byte, 0, 8); //double is 8bytes
                weight[i] = BitConverter.ToDouble(temp_byte, 0);
            }

            fsSource.Read(temp_byte, 0, 4); //int is 4bytes
            int bias_num = BitConverter.ToInt32(temp_byte, 0);
            double[] bias = new double[bias_num];
            for (int i = 0; i < bias_num; i++)
            {
                fsSource.Read(temp_byte, 0, 8); //double is 8bytes
                bias[i] = BitConverter.ToDouble(temp_byte, 0);
            }
            
            fsSource.Close();
            //nn = NN_init(layer_num, layer_neuron_num, weight, bias);
            //Init NN
            fixed (int* p = layer_neuron_num)
            {
                nn = new neural_network(layer_num, p); 
                fixed (double* p1 = weight)
                {
                    fixed (double* p2 = bias)
                    {
                        nn.set_parameter(p1, p2);
                    }
                }
            }        
        }

        private void button1_Click(object sender, EventArgs e)
        {
            changeImg();
        }

        private void changeImg()
        {
            //Change MNIST Image
            Random rnd = new Random();

            image_data = new byte[784];
            mnist.ReadImg("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte", rnd.Next(0, 9999), image_data);
            this.Test_label = mnist.label;

            //Show img to picturebox
            Bitmap img = new Bitmap(28, 28);
            Color curColor;
            for (int n1 = 0; n1 < 28; n1++)
            {
                for (int n2 = 0; n2 < 28; n2++)
                {
                    int temp = image_data[n1 * 28 + n2];
                    curColor = Color.FromArgb(temp, temp, temp);
                    img.SetPixel(n2, n1, curColor);

                }
            }
            pictureBox1.Image = img;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            //Use NN Classification
            if (nn == null)
                MessageBox.Show("NN Not Initialization",
                    "Important Note",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Exclamation,
                    MessageBoxDefaultButton.Button1);
            else
            {
                double[] input_data = new double[784];
                double[] output_data = new double[10];
                //transform image_data to input data
                for (int input_count = 0; input_count < 784; input_count++)
                    input_data[input_count] = image_data[input_count];
                //Get NN Output
                unsafe
                {
                    fixed (double* p = input_data)
                    {
                        double *p2 = nn.output(p);
                        for (int i = 0; i < 10; i++)
                            output_data[i] = p2[i];
                    }
                }
                double maxValue = output_data.Max();

                int max = output_data.ToList().IndexOf(maxValue);
                label1.Text = max.ToString();
            }
        }


        private void button4_Click(object sender, EventArgs e)
        {
            if (!Train_start)
            {
                Train_start = true;
                form2 = new Form2(this);
                form2.Show();
            }
        }
    }
}
