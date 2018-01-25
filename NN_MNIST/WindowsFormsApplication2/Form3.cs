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
using System.Windows.Forms.DataVisualization.Charting;

namespace WindowsFormsApplication2
{
    public partial class Form3 : Form
    {
        Form2 f2;
        public Form3(Form2 f2)
        {
            InitializeComponent();
            this.f2 = f2;
        }

        private void Form3_Load(object sender, EventArgs e)
        {
            chart1.Series.Clear();
            label3.Text = "";
            button1.Enabled = false;
        }
        public void Change_Label(int epoch,bool finish,float error)
        {
            label2.Invoke((MethodInvoker)delegate
            {
                label2.Text = "Now epoch: " + epoch;
            });


            if (finish)
            {
                label1.Invoke((MethodInvoker)delegate
                {
                    label1.Text = "NN Training End";
                });

                label3.Invoke((MethodInvoker)delegate
                {
                    label3.Text = "Best Classification Performmance:" + error ;
                });
                button1.Invoke((MethodInvoker)delegate
                {
                    button1.Enabled = true;
                });
            }
        }

        public void Change_iteration_label(int iteration_count)
        {
            label4.Invoke((MethodInvoker)delegate
            {
                label4.Text = "Now Iteration:" + iteration_count;
            });
        }



        public void show_to_graph(float[] array,int epoch)
        {
            Series series1 = new Series("epoch:" + epoch, array.Length);

            series1.ChartType = SeriesChartType.Line;
            for (int index = 0; index < array.Length; index++)
                series1.Points.AddXY(index, array[index]);
            //Enabel thread access UI
            chart1.Invoke((MethodInvoker)delegate
            {
                chart1.Series.Add(series1);
            });
        }

        private void button2_Click(object sender, EventArgs e)
        {
            f2.STOP_train();
            this.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            f2.Out_NN_Parameter();
        }

        private void Form3_FormClosing(object sender, FormClosingEventArgs e)
        {
            f2.STOP_train();

        }
    }
}
