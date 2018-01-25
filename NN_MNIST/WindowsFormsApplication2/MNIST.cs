using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApplication2
{
    class MNIST
    {
        public Int16 label = -1;
        public MNIST()
        {
        }
        public void ReadImg(String label_path, String image_path, int offset, byte[] image_data)
        {
            int n;
  
            byte[] tempBytes = new byte[4];
            FileStream fsSource = new FileStream(label_path, FileMode.Open, FileAccess.Read);
            fsSource.Read(tempBytes, 0, 4);

            //Check magic number 
            Array.Reverse(tempBytes);//System is Little Endian
            n = BitConverter.ToInt32(tempBytes, 0);
            if (n != 2049) this.label= - 1;

            //Skip data
            fsSource.Read(tempBytes, 0, 4);
            for (int i = 0; i < offset; i++) fsSource.Read(tempBytes, 0, 1);

            //Read Label data
            fsSource.Read(tempBytes, 0, 1);
            this.label = tempBytes[0];

            fsSource.Close();

            //read image data
            fsSource = new FileStream(image_path, FileMode.Open, FileAccess.Read);
            fsSource.Read(tempBytes, 0, 4);

            //Check magic number 
            Array.Reverse(tempBytes);//System is Little Endian
            n = BitConverter.ToInt32(tempBytes, 0);
            if (n != 2051) this.label= - 1;

            //Skip data
            fsSource.Read(image_data, 0, 12);
            for (int i = 0; i < offset; i++) fsSource.Read(image_data, 0, 784);

            //Read image data
            fsSource.Read(image_data, 0, 784);
            fsSource.Close();
        }
    }
}
