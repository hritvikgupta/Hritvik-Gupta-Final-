# Hritvik-Gupta-Final-GPU Project
matrix-matrix multiplication Using CUDA Multi-GPU's and Streams

The program contain 2 matrix of each 2048 with random values. 
There are total 4 GPUs on which matrix multiplication is run wtih 4 different kernels. You can you check the number of GPUs on your system and accordingle can set 
kernels to it.
Make File is not create for this program but rather a simple make file for matrix multiplication
1. Clone the Repository to your System ("git clone")
2. In order to compile the program execute the following command on the terminal -> ("nvcc matrix-multiplication.cu")
3. The above command will generate a file with the name "a.out"
4. Finally to run the code pass the command ("./a.out") in the Terminal

Compile:    nvcc matrix-multiplication.cu

Run:        ./a.out
