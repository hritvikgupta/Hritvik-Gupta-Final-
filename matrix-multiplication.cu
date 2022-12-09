#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#define check_Error() {                                                              \
    cudaError_t e=cudaGetLastError();                                                   \
    if(e!=cudaSuccess) {                                                                \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));   \
        cudaDeviceReset();                                                              \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}
#define Check_Allocation_Return_Value(a){   \
    if(a==NULL) {                           \
    printf("Allocation Error\n");           \
    cudaDeviceReset();                      \
    exit(EXIT_FAILURE);                     \
    }                                       \
}



__global__ void firstkernel(double *A,double *B,double *C,int width, double r){
    int cols = blockIdx.y*blockDim.y+threadIdx.y;
    int rows = blockIdx.x*blockDim.x+threadIdx.x;
    int step;
    double prod_val = 0;
    
    if((cols < (int)(width*r)) &&(rows < (int)(width*r))){
    
      for(step=0;step<width;step++){
          prod_val += A[cols*width+step] * B[step*(int)(width*r)+rows];
      }
      
      C[cols*(int)(width*r)+rows] = prod_val;
    }
}

__global__ void secondKernel(double *A,double *B,double *C,int width, double r){
    int cols = blockIdx.y*blockDim.y+threadIdx.y;
    int rows = blockIdx.x*blockDim.x+threadIdx.x;
    int step;
    double prod_val = 0;
    
    if((cols < (int)(width*r)) && (rows < (int)(width*(1-r)))){
      
      for(step=0;step<width;step++){
          prod_val += A[cols*width+step] * B[step*(int)(width*(1-r))+rows];
      }
      
      C[cols*(int)(width*(1-r))+rows] = prod_val;
    }
}

__global__ void thirdKernel(double *A,double *B,double *C,int width, double r){
    int cols = blockIdx.y*blockDim.y+threadIdx.y;
    int rows = blockIdx.x*blockDim.x+threadIdx.x;
    
    int step;
    double prod_val = 0;
    if((cols <(int)(width*(1-r))) && (rows <(int)(width*r))){
      for(step=0;step<width;step++){
          prod_val += A[cols*width+step] * B[step*(int)(width*r)+rows];
      }
      
      
      C[cols*(int)(width*r)+rows] = prod_val;
    }
}

__global__ void fourthKernel(double *A,double *B,double *C,int width, double r){
    int cols = blockIdx.y*blockDim.y+threadIdx.y;
    int rows = blockIdx.x*blockDim.x+threadIdx.x;
    
    int step;
    double prod_val = 0;
    if((cols < (int)(width*(1-r))) && (rows < (int)(width*(1-r)))){
      
      for(step=0;step<width;step++){
          prod_val += A[cols*width+step] * B[step*(int)(width*(1-r))+rows];
      }
      C[cols*(int)(width*(1-r))+rows] = prod_val;
    }
}



int main(int argc,char *argv[]){
    const int numStreams = 4;
    cudaStream_t streams[numStreams];
    int N = 2048;
    double *hA,*hB,*hC;
    int id,j,i;
    int Envs;
    double r = 0.5;
    double reverse = (1-r);
    double *hA1,*hA2,*hB1,*hB2,*hC1,*hC2,*hC3,*hC4;
    double *dA1,*dA1_2,*dA2,*dA2_2,*dB1,*dB1_2,*dB2,*dB2_2;
    double *dC1,*dC2,*dC3,*dC4;
         
    cudaGetDeviceCount(&Envs);     
    cudaMallocHost(&hA,N*N*sizeof(double));
    Check_Allocation_Return_Value(hA)
    cudaMallocHost(&hB,N*N*sizeof(double));
    Check_Allocation_Return_Value(hB)
    cudaMallocHost(&hC,N*N*sizeof(double));
    Check_Allocation_Return_Value(hC)
    memset(hC,0,N*N*sizeof(double)); 
    srand (time(NULL));
    
    for(i=0;i<N*N;i++){
        hA[i] = rand()%10;
        hB[i] = rand()%10;
    }
    

    int widthGrid= 1+N/32;
    dim3 dimGrid(widthGrid,widthGrid,1);
    dim3 dimBlock(32,32,1);
   
    id=0;
    cudaSetDevice((int)(id%Envs));
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    
    cudaMallocHost(&hA1,(int)(N*N*r*sizeof(double)));
    Check_Allocation_Return_Value(hA1)
    cudaMallocHost(&hB1,(int)(N*N*r*sizeof(double)));
    Check_Allocation_Return_Value(hB1)
    cudaMallocHost(&hC1,(int)(N*N*r*r*sizeof(double)));
    Check_Allocation_Return_Value(hC1)
    
    for(int i=0;i<(int)(N*r);i++){
        for(int j=0;j<N;j++){
            hA1[i*N+j] =  hA[i*N+j];
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<(N*r);j++){
            hB1[i*(int)(N*r)+j] =  hB[i*N+j];
        }
    }

    cudaMalloc((void**)&dA1,(int)(N*N*r*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dB1,(int)(N*N*r*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dC1,(int)(N*N*r*r*sizeof(double)));
    check_Error()
    
    id=1;
    cudaSetDevice((int)(id%Envs));
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    cudaMallocHost(&hB2,(int)(N*N*reverse*sizeof(double)));
    Check_Allocation_Return_Value(hB2)
    cudaMallocHost(&hC2,(int)(N*N*r*reverse*sizeof(double)));
    Check_Allocation_Return_Value(hC2)
    
    for(int i=0;i<N;i++){
        for(int j=0;j<(N*reverse);j++){
            hB2[i*(int)(N*reverse)+j] =  hB[i*N+(int)(N*r)+j];
        }
    }
     
    cudaMalloc((void**)&dA1_2,(int)(N*N*r*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dB2,(int)(N*N*reverse*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dC2,(int)(N*N*r*reverse*sizeof(double)));
    check_Error()
        
    id=2;
    cudaSetDevice(id%Envs);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    cudaMallocHost(&hA2,(int)(N*N*reverse*sizeof(double)));
    Check_Allocation_Return_Value(hA2)
    cudaMallocHost(&hC3,(int)(N*N*reverse*r*sizeof(double)));
    Check_Allocation_Return_Value(hC3)
    
    for(int i=0;i<(int)(N*reverse);i++){
        for(int j=0;j<N;j++){
            hA2[i*N+j] =  hA[(i+(int)(N*r))*N+j];
        }
    }
    
    cudaMalloc((void**)&dA2,(int)(N*N*reverse*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dB1_2,(int)(N*N*r*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dC3,(int)(N*N*r*reverse*sizeof(double)));
    check_Error()  
        

    id=3;
    cudaSetDevice(id%Envs);
    cudaStreamCreateWithFlags(&streams[id],cudaStreamNonBlocking);
    cudaMallocHost(&hC4,(int)(N*N*reverse*reverse*sizeof(double)));
    Check_Allocation_Return_Value(hC4)
    cudaMalloc((void**)&dA2_2,(int)(N*N*reverse*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dB2_2,(int)(N*N*reverse*sizeof(double)));
    check_Error()
    cudaMalloc((void**)&dC4,(int)(N*N*reverse*reverse*sizeof(double)));
    check_Error()
    
    id=0;
    cudaSetDevice(id%Envs);
    cudaMemcpyAsync(dA1,hA1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    cudaMemcpyAsync(dB1,hB1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    firstkernel <<< dimGrid,dimBlock,0,streams[id]>>>(dA1,dB1,dC1,N,r);
    check_Error()

    
    id=1;
    cudaSetDevice(id%Envs);
    cudaMemcpyAsync(dA1_2,hA1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    cudaMemcpyAsync(dB2,hB2,(int)(N*N*reverse*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    secondKernel <<< dimGrid,dimBlock,0,streams[id]>>>(dA1_2,dB2,dC2,N,r);
    check_Error()
    
    
    id=2;
    cudaSetDevice(id%Envs);
    cudaMemcpyAsync(dA2,hA2,(int)(N*N*reverse*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    cudaMemcpyAsync(dB1_2,hB1,(int)(N*N*r*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    thirdKernel <<< dimGrid,dimBlock,0,streams[id]>>>(dA2,dB1_2,dC3,N,r);
    check_Error()
    
    id=3;
    cudaSetDevice(id%Envs);
    cudaMemcpyAsync(dA2_2,hA2,(int)(N*N*reverse*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    cudaMemcpyAsync(dB2_2,hB2,(int)(N*N*reverse*sizeof(double)),cudaMemcpyHostToDevice,streams[id]);
    check_Error()
    fourthKernel <<< dimGrid,dimBlock,0,streams[id]>>>(dA2_2,dB2_2,dC4,N,r);
    check_Error()
    cudaMemcpyAsync(hC1,dC1,(int)(N*N*r*r*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    check_Error()
    cudaMemcpyAsync(hC2,dC2,(int)(N*N*r*reverse*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    check_Error()
    cudaMemcpyAsync(hC3,dC3,(int)(N*N*r*reverse*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    check_Error()
    cudaMemcpyAsync(hC4,dC4,(int)(N*N*reverse*reverse*sizeof(double)),cudaMemcpyDeviceToHost,streams[id]);
    check_Error()

    for(i=0;i<4;i++)
    {
      id = i;
      cudaSetDevice(id%Envs);
      cudaStreamSynchronize(streams[id]);
    }

    for(i=0;i<(int)N*r;i++){
        for(j=0;j<(int)N*r;j++){
              hC[i*N+j] = hC1[i*(int)(N*r)+j];
        }
    }
    
    
    
    for(i=0;i<(int)N*r;i++){
        for(j=0;j<(int)(N*reverse);j++){
             hC[i*N+j+(int)(N*r)] = hC2[i*(int)(N*reverse)+j];
        }
    }
    
    for(i=0;i<(int)(N*reverse);i++){
        for(j=0;j<(int)(N*r);j++){
             hC[(i+(int)(N*r))*N+j] = hC3[i*(int)(N*r)+j];
        }
    }

    
    for(i=0;i<(int)(N*reverse);i++){
        for(j=0;j<(int)(N*reverse);j++){
            hC[(i+(int)(N*r))*N+j+(int)(N*r)] = hC4[i*(int)(N*reverse)+j];
        }
    }
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFreeHost(hA1);
    cudaFreeHost(hA2);
    cudaFreeHost(hB1);
    cudaFreeHost(hB2);
    cudaFreeHost(hC1);
    cudaFreeHost(hC2);
    cudaFreeHost(hC3);
    cudaFreeHost(hC4);
    
    id=0;
    cudaSetDevice(id%Envs);
    cudaFree(dA1);
    check_Error()
    cudaFree(dB1);
    check_Error()
    cudaFree(dC1);
    check_Error()
    
    id=1;
    cudaSetDevice(id%Envs);
    cudaFree(dA1_2);
    check_Error()
    cudaFree(dB2);
    check_Error()
    cudaFree(dC2);
    check_Error()
    
    id=2;
    cudaSetDevice(id%Envs);
    cudaFree(dA2);
    check_Error()
    cudaFree(dB1_2);
    check_Error()
    cudaFree(dC3);
    check_Error()
    
    id=3;
    cudaSetDevice(id%Envs);
    cudaFree(dA2_2);
    check_Error()
    cudaFree(dB2_2);
    check_Error()
    cudaFree(dC4);
    check_Error()
    
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);
    cudaStreamDestroy(streams[3]);
    
    return(0);
}