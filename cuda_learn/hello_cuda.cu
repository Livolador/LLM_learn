#include <stdio.h>

__global__ void hello_cuda()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    hello_cuda<<<4, 4>>>();
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
        return 1;
    }
    else {
        printf("No CUDA error\n");
    }
    cudaDeviceSynchronize();
}
