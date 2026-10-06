#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void reduce_sum_kernel(const float* input, float* output, int N) {
    extern __shared__ float arr[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        arr[tid] = input[i];
    } else {
        arr[tid] = 0;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            arr[tid] += arr[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = arr[0];
    }
}

__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

static void check(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "tensor must be on the GPU");
    TORCH_CHECK(t.dtype() == torch::kFloat32, "tensor must be float32");
    TORCH_CHECK(t.is_contiguous(), "tensor must be contiguous");
}

torch::Tensor vector_add(torch::Tensor A, torch::Tensor B) {
    check(A); check(B);
    int N = A.numel();
    auto C = torch::empty_like(A);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

// Returns one partial sum per block (same design as your reduction program)
torch::Tensor reduce_sum_partials(torch::Tensor x) {
    check(x);
    int N = x.numel();
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    auto out = torch::empty({blocks}, x.options());
    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

torch::Tensor naive_matmul(torch::Tensor A, torch::Tensor B) {
    check(A); check(B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    naive_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector add (CUDA)");
    m.def("reduce_sum_partials", &reduce_sum_partials, "Per-block partial sums (CUDA)");
    m.def("naive_matmul", &naive_matmul, "Naive matmul (CUDA)");
}
