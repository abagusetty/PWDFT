#ifndef __PWDFT_GPUWRAPPER
#define __PWDFT_GPUWRAPPER

#include <stdexcept>
#include <string>

#ifdef NWPW_CUDA
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cusolverDn.h>

class cuda_exception : public std::exception {

  std::string file_;
  int line_;
  cudaError_t err_code_;

  const char *what() const noexcept override {
    std::stringstream ss;
    ss << "CUDA Exception, " << cudaGetErrorString(err_code_) << " at "
       << std::endl
       << file_ << ", L: " << line_ << std::endl;
    auto msg = ss.str();
    return strdup(msg.c_str());
  }

public:
  cuda_exception(std::string file, int line, cudaError_t err)
      : file_(file), line_(line), err_code_(err) {}
};

class cufft_exception : public std::exception {

  std::string file_;
  int line_;
  cufftResult err_code_;

  static const char *_cudaGetErrorEnum(cufftResult error) {
    switch (error) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    }
    return "<unknown>";
  }

  const char *what() const noexcept override {
    std::stringstream ss;
    ss << "CUFFT Exception, "
       << " Error Code: " << _cudaGetErrorEnum(err_code_) << std::endl
       << " at " << file_ << ", L: " << line_ << std::endl;
    auto msg = ss.str();
    return strdup(msg.c_str());
  }

public:
  cufft_exception(std::string file, int line, cufftResult err)
      : file_(file), line_(line), err_code_(err) {}
};

class cublas_exception : public std::exception {

  std::string file_;
  int line_;
  cublasStatus_t err_code_;

  const char *what() const noexcept override {
    std::stringstream ss;
    ss << "CUBLAS Exception, "
       << " Error Code: " << cublasGetStatusString(err_code_) << std::endl
       << " at " << file_ << " : " << line_ << std::endl;
    auto msg = ss.str();
    return strdup(msg.c_str());
  }

public:
  cublas_exception(std::string file, int line, cublasStatus_t err)
      : file_(file), line_(line), err_code_(err) {}
};

#define NWPW_GPU_ERROR(CALL)                                                   \
  do {                                                                         \
    cudaError_t err = CALL;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw cuda_exception(__FILE__, __LINE__, err);                           \
    }                                                                          \
  } while (0)

#define NWPW_GPUFFT_ERROR(CALL)                                                \
  do {                                                                         \
    cufftResult err = CALL;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
      throw cufft_exception(__FILE__, __LINE__, err);                          \
    }                                                                          \
  } while (0)

#define NWPW_GPUBLAS_ERROR(CALL)                                               \
  do {                                                                         \
    cublasStatus_t err = CALL;                                                 \
    if (err != rocblas_status_success) {                                       \
      throw cublas_exception(__FILE__, __LINE__, err);                         \
    }                                                                          \
  } while (0)

#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuHostMalloc cudaHostAlloc
#define gpuHostMallocDefault cudaHostAllocDefault
#define gpuMallocManaged cudaMallocManaged
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpy2DAsync cudaMemcpy2DAsync
#define gpuMemcpy2D cudaMemcpy2D
#define gpuFreeHost cudaFreeHost
#define gpuFree cudaFree
#define gpuMemPrefetchAsync cudaMemPrefetchAsync
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroyWithFlags cudaStreamDestroyWithFlags
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpublasStatus_t cublasStatus_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuMemGetInfo cudaMemGetInfo
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuDeviceReset cudaDeviceReset
#define gpuMallocHost cudaMallocHost
#define gpuEvent_t cudaEvent_t
#define gpuMemset cudaMemset
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define gpublasCreate cublasCreate
#define gpublasDestroy cublasDestroy
#define gpublasHandle_t cublasHandle_t
#define gpublasSetStream cublasSetStream
#define gpublasDgemm cublasDgemm
#define gpublasZgemm cublasZgemm
#define gpublasSideMode_t cublasSideMode_t
#define gpublasFillMode_t cublasFillMode_t
#define gpublasDiagType_t cublasDiagType_t
#define gpublasOperation_t cublasOperation_t
#define gpublasSetMatrixAsync cublasSetMatrixAsync
#define gpublasGetMatrixAsync cublasGetMatrixAsync
#define GPUBLAS_OP_C CUBLAS_OP_C
#define GPUBLAS_OP_T CUBLAS_OP_T
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_DIAG_UNIT CUBLAS_DIAG_UNIT
#define GPUBLAS_DIAG_NON_UNIT CUBLAS_DIAG_NON_UNIT
#define GPUBLAS_SIDE_LEFT CUBLAS_SIDE_LEFT
#define GPUBLAS_SIDE_RIGHT CUBLAS_SIDE_RIGHT
#define GPUBLAS_FILL_MODE_LOWER CUBLAS_FILL_MODE_LOWER
#define GPUBLAS_FILL_MODE_UPPER CUBLAS_FILL_MODE_UPPER
#define gpusolverStatus_t cusolverStatus_t
#define GPUSOLVER_STATUS_SUCCESS CUSOLVER_STATUS_SUCCESS
#define gpusolverDnHandle_t cusolverDnHandle_t
#define gpusolverDnCreate cusolverDnCreate
#define gpusolverDnDestroy cusolverDnDestroy
#define gpusolverDnSetStream cusolverDnSetStream
#define gpublasDscal cublasDscal
#define gpublasZscal cublasZscal
#define gpublasDaxpy cublasDaxpy
#define gpublasZaxpy cublasZaxpy
#define gpuDoubleComplex cuDoubleComplex

#elif defined(NWPW_HIP)

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <rocfft/rocfft.h>
#include <rocsolver/rocsolver.h>

class hip_exception : public std::exception {

  std::string file_;
  int line_;
  hipError_t err_code_;

  const char *what() const noexcept override {
    std::stringstream ss;
    ss << "HIP Exception, " << hipGetErrorString(err_code_) << " at "
       << std::endl
       << file_ << ", L: " << line_ << std::endl;
    auto msg = ss.str();
    return strdup(msg.c_str());
  }

public:
  hip_exception(std::string file, int line, hipError_t err)
      : file_(file), line_(line), err_code_(err) {}
};

class rocfft_exception : public std::exception {

  std::string file_;
  int line_;
  rocfft_status err_code_;

  static const char *_rocfftGetErrorEnum(rocfft_status error) {
    switch (error) {
    case rocfft_status_success:
      return "ROCFFT_STATUS_SUCCESS";

    case rocfft_status_failure:
      return "ROCFFT_STATUS_FAILURE";

    case rocfft_status_invalid_arg_value:
      return "ROCFFT_STATUS_INVALID_ARG_VALUE";

    case rocfft_status_invalid_dimensions:
      return "ROCFFT_STATUS_INVALID_DIMENSIONS";

    case rocfft_status_invalid_array_type:
      return "ROCFFT_STATUS_INVALID_ARRAY_TYPE";

    case rocfft_status_invalid_strides:
      return "ROCFFT_STATUS_INVALID_STRIDES";

    case rocfft_status_invalid_distance:
      return "ROCFFT_STATUS_INVALID_DISTANCE";

    case rocfft_status_invalid_offset:
      return "ROCFFT_STATUS_INVALID_OFFSET";

    case rocfft_status_invalid_work_buffer:
      return "ROCFFT_STATUS_INVALID_WORK_BUFFER";
    }
    return "<unknown>";
  }

  const char *what() const noexcept override {
    std::stringstream ss;
    ss << "ROCFFT Exception, "
       << " Error Code: " << _rocfftGetErrorEnum(err_code_) << std::endl
       << " at " << file_ << " : " << line_ << std::endl;
    auto msg = ss.str();
    return strdup(msg.c_str());
  }

public:
  rocfft_exception(std::string file, int line, rocfft_status err)
      : file_(file), line_(line), err_code_(err) {}
};

class rocblas_exception : public std::exception {

  std::string file_;
  int line_;
  rocblas_status err_code_;

  const char *what() const noexcept override {
    std::stringstream ss;
    ss << "rocBLAS Exception, "
       << " Error Code: " << rocblas_status_to_string(err_code_) << std::endl
       << " at " << file_ << " : " << line_ << std::endl;
    auto msg = ss.str();
    return strdup(msg.c_str());
  }

public:
  rocblas_exception(std::string file, int line, rocblas_status err)
      : file_(file), line_(line), err_code_(err) {}
};

#define NWPW_GPU_ERROR(CALL)                                                   \
  do {                                                                         \
    hipError_t err = CALL;                                                     \
    if (err != hipSuccess) {                                                   \
      throw hip_exception(__FILE__, __LINE__, err);                            \
    }                                                                          \
  } while (0)

#define NWPW_GPUFFT_ERROR(CALL)                                                \
  do {                                                                         \
    rocfft_status err = CALL;                                                  \
    if (err != rocfft_status_success) {                                        \
      throw rocfft_exception(__FILE__, __LINE__, err);                         \
    }                                                                          \
  } while (0)

#define NWPW_GPUBLAS_ERROR(CALL)                                               \
  do {                                                                         \
    rocblas_status err = CALL;                                                 \
    if (err != rocblas_status_success) {                                       \
      throw rocblas_exception(__FILE__, __LINE__, err);                        \
    }                                                                          \
  } while (0)

#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuHostMalloc hipHostMalloc
#define gpuHostMallocDefault hipHostMallocDefault
#define gpuMallocManaged hipMallocManaged
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpy2DAsync hipMemcpy2DAsync
#define gpuMemcpy2D hipMemcpy2D
#define gpuFreeHost hipHostFree
#define gpuFree hipFree
#define gpuMemPrefetchAsync hipMemPrefetchAsync // not sure about this
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroyWithFlags hipStreamDestroyWithFlags
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpublasStatus_t rocblas_status
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuMemGetInfo hipMemGetInfo
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuDeviceReset hipDeviceReset
#define gpuMallocHost hipHostMalloc
#define gpuEvent_t hipEvent_t
#define gpuMemset hipMemset
#define GPUBLAS_STATUS_SUCCESS rocblas_status_success
#define gpublasCreate rocblas_create_handle
#define gpublasDestroy rocblas_destory_handle
#define gpublasHandle_t rocblas_handle
#define gpublasSetStream rocblas_set_stream
#define gpublasSideMode_t rocblas_side
#define gpublasFillMode_t rocblas_fill
#define gpublasDiagType_t rocblas_diagonal
#define gpublasOperation_t rocblas_operation
#define gpublasSetMatrixAsync rocblas_set_matrix_async
#define gpublasGetMatrixAsync rocblas_get_matrix_async
#define gpublasDgemm rocblas_dgemm
#define gpublasZgemm rocblas_zgemm
#define GPUBLAS_OP_C rocblas_operation_conjugate_transpose
#define GPUBLAS_OP_T rocblas_operation_transpose
#define GPUBLAS_OP_N rocblas_operation_none
#define GPUBLAS_DIAG_UNIT rocblas_diagonal_unit
#define GPUBLAS_DIAG_NON_UNIT rocblas_diagonal_non_unit
#define GPUBLAS_SIDE_LEFT rocblas_side_left
#define GPUBLAS_SIDE_RIGHT rocblas_side_right
#define GPUBLAS_FILL_MODE_LOWER rocblas_fill_lower
#define GPUBLAS_FILL_MODE_UPPER rocblas_fill_upper
#define gpusolverStatus_t rocsolver_status
#define GPUSOLVER_STATUS_SUCCESS rocblas_status_success
#define gpusolverDnHandle_t rocsolver_handle
#define gpusolverDnCreate rocsolver_create_handle
#define gpusolverDnDestroy rocsolver_destroy_handle
#define gpusolverDnSetStream rocsolver_set_stream
#define gpublasDscal rocblasDscal
#define gpublasZscal rocblasZscal
#define gpublasDaxpy rocblasDaxpy
#define gpublasZaxpy rocblasZaxpy
#define gpuDoubleComplex rocblas_double_complex

#elif defined(NWPW_SYCL)

#include "sycl_device.hpp"
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/exceptions.hpp>
#include <oneapi/mkl/lapack.hpp>

#define __host__
#define __global__ __inline__ __attribute__((always_inline))
#define __device__ __attribute__((always_inline))
static int gpuMemcpyHostToDevice{0};
static int gpuMemcpyDeviceToHost{0};
static int gpuMemcpyDeviceToDevice{0};
static int gpuHostMallocDefault{0};
// using gpublasHandle_t = int;
using gpusolverDnHandle_t = int;
using gpuDoubleComplex = std::complex<double>;

static inline double atomicAdd(double *addr, const double val) {
  return sycl::atomic_ref<double, sycl::memory_order::relaxed,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>(*addr)
      .fetch_add(val);
}
//#define atomicAdd(addr,val) sycl::atomic_ref<double,
//sycl::memory_order::relaxed, sycl::memory_scope::device,
//sycl::access::address_space::global_space>(*addr).fetch_add( val )
#define atomicSub(addr, val)                                                   \
  sycl::atomic_ref<int, sycl::memory_order::relaxed,                           \
                   sycl::memory_scope::device,                                 \
                   sycl::access::address_space::global_space>(*addr)           \
      .fetch_sub(val)

// the date 20240207 is the build date of oneapi/eng-compiler/2024.04.15.002
#if (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION > 20240227)
#define __syncthreads()                                                        \
  (sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_work_group<3>()))
#define __syncwarp()                                                           \
  (sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group()))
#else
#define __syncthreads()                                                        \
  (sycl::group_barrier(sycl::ext::oneapi::experimental::this_group<3>()))
#define __syncwarp()                                                           \
  (sycl::group_barrier(sycl::ext::oneapi::experimental::this_sub_group()))
#endif // SYCL_COMPILER_VERSION

static inline void gpuGetDeviceCount(int *count) { syclGetDeviceCount(count); }
static inline void gpuSetDevice(int deviceID) { syclSetDevice(deviceID); }
static inline void gpuGetDevice(int *deviceID) { syclGetDevice(deviceID); }
#define gpuDeviceReset()                                                       \
  {}

static inline void gpuMemcpy(void *dst, const void *src, size_t count,
                             int kind) {
  sycl_get_queue()->memcpy(dst, src, count).wait();
}
static inline void gpuDeviceSynchronize() { sycl_get_queue()->wait(); }
static inline void gpuMemcpyAsync(void *dst, const void *src, size_t count,
                                  int kind, sycl::queue *stream) {
  stream->memcpy(dst, src, count);
}

static inline void gpuMemset(void *ptr, int val, size_t size) {
  sycl_get_queue()->memset(ptr, val, size).wait();
}
static inline void gpuMemGetInfo(size_t *free, size_t *total) {
  *free = sycl_get_queue()
              ->get_device()
              .get_info<sycl::ext::intel::info::device::free_memory>();
  *total = sycl_get_queue()
               ->get_device()
               .get_info<sycl::info::device::global_mem_size>();
}

static inline void gpuMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                                    size_t spitch, size_t width, size_t height,
                                    int kind, sycl::queue *stream) {
  stream->ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height);
}
static inline void gpuMemcpy2D(void *dst, size_t dpitch, const void *src,
                               size_t spitch, size_t width, size_t height,
                               int kind) {
  sycl_get_queue()
      ->ext_oneapi_memcpy2d(dst, dpitch, src, spitch, width, height)
      .wait();
}

static inline void gpuMalloc(void **ptr, size_t size) {
  (*ptr) = (void *)sycl::malloc_device(size, *sycl_get_queue());
}
static inline void gpuHostMalloc(void **ptr, size_t size, int type) {
  (*ptr) = (void *)sycl::malloc_host(size, *sycl_get_queue());
}
static inline void gpuMallocHost(void **ptr, size_t size) {
  (*ptr) = (void *)sycl::malloc_host(size, *sycl_get_queue());
}
static inline void gpuFree(void *ptr) {
  sycl::free(ptr, sycl_get_queue()->get_context());
}
static inline void gpuStreamCreate(sycl::queue **syclStream) {
  (*syclStream) = new sycl::queue(
      sycl_get_queue()->get_context(), sycl_get_queue()->get_device(),
      asyncHandler,
      sycl::property_list{sycl::property::queue::enable_profiling{},
                          sycl::property::queue::in_order{}});
}
static inline void gpuStreamDestroy(sycl::queue *stream) {
  stream->wait();
  delete stream;
}
static inline void gpuStreamSynchronize(sycl::queue *stream) { stream->wait(); }
static inline void gpuFreeHost(void *ptr) {
  sycl::free(ptr, sycl_get_queue()->get_context());
  //::operator delete(ptr);
  // std::free(ptr);
}
static inline void gpuEventCreate(sycl::event **syclevent) {
  *syclevent = new sycl::event{};
}
static inline void gpuEventDestroy(sycl::event **event) { delete event; }
static inline void gpuEventRecord(sycl::event *&event, sycl::queue *stream) {
  *event = stream->ext_oneapi_submit_barrier();
}

static inline void gpuEventElapsedTime(float *ms, sycl::event *startEvent,
                                       sycl::event *endEvent) {
  *ms =
      (endEvent
           ->get_profiling_info<sycl::info::event_profiling::command_end>() -
       startEvent
           ->get_profiling_info<sycl::info::event_profiling::command_start>()) /
      1000000.0f;
}

// oneMKL functionality

using gpuStream_t = sycl::queue *;
using gpublasHandle_t = sycl::queue *;
using gpuEvent_t = sycl::event *;
using gpublasOperation_t = oneapi::mkl::tranpose;

#define GPUBLAS_OP_T oneapi::mkl::transpose::C
#define GPUBLAS_OP_N oneapi::mkl::transpose::T
#define GPUBLAS_OP_N oneapi::mkl::transpose::N
#define GPUBLAS_DIAG_UNIT oneapi::mkl::diag::U
#define GPUBLAS_DIAG_NON_UNIT oneapi::mkl::diag::N
#define GPUBLAS_SIDE_LEFT oneapi::mkl::side::left
#define GPUBLAS_SIDE_RIGHT oneapi::mkl::side::right
#define GPUBLAS_FILL_MODE_LOWER oneapi::mkl::uplo::L
#define GPUBLAS_FILL_MODE_UPPER oneapi::mkl::uplo::U

#define checkGPUErrors(fn) (fn);
#define checkGPU(fn) (fn);
static inline void gpuGetLastError() {}

static inline void gpublasDgemm(gpublasHandle_t handle,
                                gpublasOperation_t transa,
                                gpublasOperation_t transb, int m, int n, int k,
                                const double *alpha, const double *A, int lda,
                                const double *B, int ldb, const double *beta,
                                double *C, int ldc) {
  const double alpha_val = *alpha;
  const double beta_val = *beta;
  oneapi::mkl::blas::column_major::gemm(handle, transa, transb, m, n, k, alpha_val,
					A, lda, B, ldb, beta_val, C, ldc);
}
static inline void gpublasZgemm(gpublasHandle_t handle,
                                gpublasOperation_t transa,
                                gpublasOperation_t transb, int m, int n, int k,
				const gpuDoubleComplex *alpha,
				const gpuDoubleComplex *A, int lda,
				const gpuDoubleComplex *B, int ldb,
				const gpuDoubleComplex *beta,
				gpuDoubleComplex *C, int ldc) {
  const gpuDoubleComplex alpha_val = *alpha;
  const gpuDoubleComplex beta_val = *beta;
  oneapi::mkl::blas::column_major::gemm(handle, transa, transb, m, n, k, alpha_val,
					A, lda, B, ldb, beta_val, C, ldc);
}

namespace detail {
  static inline void gpublasMatrixAsync(int rows, int cols, size_t elem_size,
					const void *from_ptr, int from_ld, void *to_ptr,
					int to_ld, sycl::queue *que) {
    if (to_ptr == from_ptr && to_ld == from_ld) {
      return;
    }
 
    if (to_ld == from_ld) {
      size_t copy_size = elem_size * ((cols - 1) * (size_t)to_ld + rows);
      que->memcpy(to_ptr, from_ptr, copy_size);
    } else {
      gpuMemcpy2DAsync(to_ptr, elem_size * to_ld, from_ptr, elem_size * from_ld,
		       elem_size * rows, cols, 0, que);
    }
  }  
}
static inline void gpublasSetMatrixAsync(int rows, int cols, size_t elem_size,
					 const void *from_ptr, int from_ld, void *to_ptr,
					 int to_ld, sycl::queue *que) {
  detail::gpublasMatrixAsync(rows, cols, elem_size, from_ptr, from_ld, to_ptr, to_ld, que);
}

static inline void gpublasGetMatrixAsync(int rows, int cols, size_t elem_size,
					 const void *from_ptr, int from_ld, void *to_ptr,
					 int to_ld, sycl::queue *que) {
  detail::gpublasMatrixAsync(rows, cols, elem_size, from_ptr, from_ld, to_ptr, to_ld, que);}

#define NWPW_GPUBLAS_ERROR(CALL)                                               \
  do {                                                                         \
    try {                                                                      \
      CALL;                                                                    \
    } catch (oneapi::mkl::exception const &ex) {                               \
      std::stringstream msg;                                                   \
      msg << "Fatal oneMKL::BLAS error: " << __FILE__ << " : " << __LINE__     \
          << std::endl;                                                        \
      throw(std::runtime_error(ex.what()));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define NWPW_GPUSOLVER_ERROR(CALL)                                             \
  do {                                                                         \
    try {                                                                      \
      CALL;                                                                    \
    } catch (oneapi::mkl::exception const &ex) {                               \
      std::stringstream msg;                                                   \
      msg << "Fatal oneMKL::LAPACK error: " << __FILE__ << " : " << __LINE__   \
          << std::endl;                                                        \
      throw(std::runtime_error(ex.what()));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#define NWPW_GPUFFT_ERROR(CALL)                                                \
  do {                                                                         \
    try {                                                                      \
      CALL;                                                                    \
    } catch (oneapi::mkl::exception const &ex) {                               \
      std::stringstream msg;                                                   \
      msg << "Fatal oneMKL::FFT error: " << __FILE__ << " : " << __LINE__      \
          << std::endl;                                                        \
      throw(std::runtime_error(ex.what()));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

#endif // NWPW_CUDA

#endif /* __PWDFT_GPUWRAPPER */
