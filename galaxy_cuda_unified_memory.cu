// on dione, first load the cuda module
//    module load cuda
//
// compile your program with
//    nvcc -O3 -arch=sm_70 --ptxas-options=-v -o galaxy galaxy_cuda.cu -lm
//
// run your program with 
//    srun -p gpu -c 1 ./galaxy_cuda RealGalaxies_100k_arcmin.dat SyntheticGalaxies_100k_arcmin.dat omega.out

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math_constants.h>
#include <sys/time.h>

static const long INPUT_SIZE = 100000L;
static const long long RESULT_SIZE = (long long)INPUT_SIZE * (long long)INPUT_SIZE;

static const int TOTAL_DEGREES = 90;
static const int BINS_PER_DEGREE = 4;
static const int BIN_COUNT = TOTAL_DEGREES * BINS_PER_DEGREE;

static const int THREADS_PER_BLOCK = 128;
static const int BLOCK_COUNT = INPUT_SIZE / THREADS_PER_BLOCK + 1;

long int CPUMemory = 0L;
long int GPUMemory = 0L;
long int UnifiedMemory = 0L;


__global__
void ArcToVector(float *decl, float *rasc, float *x, float *y, float *z, int N)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;

   if (i < N) {
      float t = CUDART_PI_F / 2 - decl[i];
      float p = rasc[i];
      x[i] = cos(p) * sin(t);
      y[i] = sin(p) * sin(t);
      z[i] = cos(t);
 }
}

__device__ int calculate_histogram_index(float angle) {
   return (int)(4.0f * angle);
}

__device__ float radian_to_degrees(float rad) {
   return acosf(rad) / CUDART_PI_F * 180.0f;
}

__device__ float ensure_max_1(float f) {
   return (f > 1.0f) ? 1.0f : f;
}

__global__
void CountHist(float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, unsigned int *result_hist, int N)
{
   __shared__ unsigned int thread_blck_hist[BIN_COUNT];

   int x1_idx = blockDim.x * blockIdx.x + threadIdx.x;
   int angles_per_thread = blockDim.x;
      
   for (int i = threadIdx.x; i < BIN_COUNT; i += angles_per_thread)
      thread_blck_hist[i] = 0U;
   __syncthreads();

   if (x1_idx < N) {
      
      int x2_start = angles_per_thread * blockIdx.y;
      int x2_end = x2_start + angles_per_thread;
      x2_end = min(N, x2_end);

      float x1_val = x1[x1_idx];
      float y1_val = y1[x1_idx];
      float z1_val = z1[x1_idx];

      for (int x2_idx = x2_start; x2_idx < x2_end; x2_idx++) {
         float x2_val = x2[x2_idx];
         float y2_val = y2[x2_idx];
         float z2_val = z2[x2_idx];

         float radians = ensure_max_1(x1_val * x2_val + y1_val * y2_val + z1_val * z2_val);
         float angle = radian_to_degrees(radians);
         int hist_idx = calculate_histogram_index(angle);

         atomicAdd(&thread_blck_hist[hist_idx], 1U);
      }
   }
   __syncthreads();

   for (int i = threadIdx.x; i < BIN_COUNT; i += angles_per_thread)
      atomicAdd(&result_hist[i], thread_blck_hist[i]);
}


int main(int argc, char *argv[])
{
   int parseargs_readinput(int argc, char *argv[], float *real_rasc, float *real_decl, float *rand_rasc, float *rand_decl);
   int getDevice();

   int check_correct_hist_sum(unsigned int to_sum[], int arr_length, long long expected_sum, const char* hist_name);

   double walltime;
   struct timeval _ttime;
   struct timezone _tzone;

   FILE *outfil;

   const int arr_size = INPUT_SIZE * sizeof(float);
   const int hist_arr_size = TOTAL_DEGREES * BINS_PER_DEGREE * sizeof(unsigned int);

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   gettimeofday(&_ttime, &_tzone);
   walltime = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

   float *real_rasc; cudaMallocManaged(&real_rasc, arr_size);
   float *real_decl; cudaMallocManaged(&real_decl, arr_size);
   float *rand_rasc; cudaMallocManaged(&rand_rasc, arr_size);
   float *rand_decl; cudaMallocManaged(&rand_decl, arr_size);
   UnifiedMemory += 4L * arr_size;

   if ( parseargs_readinput(argc, argv, real_rasc, real_decl, rand_rasc, rand_decl) != 0 ) return(-1);

   // For your entertainment: some performance parameters of the GPU you are running your programs on!
   // if ( getDevice() != 0 ) return(-1);
   
   float *d_real_x; cudaMalloc(&d_real_x, arr_size);
   float *d_real_y; cudaMalloc(&d_real_y, arr_size);
   float *d_real_z; cudaMalloc(&d_real_z, arr_size);
   float *d_rand_x; cudaMalloc(&d_rand_x, arr_size);
   float *d_rand_y; cudaMalloc(&d_rand_y, arr_size);
   float *d_rand_z; cudaMalloc(&d_rand_z, arr_size);
   GPUMemory += 6L * arr_size;
   
   unsigned int *hist_DD; cudaMallocManaged(&hist_DD, hist_arr_size);
   unsigned int *hist_RR; cudaMallocManaged(&hist_RR, hist_arr_size);
   unsigned int *hist_DR; cudaMallocManaged(&hist_DR, hist_arr_size);
   UnifiedMemory += 3L * hist_arr_size;

   ArcToVector<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>(real_decl, real_rasc, d_real_x, d_real_y, d_real_z, INPUT_SIZE);  // Real
   ArcToVector<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>(rand_decl, rand_rasc, d_rand_x, d_rand_y, d_rand_z, INPUT_SIZE);  // Rand

   dim3 dimGrid(BLOCK_COUNT, BLOCK_COUNT);
   CountHist<<<dimGrid, THREADS_PER_BLOCK>>>(d_real_x, d_real_y, d_real_z, d_real_x, d_real_y, d_real_z, hist_DD, INPUT_SIZE); // DD
   CountHist<<<dimGrid, THREADS_PER_BLOCK>>>(d_rand_x, d_rand_y, d_rand_z, d_rand_x, d_rand_y, d_rand_z, hist_RR, INPUT_SIZE); // RR
   CountHist<<<dimGrid, THREADS_PER_BLOCK>>>(d_real_x, d_real_y, d_real_z, d_rand_x, d_rand_y, d_rand_z, hist_DR, INPUT_SIZE); // DR
   
   cudaDeviceSynchronize();

   int correct_sum = 1;
   if (check_correct_hist_sum(hist_DD, BIN_COUNT, RESULT_SIZE, "DD") == 0) correct_sum = 0;
   if (check_correct_hist_sum(hist_RR, BIN_COUNT, RESULT_SIZE, "RR") == 0) correct_sum = 0;
   if (check_correct_hist_sum(hist_DR, BIN_COUNT, RESULT_SIZE, "DR") == 0) correct_sum = 0;
   if (correct_sum != 1) {
      printf("   Histogram sum shoule be %lld, exiting..\n", RESULT_SIZE);
      return (0);
   }

   printf("Omega values for the histograms:\n");
   outfil = fopen(argv[3],"w");
   if ( outfil == NULL ) {printf("Cannot open output file %s\n",argv[3]);return(-1);}
   fprintf(outfil,"bin start\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
   for ( int i = 0; i < BIN_COUNT; ++i ) {
      if ( hist_RR[i] > 0 ) {
          double omega =  (hist_DD[i]-2*hist_DR[i]+hist_RR[i])/((double)(hist_RR[i]));

          fprintf(outfil,"%6.3f\t%15lf\t%15u\t%15u\t%15u\n",((float)i)/BINS_PER_DEGREE, omega,
             hist_DD[i], hist_DR[i], hist_RR[i]);
          if (i < 10) printf("   angle %.2f deg. -> %.2f deg. : %.3f\n", i * 0.25, (i + 1) * 0.25, omega);
      } else if ( i < 5 ) printf("         ");
   }

   printf("\n");

   fclose(outfil);

   printf("   Results written to file %s\n",argv[3]);
   printf("   CPU memory allocated  = %.2lf MB\n",CPUMemory/1000000.0);
   printf("   GPU memory allocated  = %.2lf MB\n",GPUMemory/1000000.0);
   printf("   Unified memory allocated  = %.2lf MB\n",UnifiedMemory/1000000.0);

   gettimeofday(&_ttime, &_tzone);
   walltime = (double)(_ttime.tv_sec) + (double)(_ttime.tv_usec/1000000.0) - walltime;

   printf("   Total wall clock time = %.2lf s\n", walltime);

   cudaFree(real_rasc); cudaFree(real_decl); cudaFree(rand_rasc); cudaFree(rand_decl);
   cudaFree(d_real_x); cudaFree(d_real_y); cudaFree(d_real_z); cudaFree(d_rand_x); cudaFree(d_rand_y); cudaFree(d_rand_z);
   cudaFree(hist_DD); cudaFree(hist_RR); cudaFree(hist_DR);

   return(0);
}


int check_correct_hist_sum(unsigned int to_sum[], int arr_length, long long expected_sum, const char* hist_name)
{
   long long sum = 0LL;
   for (int i = 0; i < arr_length; ++i)
      sum += to_sum[i];

   printf("Histogram %s : sum = %lld\n", hist_name, sum);
   return (sum == expected_sum) ? 1 : 0;
}


int parseargs_readinput(int argc, char *argv[], float *real_rasc, float *real_decl, float *rand_rasc, float *rand_decl)
{
   FILE *real_data_file, *rand_data_file, *out_file;
   float arcmin2rad = 1.0f / 60.0f / 180.0f * CUDART_PI_F;
   int Number_of_Galaxies;

   if (argc != 4)
   {
      printf("Usage: galaxy real_data random_data output_file\n   All MPI processes will be killed\n");
      return (1);
   }
   if (argc == 4)
   {
      printf("Running galaxy_mpi %s %s %s\n", argv[1], argv[2], argv[3]);

      real_data_file = fopen(argv[1], "r");
      if (real_data_file == NULL)
      {
         printf("Usage: galaxy  real_data  random_data  output_file\n");
         printf("ERROR: Cannot open real data file %s\n", argv[1]);
         return (1);
      }
      else
      {
         fscanf(real_data_file, "%d", &Number_of_Galaxies);
         for (int i = 0; i < INPUT_SIZE; ++i)
         {
            float rasc, decl;
            if (fscanf(real_data_file, "%f %f", &rasc, &decl) != 2)
            {
               printf("ERROR: Cannot read line %d in real data file %s\n", i + 1, argv[1]);
               fclose(real_data_file);
               return (1);
            }
            real_rasc[i] = rasc * arcmin2rad;
            real_decl[i] = decl * arcmin2rad;
         }
         fclose(real_data_file);
         printf("Successfully read %d lines from %s\n", INPUT_SIZE, argv[1]);
      }

      rand_data_file = fopen(argv[2], "r");
      if (rand_data_file == NULL)
      {
         printf("Usage: galaxy  real_data  random_data  output_file\n");
         printf("ERROR: Cannot open random data file %s\n", argv[2]);
         return (1);
      }
      else
      {
         fscanf(rand_data_file, "%d", &Number_of_Galaxies);
         for (int i = 0; i < INPUT_SIZE; ++i)
         {
            float rasc, decl;
            if (fscanf(rand_data_file, "%f %f", &rasc, &decl) != 2)
            {
               printf("ERROR: Cannot read line %d in real data file %s\n", i + 1, argv[2]);
               fclose(rand_data_file);
               return (1);
            }
            rand_rasc[i] = rasc * arcmin2rad;
            rand_decl[i] = decl * arcmin2rad;
         }
         fclose(rand_data_file);
         printf("Successfully read %d lines from %s\n", INPUT_SIZE, argv[2]);
      }
      out_file = fopen(argv[3], "w");
      if (out_file == NULL)
      {
         printf("Usage: galaxy  real_data  random_data  output_file\n");
         printf("ERROR: Cannot open output file %s\n", argv[3]);
         return (1);
      }
      else
         fclose(out_file);
   }

   return (0);
}


int getDevice(void)
{

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("         clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio    =    %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                  =    %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim                =    %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels            =    ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                =    %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(0);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}
