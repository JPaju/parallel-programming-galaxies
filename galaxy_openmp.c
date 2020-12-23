// On dione, first load the gcc module:
//    module load gcc
// For OpemMP programs, compile with
//    gcc -O3 -fopenmp -o galaxy_openmp galaxy_openmp.c -lm
// and run with
//    srun -N 1 -c 40 ./galaxy_openmp RealGalaxies_100k_arcmin.txt SyntheticGalaxies_100k_arcmin.txt omega.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

static const long INPUT_SIZE = 100000L;
static const long long RESULT_SIZE = (long long)INPUT_SIZE * (long long)INPUT_SIZE;

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float *real_x, *real_y, *real_z, *rand_x, *rand_y, *rand_z;
float pif;
long int MemoryAllocatedCPU = 0L;

int main(int argc, char *argv[])
{
    int parseargs_readinput(int argc, char *argv[]);

    int check_correct_hist_sum(long int to_sum[], int arr_length, long long expected_sum, char hist_name[]);
    int calculate_histogram_index(float angle);
    float radian_to_degrees(float rad);
    float ensure_max_1(float f);

    struct timeval _ttime;
    struct timezone _tzone;

    pif = acosf(-1.0f);

    gettimeofday(&_ttime, &_tzone);
    double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    real_rasc = (float *)calloc(INPUT_SIZE, sizeof(float));
    real_decl = (float *)calloc(INPUT_SIZE, sizeof(float));
    rand_rasc = (float *)calloc(INPUT_SIZE, sizeof(float));
    rand_decl = (float *)calloc(INPUT_SIZE, sizeof(float));

    real_x = (float *)calloc(INPUT_SIZE, sizeof(float));
    real_y = (float *)calloc(INPUT_SIZE, sizeof(float));
    real_z = (float *)calloc(INPUT_SIZE, sizeof(float));
    rand_x = (float *)calloc(INPUT_SIZE, sizeof(float));
    rand_y = (float *)calloc(INPUT_SIZE, sizeof(float));
    rand_z = (float *)calloc(INPUT_SIZE, sizeof(float));
    MemoryAllocatedCPU += 10L * INPUT_SIZE * sizeof(float);

    long int histogram_DD[360] = {0L};
    long int histogram_DR[360] = {0L};
    long int histogram_RR[360] = {0L};
    MemoryAllocatedCPU += 3L * 360L * sizeof(long int);

    if (parseargs_readinput(argc, argv) != 0)
    {
        printf("   Program stopped.\n");
        return (0);
    }
    printf("   Input data read, now calculating histograms\n");

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        float t = pif / 2 - real_decl[i];
        float p = real_rasc[i];
        real_x[i] = cos(p) * sin(t);
        real_y[i] = sin(p) * sin(t);
        real_z[i] = cos(t);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        float t = pif / 2 - rand_decl[i];
        float p = rand_rasc[i];
        rand_x[i] = cos(p) * sin(t);
        rand_y[i] = sin(p) * sin(t);
        rand_z[i] = cos(t);
    }

    printf("   Computing DD...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : histogram_DD)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        float xx1 = real_x[i];
        float yy1 = real_y[i];
        float zz1 = real_z[i];

        for (int j = i; j < INPUT_SIZE; j++)
        {
            float xx2 = real_x[j];
            float yy2 = real_y[j];
            float zz2 = real_z[j];

            float radians = ensure_max_1(xx1 * xx2 + yy1 * yy2 + zz1 * zz2);
            float angle = radian_to_degrees(radians);
            int index = calculate_histogram_index(angle);
            histogram_DD[index] += (i == j) ? 1L : 2L;
        }
    }

    printf("   Computing RR...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : histogram_RR)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        float xx1 = rand_x[i];
        float yy1 = rand_y[i];
        float zz1 = rand_z[i];

        for (int j = i; j < INPUT_SIZE; j++)
        {
            float xx2 = rand_x[j];
            float yy2 = rand_y[j];
            float zz2 = rand_z[j];

            float radians = ensure_max_1(xx1 * xx2 + yy1 * yy2 + zz1 * zz2);
            float angle = radian_to_degrees(radians);
            int index = calculate_histogram_index(angle);
            histogram_RR[index] += (i == j) ? 1L : 2L;
        }
    }

    printf("   Computing DR...\n");
#pragma omp parallel for schedule(dynamic) reduction(+ \
                                                     : histogram_DR)
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        float xx1 = real_x[i];
        float yy1 = real_y[i];
        float zz1 = real_z[i];

        for (int j = 0; j < INPUT_SIZE; j++)
        {
            float xx2 = rand_x[j];
            float yy2 = rand_y[j];
            float zz2 = rand_z[j];

            float radians = ensure_max_1(xx1 * xx2 + yy1 * yy2 + zz1 * zz2);
            float angle = radian_to_degrees(radians);
            int index = calculate_histogram_index(angle);
            histogram_DR[index] += 1L;
        }
    }

    // check point: the sum of all historgram entries should be 10 000 000 000
    int correct_value = 1;
    if (check_correct_hist_sum(histogram_DD, 360, RESULT_SIZE, "DD") == 0)
        correct_value = 0;
    if (check_correct_hist_sum(histogram_RR, 360, RESULT_SIZE, "RR") == 0)
        correct_value = 0;
    if (check_correct_hist_sum(histogram_DR, 360, RESULT_SIZE, "DR") == 0)
        correct_value = 0;

    if (correct_value != 1)
    {
        printf("   Histogram sums should be %lld. Ending program prematurely\n", RESULT_SIZE);
        return (0);
    }

    printf("   Omega values for the histograms:\n");
    float omega[360];
    for (int i = 0; i < 10; ++i)
        if (histogram_RR[i] != 0L)
        {
            omega[i] = (histogram_DD[i] - 2L * histogram_DR[i] + histogram_RR[i]) / ((float)(histogram_RR[i]));
            if (i < 10)
                printf("      angle %.2f deg. -> %.2f deg. : %.3f\n", i * 0.25, (i + 1) * 0.25, omega[i]);
        }

    FILE *out_file = fopen(argv[3], "w");
    if (out_file == NULL)
        printf("   ERROR: Cannot open output file %s\n", argv[3]);
    else
    {
        for (int i = 0; i < 360; ++i)
            if (histogram_RR[i] != 0L)
                fprintf(out_file, "%.2f  : %.3f\n", i * 0.25, omega[i]);
        fclose(out_file);
        printf("   Omega values written to file %s\n", argv[3]);
    }

    free(real_rasc);
    free(real_decl);
    free(rand_rasc);
    free(rand_decl);

    free(real_x);
    free(real_y);
    free(real_z);
    free(rand_x);
    free(rand_y);
    free(rand_z);

    printf("   Total memory allocated = %.1lf MB\n", MemoryAllocatedCPU / 1000000.0);
    gettimeofday(&_ttime, &_tzone);
    double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.0;

    printf("   Wall clock run time    = %.1lf secs\n", time_end - time_start);

    return (0);
}

int calculate_histogram_index(float angle)
{
    return (int)(4.0f * angle);
}

float radian_to_degrees(float rad)
{
    return acosf(rad) / pif * 180.0f;
}

float ensure_max_1(float f)
{
    return (f > 1.0f) ? 1.0f : f;
}

int check_correct_hist_sum(long int to_sum[], int arr_length, long long expected_sum, char hist_name[])
{
    long long sum = 0LL;
    for (int i = 0; i < arr_length; ++i)
        sum += to_sum[i];

    printf("   Histogram %s : sum = %lld\n", hist_name, sum);
    return (sum == expected_sum) ? 1 : 0;
}

int parseargs_readinput(int argc, char *argv[])
{
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f / 60.0f / 180.0f * pif;
    int Number_of_Galaxies;

    if (argc != 4)
    {
        printf("   Usage: galaxy real_data random_data output_file\n   All MPI processes will be killed\n");
        return (1);
    }
    if (argc == 4)
    {
        printf("   Running galaxy_openmp %s %s %s\n", argv[1], argv[2], argv[3]);

        real_data_file = fopen(argv[1], "r");
        if (real_data_file == NULL)
        {
            printf("   Usage: galaxy  real_data  random_data  output_file\n");
            printf("   ERROR: Cannot open real data file %s\n", argv[1]);
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
                    printf("   ERROR: Cannot read line %d in real data file %s\n", i + 1, argv[1]);
                    fclose(real_data_file);
                    return (1);
                }
                real_rasc[i] = rasc * arcmin2rad;
                real_decl[i] = decl * arcmin2rad;
            }
            fclose(real_data_file);
            printf("   Successfully read %d lines from %s\n", INPUT_SIZE, argv[1]);
        }

        rand_data_file = fopen(argv[2], "r");
        if (rand_data_file == NULL)
        {
            printf("   Usage: galaxy  real_data  random_data  output_file\n");
            printf("   ERROR: Cannot open random data file %s\n", argv[2]);
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
                    printf("   ERROR: Cannot read line %d in real data file %s\n", i + 1, argv[2]);
                    fclose(rand_data_file);
                    return (1);
                }
                rand_rasc[i] = rasc * arcmin2rad;
                rand_decl[i] = decl * arcmin2rad;
            }
            fclose(rand_data_file);
            printf("   Successfully read %d lines from %s\n", INPUT_SIZE, argv[2]);
        }
        out_file = fopen(argv[3], "w");
        if (out_file == NULL)
        {
            printf("   Usage: galaxy  real_data  random_data  output_file\n");
            printf("   ERROR: Cannot open output file %s\n", argv[3]);
            return (1);
        }
        else
            fclose(out_file);
    }

    return (0);
}
