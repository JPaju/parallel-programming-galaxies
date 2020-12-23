// On dione, first load the mpi module
//    module load openmpi
//
// For MPI programs, compile with
//    mpicc -O3 -o galaxy_mpi galaxy_mpi.c -lm
//
// and run with e.g. 100 cores
//    srun -n 100 ./galaxy_mpi data_100k_arcmin.dat rand_100k_arcmin.dat omega.out

// Uncomment as necessary
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const long INPUT_SIZE = 100000L;
static const long long RESULT_SIZE = (long long)INPUT_SIZE * (long long)INPUT_SIZE;

float pif;
long int MemoryAllocatedCPU = 0L;

int main(int argc, char *argv[])
{
    int parseargs_readinput(int argc, char *argv[], float *real_rasc, float *real_decl, float *rand_rasc, float *rand_decl);

    void arc_to_vector(float *decl, float *rasc, float *x, float *y, float *z);
    void count_histogram_values(float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, long int hist[], int start, int end);
    void count_histogram_values_single(float *x1, float *y1, float *z1, long int hist[], int start, int end);
    int check_correct_hist_sum(long int to_sum[], int arr_length, long long expected_sum, char hist_name[]);

    pif = acosf(-1.0f);

    // MPI initialization
    int rank_count, rank;
    MPI_Status status;
    double time_start;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        time_start = MPI_Wtime();

    // Memory allocation
    float *real_rasc = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *real_decl = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *rand_rasc = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *rand_decl = (float *)calloc(INPUT_SIZE, sizeof(float));

    float *real_x = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *real_y = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *real_z = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *rand_x = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *rand_y = (float *)calloc(INPUT_SIZE, sizeof(float));
    float *rand_z = (float *)calloc(INPUT_SIZE, sizeof(float));
    MemoryAllocatedCPU += 10L * INPUT_SIZE * sizeof(float);

    long priv_hist_DD[360] = {0L};
    long priv_hist_DR[360] = {0L};
    long priv_hist_RR[360] = {0L};
    MemoryAllocatedCPU += 3L * 360L * sizeof(long);

    long combined_hist_DD[360] = {0L};
    long combined_hist_DR[360] = {0L};
    long combined_hist_RR[360] = {0L};
    MemoryAllocatedCPU += 3L * 360L * sizeof(long);

    // Read input
    if (rank == 0)
    {
        if (parseargs_readinput(argc, argv, real_rasc, real_decl, rand_rasc, rand_decl) != 0)
        {
            printf("Program stopped.\n");
            return (0);
        }

        printf("Input data read, calculating histograms\n\n");
    }

    MPI_Bcast(real_rasc, INPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rand_rasc, INPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(real_decl, INPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rand_decl, INPUT_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int num_rows = INPUT_SIZE / rank_count;
    int start_row = rank * num_rows;

    if (rank == rank_count - 1 && num_rows * rank_count != INPUT_SIZE)
        num_rows = num_rows + (INPUT_SIZE - num_rows * rank_count);

    int end_row = start_row + num_rows;

    arc_to_vector(real_decl, real_rasc, real_x, real_y, real_z);
    arc_to_vector(rand_decl, rand_rasc, rand_x, rand_y, rand_z);

    count_histogram_values_single(real_x, real_y, real_z, priv_hist_DD, start_row, end_row);
    count_histogram_values_single(rand_x, rand_y, rand_z, priv_hist_RR, start_row, end_row);
    count_histogram_values(real_x, real_y, real_z, rand_x, rand_y, rand_z, priv_hist_DR, start_row, end_row);

    MPI_Reduce(priv_hist_DD, combined_hist_DD, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(priv_hist_RR, combined_hist_RR, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(priv_hist_DR, combined_hist_DR, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int correct_value = 1;
        if (check_correct_hist_sum(combined_hist_DD, 360, RESULT_SIZE, "DD") == 0)
            correct_value = 0;
        if (check_correct_hist_sum(combined_hist_RR, 360, RESULT_SIZE, "RR") == 0)
            correct_value = 0;
        if (check_correct_hist_sum(combined_hist_DR, 360, RESULT_SIZE, "DR") == 0)
            correct_value = 0;

        if (correct_value != 1)
        {
            printf("Histogram sums should be %lld. Ending program prematurely\n", RESULT_SIZE);
            MPI_Finalize();
            return (0);
        }

        printf("Omega values for the histograms:\n");
        float omega[360];
        for (int i = 0; i < 10; ++i)
            if (combined_hist_RR[i] != 0L)
            {
                omega[i] = (combined_hist_DD[i] - 2L * combined_hist_DR[i] + combined_hist_RR[i]) / ((float)(combined_hist_RR[i]));
                if (i < 10)
                    printf("   angle %.2f deg. -> %.2f deg. : %.3f\n", i * 0.25, (i + 1) * 0.25, omega[i]);
            }

        FILE *out_file = fopen(argv[3], "w");
        if (out_file == NULL)
            printf("ERROR: Cannot open output file %s\n", argv[3]);
        else
        {
            for (int i = 0; i < 360; ++i)
                if (combined_hist_RR[i] != 0L)
                    fprintf(out_file, "%.2f  : %.3f\n", i * 0.25, omega[i]);
            fclose(out_file);
            printf("Omega values written to file %s\n", argv[3]);
        }
    }

    free(real_rasc);
    free(real_decl);
    free(rand_rasc);
    free(rand_decl);

    free(rand_x);
    free(rand_y);
    free(rand_z);
    free(real_x);
    free(real_y);
    free(real_z);

    if (rank == 0)
    {
        printf("Total memory allocated = %.1lf MB\n", MemoryAllocatedCPU / 1000000.0);
        printf("Wall clock run time    = %.1lf secs\n", MPI_Wtime() - time_start);
    }

    MPI_Finalize();
    return (0);
}

void arc_to_vector(float *decl, float *rasc, float *x, float *y, float *z)
{
    for (int i = 0; i < INPUT_SIZE; i++)
    {
        float t = pif / 2 - decl[i];
        float p = rasc[i];
        x[i] = cos(p) * sin(t);
        y[i] = sin(p) * sin(t);
        z[i] = cos(t);
    }
}

void count_histogram_values(float *x1, float *y1, float *z1, float *x2, float *y2, float *z2, long int hist[], int start, int end)
{
    int calculate_histogram_index(float angle);
    float radian_to_degrees(float rad);
    float ensure_max_1(float f);

    for (int i = start; i < end; i++)
    {
        float xx1 = x1[i];
        float yy1 = y1[i];
        float zz1 = z1[i];

        for (int j = 0; j < INPUT_SIZE; j++)
        {
            float xx2 = x2[j];
            float yy2 = y2[j];
            float zz2 = z2[j];

            float radians = ensure_max_1(xx1 * xx2 + yy1 * yy2 + zz1 * zz2);
            float angle = radian_to_degrees(radians);
            int index = calculate_histogram_index(angle);
            hist[index] += 1L;
        }
    }
}

void count_histogram_values_single(float *x1, float *y1, float *z1, long int hist[], int start, int end)
{
    float radian_to_degrees(float rad);
    float ensure_max_1(float f);
    int calculate_histogram_index(float angle);

    for (int i = start; i < end; i++)
    {
        float xx1 = x1[i];
        float yy1 = y1[i];
        float zz1 = z1[i];

        for (int j = i; j < INPUT_SIZE; j++)
        {
            float xx2 = x1[j];
            float yy2 = y1[j];
            float zz2 = z1[j];

            float radians = ensure_max_1(xx1 * xx2 + yy1 * yy2 + zz1 * zz2);
            float angle = radian_to_degrees(radians);
            int index = calculate_histogram_index(angle);
            hist[index] += (i == j) ? 1L : 2L;
        }
    }
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

    printf("Histogram %s : sum = %lld\n", hist_name, sum);
    return (sum == expected_sum) ? 1 : 0;
}

int parseargs_readinput(int argc, char *argv[], float *real_rasc, float *real_decl, float *rand_rasc, float *rand_decl)
{
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f / 60.0f / 180.0f * pif;
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
