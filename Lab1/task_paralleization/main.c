// Compile with: gcc -fopenmp main.c -o main -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void create_histogram_serial(double *values, size_t count, double min, double max, double *execution_time) {
    double bin_width = 1.0;
    int bins = (int)((max - min) / bin_width);

    int *histogram = (int *)calloc(bins, sizeof(int));
    if (!histogram) {
        fprintf(stderr, "Memory allocation failed for histogram.\n");
        return;
    }

    double start_time = omp_get_wtime();

    // Populate histogram
    for (size_t i = 0; i < count; i++) {
        int bin_index = (int)((values[i] - min) / bin_width);
        if (bin_index >= 0 && bin_index < bins) {
            histogram[bin_index]++;
        }
    }

    double end_time = omp_get_wtime();
    *execution_time = end_time - start_time;

    // Optionally print histogram
    /*
    printf("Histogram (Serial):\n");
    for (int i = 0; i < bins; i++) {
        double bin_start = min + i * bin_width;
        double bin_end = bin_start + bin_width;
        printf("%.1f - %.1f: %d\n", bin_start, bin_end, histogram[i]);
    }
    */

    free(histogram);
}

void create_histogram_task_parallel(double *values, size_t count, double min, double max, double serial_time) {
    double bin_width = 1.0;
    int bins = (int)((max - min) / bin_width);

    int *histogram = (int *)calloc(bins, sizeof(int));
    if (!histogram) {
        fprintf(stderr, "Memory allocation failed for histogram.\n");
        return;
    }

    double start_time = omp_get_wtime();

    // Determine the number of threads
    int num_threads = omp_get_max_threads();

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Create tasks for each thread
            for (int t = 0; t < num_threads; t++) {
                #pragma omp task firstprivate(t)
                {
                    size_t chunk_size = count / num_threads;
                    size_t start_index = t * chunk_size;
                    size_t end_index = (t == num_threads - 1) ? count : start_index + chunk_size;

                    for (size_t i = start_index; i < end_index; i++) {
                        int bin_index = (int)((values[i] - min) / bin_width);
                        if (bin_index >= 0 && bin_index < bins) {
                            #pragma omp atomic
                            histogram[bin_index]++;
                        }
                    }
                }
            }
        } // End of single region
    } // End of parallel region

    double end_time = omp_get_wtime();
    double task_parallel_execution_time = end_time - start_time;

    // Compute Speed-up and Efficiency
    int threads_used = num_threads;
    double speedup = serial_time / task_parallel_execution_time;
    double efficiency = speedup / threads_used * 100.0;

    printf("Task Parallel Execution Time: %f seconds\n", task_parallel_execution_time);
    printf("Speed-up: %f\n", speedup);
    printf("Efficiency: %f%%\n", efficiency);

    // Optionally print histogram
    /*
    printf("Histogram (Task Parallel):\n");
    for (int i = 0; i < bins; i++) {
        double bin_start = min + i * bin_width;
        double bin_end = bin_start + bin_width;
        printf("%.1f - %.1f: %d\n", bin_start, bin_end, histogram[i]);
    }
    */

    free(histogram);
}

int main() {
    double total_start_time = omp_get_wtime();

    FILE *file = fopen("../data/data100000000.txt", "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    // Dynamic array for values
    double *values = NULL;
    size_t count = 0;
    size_t capacity = 0;
    const size_t INITIAL_CAPACITY = 1000000; // Start with 1 million entries

    // Allocate initial memory
    capacity = INITIAL_CAPACITY;
    values = (double *)malloc(capacity * sizeof(double));
    if (!values) {
        fprintf(stderr, "Memory allocation failed for values array.\n");
        fclose(file);
        return 1;
    }

    // Read values from file
    while (fscanf(file, "%lf", &values[count]) == 1) {
        count++;

        // Reallocate memory if needed
        if (count >= capacity) {
            capacity *= 2; // Double the capacity
            double *temp = (double *)realloc(values, capacity * sizeof(double));
            if (!temp) {
                fprintf(stderr, "Memory reallocation failed.\n");
                free(values);
                fclose(file);
                return 1;
            }
            values = temp;
        }
    }
    fclose(file);

    if (count == 0) {
        fprintf(stderr, "No values read from file.\n");
        free(values);
        return 1;
    }

    // Determine range of values
    double min = values[0], max = values[0];

    #pragma omp parallel for reduction(min:min) reduction(max:max)
    for (size_t i = 1; i < count; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }

    // Adjust min and max to align with bin boundaries
    min = floor(min);
    max = ceil(max);

    // Run serial histogram to get serial execution time
    double serial_execution_time;
    create_histogram_serial(values, count, min, max, &serial_execution_time);
    printf("\nSerial Execution Time: %f seconds\n", serial_execution_time);

    // Set different thread counts for analysis
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        omp_set_num_threads(num_threads);
        printf("\nRunning Task Parallel Version with %d thread(s):\n", num_threads);
        create_histogram_task_parallel(values, count, min, max, serial_execution_time);
    }

    free(values);

    double total_end_time = omp_get_wtime();
    double total_execution_time = total_end_time - total_start_time;

    printf("Total Execution Time: %f seconds\n", total_execution_time);

    return 0;
}

