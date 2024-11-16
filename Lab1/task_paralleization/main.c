#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_VALUES 1000 // Maximum number of values to handle



void create_histogram_serial(double *values, int count, double min, double max, double *execution_time) {
    double bin_width = 1.0;
    int bins = (int)((max - min) / bin_width);
    int *histogram = (int *)calloc(bins, sizeof(int));
    if (!histogram) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    double start_time = omp_get_wtime();

    // Populate histogram
    for (int i = 0; i < count; i++) {
        int bin_index = (int)((values[i] - min) / bin_width);
        if (bin_index >= 0 && bin_index < bins) {
            histogram[bin_index]++;
        }
    }

    double end_time = omp_get_wtime();
    *execution_time = end_time - start_time;

    // Print histogram
    printf("Histogram (Serial):\n");
    for (int i = 0; i < bins; i++) {
        double bin_start = min + i * bin_width;
        double bin_end = bin_start + bin_width;
        printf("%.1f - %.1f: ", bin_start, bin_end);
        for (int j = 0; j < histogram[i]; j++) {
            printf("*");
        }
        printf("\n");
    }

    free(histogram);
}


void create_histogram_task_parallel(double *values, int count, double min, double max, double serial_time) {
    int bins = (int)(max - min) + 1; // Each bin covers 1.0 unit
    int *histogram = (int *)calloc(bins, sizeof(int));
    if (!histogram) {
        fprintf(stderr, "Memory allocation failed for histogram.\n");
        return;
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task 1: Populate histogram
            #pragma omp task
            {
                for (int i = 0; i < count; i++) {
                    int bin_index = (int)(values[i] - min);
                    if (bin_index >= 0 && bin_index < bins) {
                        #pragma omp atomic
                        histogram[bin_index]++;
                    }
                }
            }

            // Taskwait to ensure histogram is populated before printing
            #pragma omp taskwait

            // Task 2: Print histogram
            #pragma omp task
            {
                printf("Histogram (Task Parallel):\n");
                for (int i = 0; i < bins; i++) {
                    double bin_start = min + i;
                    double bin_end = bin_start + 0.9;
                    printf("%.1f - %.1f: ", bin_start, bin_end);
                    for (int j = 0; j < histogram[i]; j++) {
                        printf("*");
                    }
                    printf("\n");
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    double task_parallel_execution_time = end_time - start_time;

    // Compute Speed-up and Efficiency
    int threads_used = omp_get_max_threads();
    double speedup = serial_time / task_parallel_execution_time;
    double efficiency = speedup / threads_used * 100.0;

    printf("Task Parallel Execution Time: %f seconds\n", task_parallel_execution_time);
    printf("Speed-up: %f\n", speedup);
    printf("Efficiency: %f%%\n", efficiency);

    free(histogram);
}

int main() {
    double total_start_time = omp_get_wtime();

    FILE *file = fopen("../data/data50.txt", "r");
    if (!file) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    double values[MAX_VALUES];
    int count = 0;

    // Read values from file
    while (fscanf(file, "%lf", &values[count]) == 1) {
        count++;
        if (count >= MAX_VALUES) {
            fprintf(stderr, "Too many values in file.\n");
            break;
        }
    }
    fclose(file);

    if (count == 0) {
        fprintf(stderr, "No values read from file.\n");
        return 1;
    }

    // Determine range of values
    double min = values[0], max = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }

    // Adjust min and max to align with bin boundaries
    min = floor(min);
    max = ceil(max);

    // Run serial histogram to get serial execution time
    double serial_execution_time;
    create_histogram_serial(values, count, min, max, &serial_execution_time);

    // Set different thread counts for analysis
    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        omp_set_num_threads(num_threads);
        printf("\nRunning Task Parallel Version with %d thread(s):\n", num_threads);
        create_histogram_task_parallel(values, count, min, max, serial_execution_time);
    }

    double total_end_time = omp_get_wtime();
    double total_execution_time = total_end_time - total_start_time;

    printf("Total Execution Time: %f seconds\n", total_execution_time);

    return 0;
}
