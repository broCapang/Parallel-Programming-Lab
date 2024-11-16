// Compile with: gcc -fopenmp main.c -o main -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Function to read values from a file and return the count
size_t read_values(const char *filename, double **values) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file '%s'.\n", filename);
        return 0;
    }

    size_t count = 0;
    size_t capacity = 1000000; // Initial capacity of 1 million
    *values = (double *)malloc(capacity * sizeof(double));
    if (!*values) {
        fprintf(stderr, "Memory allocation failed for values array.\n");
        fclose(file);
        return 0;
    }

    while (fscanf(file, "%lf", &(*values)[count]) == 1) {
        count++;

        // Reallocate memory if needed
        if (count >= capacity) {
            capacity *= 2;
            double *temp = (double *)realloc(*values, capacity * sizeof(double));
            if (!temp) {
                fprintf(stderr, "Memory reallocation failed.\n");
                free(*values);
                fclose(file);
                return 0;
            }
            *values = temp;
        }
    }
    fclose(file);

    return count;
}

void create_histogram_combined(double *values, size_t count, double min, double max, double serial_time, int num_threads) {
    double bin_width = 1.0;
    int bins = (int)((max - min) / bin_width);

    int *histogram = (int *)calloc(bins, sizeof(int));
    if (!histogram) {
        fprintf(stderr, "Memory allocation failed for histogram.\n");
        return;
    }

    double start_time = omp_get_wtime();

    omp_set_num_threads(num_threads); // Set the number of threads

    // Use tasks for different stages, and data parallelism within tasks
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Task 1: Populate histogram
            #pragma omp task
            {
                // Data parallelism within the task using OpenMP for loop
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < count; i++) {
                    int bin_index = (int)((values[i] - min) / bin_width);
                    if (bin_index >= 0 && bin_index < bins) {
                        #pragma omp atomic
                        histogram[bin_index]++;
                    }
                }
            }

            // Task 2: Perform some additional computations (if any)
            #pragma omp task
            {
                // For demonstration, let's compute the mean value in parallel
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (size_t i = 0; i < count; i++) {
                    sum += values[i];
                }
                double mean = sum / count;
                // Store or use the mean as needed
                // For this example, we'll just print it
                #pragma omp critical
                {
                    printf("Mean Value: %f\n", mean);
                }
            }

            // Wait for both tasks to complete before proceeding
            #pragma omp taskwait

            // Task 3: Optionally process the histogram (e.g., find the mode)
            #pragma omp task
            {
                int max_count = 0;
                int mode_bin = 0;
                for (int i = 0; i < bins; i++) {
                    if (histogram[i] > max_count) {
                        max_count = histogram[i];
                        mode_bin = i;
                    }
                }
                double mode_value = min + mode_bin * bin_width + bin_width / 2.0;
                #pragma omp critical
                {
                    printf("Mode Value: %f with count %d\n", mode_value, max_count);
                }
            }

            // Wait for the last task to complete
            #pragma omp taskwait
        } // End of single region
    } // End of parallel region

    double end_time = omp_get_wtime();
    double combined_parallel_execution_time = end_time - start_time;

    // Compute Speed-up and Efficiency
    double speedup = serial_time / combined_parallel_execution_time;
    double efficiency = speedup / num_threads * 100.0;

    printf("Combined Parallel Execution Time: %f seconds\n", combined_parallel_execution_time);
    printf("Speed-up: %f\n", speedup);
    printf("Efficiency: %f%%\n", efficiency);

    free(histogram);
}

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

    // Additional computations (mean and mode)
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    double mean = sum / count;

    int max_count = 0;
    int mode_bin = 0;
    for (int i = 0; i < bins; i++) {
        if (histogram[i] > max_count) {
            max_count = histogram[i];
            mode_bin = i;
        }
    }
    double mode_value = min + mode_bin * bin_width + bin_width / 2.0;

    double end_time = omp_get_wtime();
    *execution_time = end_time - start_time;

    printf("Serial Execution Time: %f seconds\n", *execution_time);
    printf("Mean Value: %f\n", mean);
    printf("Mode Value: %f with count %d\n", mode_value, max_count);

    free(histogram);
}

int main() {
    double total_start_time = omp_get_wtime();

    double *values = NULL;
    size_t count = read_values("../data/data100000000.txt", &values);
    if (count == 0) {
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

    // Set different thread counts for analysis
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        printf("\nRunning Combined Data and Task Parallel Version with %d thread(s):\n", num_threads);
        create_histogram_combined(values, count, min, max, serial_execution_time, num_threads);
    }

    free(values);

    double total_end_time = omp_get_wtime();
    double total_execution_time = total_end_time - total_start_time;

    printf("Total Execution Time: %f seconds\n", total_execution_time);

    return 0;
}

