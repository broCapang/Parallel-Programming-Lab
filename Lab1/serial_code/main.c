#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define MAX_VALUES 1000 // Maximum number of values to handle

void create_histogram(double *values, int count, double min, double max) {
    int bins = (int)(max - min) + 1; // Each bin covers 1.0 unit (e.g., 1.0-1.9, 2.0-2.9)
    int *histogram = (int *)calloc(bins, sizeof(int));
    if (!histogram) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    // Populate histogram
    for (int i = 0; i < count; i++) {
        int bin_index = (int)(values[i] - min);
        if (bin_index >= 0 && bin_index < bins) {
            histogram[bin_index]++;
        }
    }

    // Print histogram
    printf("Histogram:\n");
    for (int i = 0; i < bins; i++) {
        double bin_start = min + i;
        double bin_end = bin_start + 0.9;
        printf("%.1f - %.1f: ", bin_start, bin_end);
        for (int j = 0; j < histogram[i]; j++) {
            printf("*");
        }
        printf("\n");
    }

    free(histogram);
}

int main() {
     double start_time = omp_get_wtime();
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

    // Determine range
    double min = values[0], max = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }

    // Adjust min and max to align with bin boundaries
    min = (int)min; // Start bins at the lowest integer
    max = (int)max; 

    create_histogram(values, count, min, max);
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    printf("Execution Time: %f seconds\n", execution_time);

    return 0;
}

