#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUM_BODIES 1000
#define NUM_STEPS 100
#define TIME_STEP 0.01

typedef struct {
    double a, b, c; 
    double v_a, v_b, v_c; 
    double mass;
} Object;

Object objects[NUM_BODIES];
double force_vectors[NUM_BODIES][3];


void initialize_objects() {
    for (int i = 0; i < NUM_BODIES; i++) {
        objects[i].a = rand() / (double)RAND_MAX;
        objects[i].b = rand() / (double)RAND_MAX;
        objects[i].c = rand() / (double)RAND_MAX;
        objects[i].v_a = rand() / (double)RAND_MAX - 0.5;
        objects[i].v_b = rand() / (double)RAND_MAX - 0.5;
        objects[i].v_c = rand() / (double)RAND_MAX - 0.5;
        objects[i].mass = rand() / (double)RAND_MAX + 1.0;
    }
}


void calculate_forces() {

    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = 0; j < 3; j++) {
            force_vectors[i][j] = 0.0;
        }
    }


    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = i + 1; j < NUM_BODIES; j++) {
            double dist_a = objects[j].a - objects[i].a;
            double dist_b = objects[j].b - objects[i].b;
            double dist_c = objects[j].c - objects[i].c;
            double distance = sqrt(dist_a * dist_a + dist_b * dist_b + dist_c * dist_c) + 1e-10; 
            double interaction_force = (objects[i].mass * objects[j].mass) / (distance * distance);

            double force_a = interaction_force * dist_a / distance;
            double force_b = interaction_force * dist_b / distance;
            double force_c = interaction_force * dist_c / distance;

            #pragma omp critical
            {
                force_vectors[i][0] += force_a;
                force_vectors[i][1] += force_b;
                force_vectors[i][2] += force_c;

                force_vectors[j][0] -= force_a;
                force_vectors[j][1] -= force_b;
                force_vectors[j][2] -= force_c;
            }
        }
    }
}


void update_object_positions() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        objects[i].v_a += (force_vectors[i][0] / objects[i].mass) * TIME_STEP;
        objects[i].v_b += (force_vectors[i][1] / objects[i].mass) * TIME_STEP;
        objects[i].v_c += (force_vectors[i][2] / objects[i].mass) * TIME_STEP;

        objects[i].a += objects[i].v_a * TIME_STEP;
        objects[i].b += objects[i].v_b * TIME_STEP;
        objects[i].c += objects[i].v_c * TIME_STEP; 
    }
}


void display_sample_positions(int step) {
    printf("Step %d:\n", step); 
    for (int i = 0; i < 5; i++) { 
        printf("Object %d: Position (%f, %f, %f), Velocity (%f, %f, %f)\n",
               i, objects[i].a, objects[i].b, objects[i].c,
               objects[i].v_a, objects[i].v_b, objects[i].v_c);
    }
}


void calculate_kinetic_energy(int step) {
    double total_kinetic_energy = 0.0;

    #pragma omp parallel for reduction(+:total_kinetic_energy)
    for (int i = 0; i < NUM_BODIES; i++) {
        double speed_squared = objects[i].v_a * objects[i].v_a +
                               objects[i].v_b * objects[i].v_b +
                               objects[i].v_c * objects[i].v_c;
        total_kinetic_energy += 0.5 * objects[i].mass * speed_squared;
    }

    printf("Step %d: Total Kinetic Energy = %f\n", step, total_kinetic_energy); 
}

int main() {
    initialize_objects();

    double start_time = omp_get_wtime();

    // Simulation loop
    for (int step = 0; step < NUM_STEPS; step++) {
        calculate_forces();
        update_object_positions();

        if (step % 10 == 0 || step == NUM_STEPS - 1) { 
            display_sample_positions(step);
            calculate_kinetic_energy(step);
        }
    }

    double end_time = omp_get_wtime();
    printf("Simulation completed in %f seconds.\n", end_time - start_time);

    return 0;
}
