//gcc -fopenmp base_code.c -o base_code -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define NUM_BODIES 10000
#define NUM_STEPS 10
#define TIME_STEP 0.01
#define G 6.67430e-11 // Gravitational constant

typedef struct {
    double x, y;      // Position
    double vx, vy;    // Velocity
    double mass;      // Mass
} Body;

Body bodies[NUM_BODIES];
double forces[NUM_BODIES][2];

// Initialize bodies with random values
void initialize_bodies() {
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].x = rand() / (double)RAND_MAX;
        bodies[i].y = rand() / (double)RAND_MAX;
        bodies[i].vx = rand() / (double)RAND_MAX - 0.5;
        bodies[i].vy = rand() / (double)RAND_MAX - 0.5;
        bodies[i].mass = rand() / (double)RAND_MAX + 1.0;
    }
}

// Compute pairwise forces using the reduced algorithm
void compute_forces() {
    // Initialize locks for each body
    omp_lock_t locks[NUM_BODIES];
    for (int i = 0; i < NUM_BODIES; i++) {
        omp_init_lock(&locks[i]);
    }

    // Reset forces
    for (int i = 0; i < NUM_BODIES; i++) {
        forces[i][0] = 0.0;
        forces[i][1] = 0.0;
    }

    // Compute forces
    #pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < NUM_BODIES; q++) {
        for (int k = q + 1; k < NUM_BODIES; k++) {
            double x_diff = bodies[q].x - bodies[k].x;
            double y_diff = bodies[q].y - bodies[k].y;

            double dist = sqrt(x_diff * x_diff + y_diff * y_diff) + 1e-10; // Avoid division by zero
            double dist_cubed = dist * dist * dist;

            double force_qk_x = G * bodies[q].mass * bodies[k].mass / dist_cubed * x_diff;
            double force_qk_y = G * bodies[q].mass * bodies[k].mass / dist_cubed * y_diff;

            // Protect updates to forces[q] and forces[k] using locks
            omp_set_lock(&locks[q]);
            forces[q][0] += force_qk_x;
            forces[q][1] += force_qk_y;
            omp_unset_lock(&locks[q]);

            omp_set_lock(&locks[k]);
            forces[k][0] -= force_qk_x;
            forces[k][1] -= force_qk_y;
            omp_unset_lock(&locks[k]);
        }
    }

    // Destroy locks
    for (int i = 0; i < NUM_BODIES; i++) {
        omp_destroy_lock(&locks[i]);
    }
}


// Update positions and velocities based on forces
void update_bodies() {
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].vx += (forces[i][0] / bodies[i].mass) * TIME_STEP;
        bodies[i].vy += (forces[i][1] / bodies[i].mass) * TIME_STEP;

        bodies[i].x += bodies[i].vx * TIME_STEP;
        bodies[i].y += bodies[i].vy * TIME_STEP;
    }
}

// Display positions and velocities for a sample of bodies
void display_sample(int step) {
    printf("Step %d:\n", step);
    for (int i = 0; i < 5; i++) { // Display the first 5 bodies
        printf("Body %d: Position (%f, %f), Velocity (%f, %f)\n",
               i, bodies[i].x, bodies[i].y, bodies[i].vx, bodies[i].vy);
    }
}

// Calculate and print the total kinetic energy of the system
void calculate_kinetic_energy(int step) {
    double total_energy = 0.0;


   for (int i = 0; i < NUM_BODIES; i++) {
        double speed_squared = bodies[i].vx * bodies[i].vx +
                               bodies[i].vy * bodies[i].vy;
        total_energy += 0.5 * bodies[i].mass * speed_squared;
    }

    printf("Step %d: Total Kinetic Energy = %f\n", step, total_energy);
}

int main() {
    // Initialize bodies
    initialize_bodies();

    double start_time = omp_get_wtime();
    // Simulation loop
    for (int step = 0; step < NUM_STEPS; step++) {
        compute_forces();
        update_bodies();

        // Display every 10 steps or the last step
        if (step % 10 == 0 || step == NUM_STEPS - 1) {
            display_sample(step);
            calculate_kinetic_energy(step);
        }
    }
    double end_time = omp_get_wtime();
    printf("Simulation completed in %f seconds.\n", end_time - start_time);


    return 0;
}

