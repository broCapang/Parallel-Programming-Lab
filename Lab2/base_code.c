// gcc -fopenmp base_code.c -o base_code -lm
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NUM_BODIES 1000
#define NUM_STEPS 100
#define TIME_STEP 0.01

typedef struct {
    double x, y, z;     // Position
    double vx, vy, vz;  // Velocity
    double mass;        // Mass
} Body;

Body bodies[NUM_BODIES];
double forces[NUM_BODIES][3];

// Initialize bodies with random values
void initialize_bodies() {
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].x = rand() / (double)RAND_MAX;
        bodies[i].y = rand() / (double)RAND_MAX;
        bodies[i].z = rand() / (double)RAND_MAX;
        bodies[i].vx = rand() / (double)RAND_MAX - 0.5;
        bodies[i].vy = rand() / (double)RAND_MAX - 0.5;
        bodies[i].vz = rand() / (double)RAND_MAX - 0.5;
        bodies[i].mass = rand() / (double)RAND_MAX + 1.0;
    }
}

// Compute forces on all bodies
void compute_forces() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = 0; j < 3; j++) {
            forces[i][j] = 0.0;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = i + 1; j < NUM_BODIES; j++) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dz = bodies[j].z - bodies[i].z;
            double distance = sqrt(dx * dx + dy * dy + dz * dz) + 1e-10;
            double force = (bodies[i].mass * bodies[j].mass) / (distance * distance);

            double fx = force * dx / distance;
            double fy = force * dy / distance;
            double fz = force * dz / distance;

            #pragma omp critical
            {
                forces[i][0] += fx;
                forces[i][1] += fy;
                forces[i][2] += fz;

                forces[j][0] -= fx;
                forces[j][1] -= fy;
                forces[j][2] -= fz;
            }
        }
    }
}

// Update body positions and velocities based on forces
void update_bodies() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        bodies[i].vx += (forces[i][0] / bodies[i].mass) * TIME_STEP;
        bodies[i].vy += (forces[i][1] / bodies[i].mass) * TIME_STEP;
        bodies[i].vz += (forces[i][2] / bodies[i].mass) * TIME_STEP;

        bodies[i].x += bodies[i].vx * TIME_STEP;
        bodies[i].y += bodies[i].vy * TIME_STEP;
        bodies[i].z += bodies[i].vz * TIME_STEP;
    }
}

// Display positions and velocities for a sample of bodies
void display_sample(int step) {
    printf("Step %d:\n", step);
    for (int i = 0; i < 5; i++) { // Display only the first 5 bodies
        printf("Body %d: Position (%f, %f, %f), Velocity (%f, %f, %f)\n",
               i, bodies[i].x, bodies[i].y, bodies[i].z,
               bodies[i].vx, bodies[i].vy, bodies[i].vz);
    }
}

// Calculate and print the total kinetic energy of the system
void calculate_kinetic_energy(int step) {
    double total_energy = 0.0;

    #pragma omp parallel for reduction(+:total_energy)
    for (int i = 0; i < NUM_BODIES; i++) {
        double speed_squared = bodies[i].vx * bodies[i].vx +
                               bodies[i].vy * bodies[i].vy +
                               bodies[i].vz * bodies[i].vz;
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

        // Display information every 10 steps or at the last step
        if (step % 10 == 0 || step == NUM_STEPS - 1) {
            display_sample(step);
            calculate_kinetic_energy(step);
        }
    }

    double end_time = omp_get_wtime();
    printf("Simulation completed in %f seconds.\n", end_time - start_time);

    return 0;
}

