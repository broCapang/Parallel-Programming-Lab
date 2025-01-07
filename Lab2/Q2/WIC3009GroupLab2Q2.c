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
} Particle;

Particle particles[NUM_BODIES];
double forceVectors[NUM_BODIES][3];
omp_lock_t particleLocks[NUM_BODIES];

void initialize_particles() {
    for (int i = 0; i < NUM_BODIES; i++) {
        particles[i].a = rand() / (double)RAND_MAX;
        particles[i].b = rand() / (double)RAND_MAX;
        particles[i].c = rand() / (double)RAND_MAX;
        particles[i].v_a = rand() / (double)RAND_MAX - 0.5;
        particles[i].v_b = rand() / (double)RAND_MAX - 0.5;
        particles[i].v_c = rand() / (double)RAND_MAX - 0.5;
        particles[i].mass = rand() / (double)RAND_MAX + 1.0;

        omp_init_lock(&particleLocks[i]);
    }
}

void calculate_forces() {

    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        for (int j = 0; j < 3; j++) {
            forceVectors[i][j] = 0.0;
        }
    }

    #pragma omp parallel for
    for (int index = 0; index < NUM_BODIES * (NUM_BODIES - 1) / 2; index++) {
        int bodyA = (int)((sqrt(8.0 * index + 1) - 1) / 2);
        int bodyB = index - bodyA * (bodyA + 1) / 2;

        double dist_a = particles[bodyB].a - particles[bodyA].a;
        double dist_b = particles[bodyB].b - particles[bodyA].b;
        double dist_c = particles[bodyB].c - particles[bodyA].c;
        double distance = sqrt(dist_a * dist_a + dist_b * dist_b + dist_c * dist_c) + 1e-10;
        double gravitationalForce = (particles[bodyA].mass * particles[bodyB].mass) / (distance * distance);

        double force_a = gravitationalForce * dist_a / distance;
        double force_b = gravitationalForce * dist_b / distance;
        double force_c = gravitationalForce * dist_c / distance;

        omp_set_lock(&particleLocks[bodyA]);
        forceVectors[bodyA][0] += force_a;
        forceVectors[bodyA][1] += force_b;
        forceVectors[bodyA][2] += force_c;
        omp_unset_lock(&particleLocks[bodyA]);

        omp_set_lock(&particleLocks[bodyB]);
        forceVectors[bodyB][0] -= force_a;
        forceVectors[bodyB][1] -= force_b;
        forceVectors[bodyB][2] -= force_c;
        omp_unset_lock(&particleLocks[bodyB]);
    }
}

void update_particle_positions() {
    #pragma omp parallel for
    for (int i = 0; i < NUM_BODIES; i++) {
        particles[i].v_a += (forceVectors[i][0] / particles[i].mass) * TIME_STEP;
        particles[i].v_b += (forceVectors[i][1] / particles[i].mass) * TIME_STEP;
        particles[i].v_c += (forceVectors[i][2] / particles[i].mass) * TIME_STEP;

        particles[i].a += particles[i].v_a * TIME_STEP;
        particles[i].b += particles[i].v_b * TIME_STEP;
        particles[i].c += particles[i].v_c * TIME_STEP;
    }
}

void display_sample_positions(int step) {
    printf("Step %d:\n", step);
    for (int i = 0; i < 5; i++) { 
        printf("Particle %d: Position (%.3f, %.3f, %.3f), Velocity (%.3f, %.3f, %.3f)\n",
               i, particles[i].a, particles[i].b, particles[i].c,
               particles[i].v_a, particles[i].v_b, particles[i].v_c);
    }
}

void display_kinetic_energy(int step) {
    double totalKineticEnergy = 0.0;

    #pragma omp parallel for reduction(+:totalKineticEnergy)
    for (int i = 0; i < NUM_BODIES; i++) {
        double velocitySquared = particles[i].v_a * particles[i].v_a + particles[i].v_b * particles[i].v_b + particles[i].v_c * particles[i].v_c;
        totalKineticEnergy += 0.5 * particles[i].mass * velocitySquared;
    }

    printf("Step %d: Total Kinetic Energy = %.3f\n", step, totalKineticEnergy);
}

void cleanup_locks() {
    for (int i = 0; i < NUM_BODIES; i++) {
        omp_destroy_lock(&particleLocks[i]);
    }
}

int main() {
    initialize_particles();

    double startTime = omp_get_wtime();

    for (int step = 0; step < NUM_STEPS; step++) {
        calculate_forces();
        update_particle_positions();

        if (step % 10 == 0 || step == NUM_STEPS - 1) {
            display_sample_positions(step);
            display_kinetic_energy(step);
        }
    }

    double endTime = omp_get_wtime();
    printf("Simulation completed in %.3f seconds.\n", endTime - startTime);

    cleanup_locks();

    return 0;
}
