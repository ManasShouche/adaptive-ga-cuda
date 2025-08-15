#ifndef GA_H
#define GA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_LINE_LENGTH 2048
#define MAX_FEATURES 42
#define MAX_CLASSES 23

// Data structures
typedef struct {
    int initialPopSize;
    int currentPopSize;
    int maxPopSize;
    float popSizeGrowthFactor;
    int maxGenerations;
    int maxStagnantGenerations;
    float convergenceThreshold;
    float baseMutationRate;
    float baseCrossoverRate;
    float operatorAdaptationRate;
    int stagnantGenerationCounter;
    float previousBestFitness;
    float currentBestFitness;
    float populationDiversity;
    float fitnessMovingAverage;
    float fitnessVariance;
    int fitnessHistoryWindow;
    double lastAdaptationTime;
    int generationsCompleted;
    int totalEvaluations;
} ControlParameters;

typedef struct {
    float* features;
    int* labels;
    int numSamples;
    int numFeatures;
    int numClasses;
    float* featureMin;
    float* featureMax;
} Dataset;

typedef struct {
    char* attackName;
    int classLabel;
} AttackMapping;

// Host-side function prototypes
int countLines(const char* filename);
int countFields(const char* line);
void createAttackMapping(AttackMapping** mappings, int* numMappings);
int getClassLabel(AttackMapping* mappings, int numMappings, const char* attackType);
void freeAttackMappings(AttackMapping* mappings, int numMappings);
Dataset loadKDDCupDataset(const char* filename);
void normalizeDataset(Dataset* dataset);
void freeDataset(Dataset* dataset);
void createInitialPopulation(float* h_population, int popSize, int features);
ControlParameters initializeControlParameters();
float updateDiversity(float* d_population, int popSize, int individualSize);
bool shouldIncreasePopulationSize(ControlParameters* params);
void adaptOperatorRates(ControlParameters* params, float* mutationRate, float* crossoverRate);
void updateControlParameters(ControlParameters* params, float newBestFitness, float avgFitness, float fitnessVariance);
bool checkTermination(ControlParameters* params);
void reallocateDeviceMemory(float** d_population, float** d_offspring, float** d_fitness, curandState** d_randStates, int oldPopSize, int newPopSize, int individualSize);
void evolveWithAutomaticControl(Dataset trainDataset, Dataset testDataset);

// Device-side kernel prototypes
__global__ void calculateDiversityKernel(float* population, int popSize, int individualSize, float* diversityResult);
__global__ void initializeRandomStatesKernel(curandState* states, unsigned long seed);
__global__ void evaluateFitnessKernel(float* population, float* fitness, float* data, int* labels, int popSize, int dataSize, int features, int numClasses, float accuracyWeight, float complexityWeight);
__global__ void adaptiveGeneticOperationsKernel(float* population, float* offspring, float* fitness, int popSize, int individualSize, float mutationRate, float crossoverRate, curandState* randStates);

#endif