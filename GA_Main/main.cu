#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>

#define MAX_LINE_LENGTH 2048
#define MAX_FEATURES 42  // KDD Cup dataset has 41 features plus the class label
#define MAX_CLASSES 23   // KDD Cup has multiple attack types plus normal

// Parameters for automatic control
typedef struct {
    // Population sizing
    int initialPopSize;
    int currentPopSize;
    int maxPopSize;
    float popSizeGrowthFactor;
    
    // Termination criteria
    int maxGenerations;
    int maxStagnantGenerations;
    float convergenceThreshold;
    
    // Genetic operators
    float baseMutationRate;
    float baseCrossoverRate;
    float operatorAdaptationRate;
    
    // Adaptive control
    int stagnantGenerationCounter;
    float previousBestFitness;
    float currentBestFitness;
    float populationDiversity;
    
    // Dynamic fitness statistics
    float fitnessMovingAverage;
    float fitnessVariance;
    int fitnessHistoryWindow;
    
    // Performance tracking
    double lastAdaptationTime;
    int generationsCompleted;
    int totalEvaluations;
} ControlParameters;

// Structure to store dataset information
typedef struct {
    float* features;
    int* labels;
    int numSamples;
    int numFeatures;
    int numClasses;
    float* featureMin;  // Min value for each feature (for normalization)
    float* featureMax;  // Max value for each feature (for normalization)
} Dataset;

// Mapping between attack types and numeric class labels
typedef struct {
    char* attackName;
    int classLabel;
} AttackMapping;

// Function to count lines in a file
int countLines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return -1;
    }
    
    int count = 0;
    char buffer[MAX_LINE_LENGTH];
    
    while (fgets(buffer, MAX_LINE_LENGTH, file)) {
        count++;
    }
    
    fclose(file);
    return count;
}

// Function to count fields in a CSV line
int countFields(const char* line) {
    int count = 1;  // Start with 1 for the first field
    
    for (int i = 0; line[i] != '\0'; i++) {
        if (line[i] == ',') {
            count++;
        }
    }
    
    return count;
}

// Function to map attack types to numeric class labels
void createAttackMapping(AttackMapping** mappings, int* numMappings) {
    // Initialize with known attack types from KDD Cup dataset
    AttackMapping* map = (AttackMapping*)malloc(MAX_CLASSES * sizeof(AttackMapping));
    int count = 0;
    
    // Normal traffic
    map[count].attackName = strdup("normal");
    map[count].classLabel = count;
    count++;
    
    // DoS attacks
    map[count].attackName = strdup("back");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("land");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("neptune");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("pod");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("smurf");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("teardrop");
    map[count].classLabel = count;
    count++;
    
    // Probe attacks
    map[count].attackName = strdup("ipsweep");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("nmap");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("portsweep");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("satan");
    map[count].classLabel = count;
    count++;
    
    // R2L attacks
    map[count].attackName = strdup("ftp_write");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("guess_passwd");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("imap");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("multihop");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("phf");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("spy");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("warezclient");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("warezmaster");
    map[count].classLabel = count;
    count++;
    
    // U2R attacks
    map[count].attackName = strdup("buffer_overflow");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("loadmodule");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("perl");
    map[count].classLabel = count;
    count++;
    
    map[count].attackName = strdup("rootkit");
    map[count].classLabel = count;
    count++;
    
    *mappings = map;
    *numMappings = count;
}

// Function to get class label from attack type
int getClassLabel(AttackMapping* mappings, int numMappings, const char* attackType) {
    for (int i = 0; i < numMappings; i++) {
        if (strcmp(mappings[i].attackName, attackType) == 0) {
            return mappings[i].classLabel;
        }
    }
    
    // If not found, return -1
    return -1;
}

// Function to free attack mappings
void freeAttackMappings(AttackMapping* mappings, int numMappings) {
    for (int i = 0; i < numMappings; i++) {
        free(mappings[i].attackName);
    }
    free(mappings);
}

// Function to load KDD Cup dataset from CSV
Dataset loadKDDCupDataset(const char* filename) {
    Dataset dataset;
    int numLines = countLines(filename);
    
    if (numLines <= 0) {
        printf("Error: Empty file or could not count lines in %s\n", filename);
        dataset.numSamples = 0;
        return dataset;
    }
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        dataset.numSamples = 0;
        return dataset;
    }
    
    // Read first line to determine number of fields
    char buffer[MAX_LINE_LENGTH];
    if (fgets(buffer, MAX_LINE_LENGTH, file) == NULL) {
        printf("Error reading first line from file: %s\n", filename);
        fclose(file);
        dataset.numSamples = 0;
        return dataset;
    }
    
    int numFields = countFields(buffer);
    
    // KDD Cup dataset has the class label as the last field
    dataset.numFeatures = numFields - 1;
    dataset.numSamples = numLines;
    
    // Allocate memory
    dataset.features = (float*)malloc(dataset.numSamples * dataset.numFeatures * sizeof(float));
    dataset.labels = (int*)malloc(dataset.numSamples * sizeof(int));
    dataset.featureMin = (float*)malloc(dataset.numFeatures * sizeof(float));
    dataset.featureMax = (float*)malloc(dataset.numFeatures * sizeof(float));
    
    // Initialize min/max values
    for (int i = 0; i < dataset.numFeatures; i++) {
        dataset.featureMin[i] = INFINITY;
        dataset.featureMax[i] = -INFINITY;
    }
    
    // Create attack mappings
    AttackMapping* mappings;
    int numMappings;
    createAttackMapping(&mappings, &numMappings);
    dataset.numClasses = numMappings;
    
    // Rewind file to start
    rewind(file);
    
    // Process each line
    int sampleIndex = 0;
    while (fgets(buffer, MAX_LINE_LENGTH, file) && sampleIndex < numLines) {
        char* token = strtok(buffer, ",");
        int featureIndex = 0;
        
        // Process each feature
        while (token != NULL && featureIndex < dataset.numFeatures) {
            // Check if the token is a number or symbolic
            float value;
            if (sscanf(token, "%f", &value) == 1) {
                // Numeric feature
                dataset.features[sampleIndex * dataset.numFeatures + featureIndex] = value;
            } else {
                // Symbolic feature - implement simple encoding
                // For simplicity, we'll just hash the string to a float value
                unsigned int hash = 0;
                for (int i = 0; token[i] != '\0'; i++) {
                    hash = 31 * hash + token[i];
                }
                value = (float)(hash % 100) / 100.0f;  // Normalize to [0,1]
                dataset.features[sampleIndex * dataset.numFeatures + featureIndex] = value;
            }
            
            // Update min/max
            if (value < dataset.featureMin[featureIndex]) {
                dataset.featureMin[featureIndex] = value;
            }
            if (value > dataset.featureMax[featureIndex]) {
                dataset.featureMax[featureIndex] = value;
            }
            
            token = strtok(NULL, ",");
            featureIndex++;
        }
        
        // Process class label (last field)
        if (token != NULL) {
            // Remove newline character if present
            int len = strlen(token);
            if (len > 0 && token[len-1] == '\n') {
                token[len-1] = '\0';
            }
            
            // Map attack type to class label
            int classLabel = getClassLabel(mappings, numMappings, token);
            if (classLabel == -1) {
                // If not found, add a new mapping
                classLabel = numMappings;
                mappings = (AttackMapping*)realloc(mappings, (numMappings + 1) * sizeof(AttackMapping));
                mappings[numMappings].attackName = strdup(token);
                mappings[numMappings].classLabel = classLabel;
                numMappings++;
                dataset.numClasses = numMappings;
            }
            
            dataset.labels[sampleIndex] = classLabel;
        }
        
        sampleIndex++;
    }
    
    fclose(file);
    
    // Update actual number of samples read
    dataset.numSamples = sampleIndex;
    
    // Free attack mappings
    freeAttackMappings(mappings, numMappings);
    
    printf("Loaded dataset from %s:\n", filename);
    printf("  Samples: %d\n", dataset.numSamples);
    printf("  Features: %d\n", dataset.numFeatures);
    printf("  Classes: %d\n", dataset.numClasses);
    
    return dataset;
}

// Function to normalize dataset features
void normalizeDataset(Dataset* dataset) {
    for (int i = 0; i < dataset->numSamples; i++) {
        for (int j = 0; j < dataset->numFeatures; j++) {
            if (dataset->featureMax[j] > dataset->featureMin[j]) {
                dataset->features[i * dataset->numFeatures + j] = 
                    (dataset->features[i * dataset->numFeatures + j] - dataset->featureMin[j]) / 
                    (dataset->featureMax[j] - dataset->featureMin[j]);
            } else {
                dataset->features[i * dataset->numFeatures + j] = 0.5f;  // Default if min == max
            }
        }
    }
}

// Free dataset memory
void freeDataset(Dataset* dataset) {
    free(dataset->features);
    free(dataset->labels);
    free(dataset->featureMin);
    free(dataset->featureMax);
}

// Kernel to measure population diversity
__global__ void calculateDiversityKernel(float* population, int popSize, int individualSize, float* diversityResult) {
    __shared__ float localSum[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sumOfSquaredDifferences = 0.0f;
    
    // Each thread processes multiple individuals
    for (int i = gid; i < popSize; i += blockDim.x * gridDim.x) {
        for (int j = i + 1; j < popSize; j++) {
            float distance = 0.0f;
            
            // Calculate Euclidean distance between individuals
            for (int k = 0; k < individualSize; k++) {
                float diff = population[i * individualSize + k] - population[j * individualSize + k];
                distance += diff * diff;
            }
            
            sumOfSquaredDifferences += sqrtf(distance);
        }
    }
    
    // Store in shared memory
    localSum[tid] = sumOfSquaredDifferences;
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            localSum[tid] += localSum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(diversityResult, localSum[0]);
    }
}

// Kernel to initialize random states
__global__ void initializeRandomStatesKernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Kernel to evaluate fitness for KDD Cup classification
__global__ void evaluateFitnessKernel(float* population, float* fitness, 
                                     float* data, int* labels, 
                                     int popSize, int dataSize, int features, int numClasses,
                                     float accuracyWeight, float complexityWeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < popSize) {
        // Get individual
        float* individual = &population[idx * features * 2];
        
        int correctPredictions = 0;
        float totalRange = 0.0f;
        
        // Calculate accuracy on dataset
        for (int i = 0; i < dataSize; i++) {
            // Simplified prediction logic
            int prediction = -1;
            float maxActivation = -1.0f;
            
            // For each class
            for (int c = 0; c < numClasses; c++) {
                float activation = 0.0f;
                
                // For each feature
                for (int j = 0; j < features; j++) {
                    float value = data[i * features + j];
                    float lowerBound = individual[j * 2];
                    float upperBound = individual[j * 2 + 1];
                    
                    // Feature contributes to activation if value is within bounds
                    if (value >= lowerBound && value <= upperBound) {
                        activation += 1.0f;
                    }
                }
                
                if (activation > maxActivation) {
                    maxActivation = activation;
                    prediction = c;
                }
            }
            
            // Check if prediction is correct
            if (prediction == labels[i]) {
                correctPredictions++;
            }
        }
        
        // Calculate complexity penalty (based on boundary ranges)
        for (int j = 0; j < features; j++) {
            totalRange += individual[j * 2 + 1] - individual[j * 2];
        }
        
        float complexityPenalty = 0.0f;
        if (features > 0) {
            complexityPenalty = totalRange / features; // Normalize by number of features
        }

        // Calculate fitness (weighted combination of accuracy and complexity)
        float accuracy = (float)correctPredictions / dataSize;
        fitness[idx] = accuracyWeight * accuracy - complexityWeight * complexityPenalty;
    }
}

// Kernel for adaptive genetic operations
__global__ void adaptiveGeneticOperationsKernel(float* population, float* offspring, float* fitness,
                                              int popSize, int individualSize, 
                                              float mutationRate, float crossoverRate,
                                              curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    // Get local random state
    curandState localState = randStates[idx];
    
    // Tournament selection for parents
    int parent1Idx = 0, parent2Idx = 0;
    
    // First parent selection (tournament of size 2)
    int candidate1 = curand(&localState) % popSize;
    int candidate2 = curand(&localState) % popSize;
    parent1Idx = (fitness[candidate1] > fitness[candidate2]) ? candidate1 : candidate2;
    
    // Second parent selection
    candidate1 = curand(&localState) % popSize;
    candidate2 = curand(&localState) % popSize;
    parent2Idx = (fitness[candidate1] > fitness[candidate2]) ? candidate1 : candidate2;
    
    // Initialize offspring with first parent
    for (int i = 0; i < individualSize; i++) {
        offspring[idx * individualSize + i] = population[parent1Idx * individualSize + i];
    }
    
    // Apply crossover with probability crossoverRate
    if (curand_uniform(&localState) < crossoverRate) {
        // Select crossover point (two-point crossover)
        int crossPoint1 = curand(&localState) % individualSize;
        int crossPoint2 = curand(&localState) % individualSize;
        
        // Ensure crossPoint1 <= crossPoint2
        if (crossPoint1 > crossPoint2) {
            int temp = crossPoint1;
            crossPoint1 = crossPoint2;
            crossPoint2 = temp;
        }
        
        // Apply crossover
        for (int i = crossPoint1; i <= crossPoint2; i++) {
            offspring[idx * individualSize + i] = population[parent2Idx * individualSize + i];
        }
    }
    
    // Apply adaptive mutation
    for (int i = 0; i < individualSize; i++) {
        if (curand_uniform(&localState) < mutationRate) {
            // Apply mutation - for feature boundaries, we need to respect constraints
            if (i % 2 == 0) { // Lower bound
                float upperBound = offspring[idx * individualSize + i + 1];
                float mutation = curand_normal(&localState) * 0.1f;
                float newValue = offspring[idx * individualSize + i] + mutation;
                
                // Ensure valid bounds (between 0 and upper bound)
                if (newValue < 0.0f) newValue = 0.0f;
                if (newValue > upperBound) newValue = upperBound;
                
                offspring[idx * individualSize + i] = newValue;
            } else { // Upper bound
                float lowerBound = offspring[idx * individualSize + i - 1];
                float mutation = curand_normal(&localState) * 0.1f;
                float newValue = offspring[idx * individualSize + i] + mutation;
                
                // Ensure valid bounds (between lower bound and 1)
                if (newValue < lowerBound) newValue = lowerBound;
                if (newValue > 1.0f) newValue = 1.0f;
                
                offspring[idx * individualSize + i] = newValue;
            }
        }
    }
    
    // Update random state
    randStates[idx] = localState;
}

// Function to create initial population
void createInitialPopulation(float* h_population, int popSize, int features) {
    int individualSize = features * 2; // Lower and upper bound for each feature
    
    for (int i = 0; i < popSize; i++) {
        for (int j = 0; j < features; j++) {
            // Create random lower and upper bounds [0,1] ensuring lower < upper
            float lowerBound = (float)rand() / RAND_MAX * 0.5f;
            float upperBound = lowerBound + (float)rand() / RAND_MAX * (1.0f - lowerBound);
            
            h_population[i * individualSize + j * 2] = lowerBound;
            h_population[i * individualSize + j * 2 + 1] = upperBound;
        }
    }
}

// Initialize control parameters
ControlParameters initializeControlParameters() {
    ControlParameters params;
    
    // Population sizing
    params.initialPopSize = 64;
    params.currentPopSize = params.initialPopSize;
    params.maxPopSize = 4096;
    params.popSizeGrowthFactor = 2.0f;
    
    // Termination criteria
    params.maxGenerations = 500;
    params.maxStagnantGenerations = 15;
    params.convergenceThreshold = 0.95f;  // Adjusted for KDD Cup classification
    
    // Genetic operators
    params.baseMutationRate = 0.1f;
    params.baseCrossoverRate = 0.8f;
    params.operatorAdaptationRate = 0.05f;
    
    // Adaptive control
    params.stagnantGenerationCounter = 0;
    params.previousBestFitness = 0.0f;
    params.currentBestFitness = 0.0f;
    params.populationDiversity = 1.0f;
    
    // Dynamic fitness statistics
    params.fitnessMovingAverage = 0.0f;
    params.fitnessVariance = 0.0f;
    params.fitnessHistoryWindow = 10;
    
    // Performance tracking
    params.lastAdaptationTime = 0.0;
    params.generationsCompleted = 0;
    params.totalEvaluations = 0;
    
    return params;
}

// Function to measure and update population diversity
float updateDiversity(float* d_population, int popSize, int individualSize) {
    float* d_diversity;
    float h_diversity = 0.0f;
    
    cudaMalloc(&d_diversity, sizeof(float));
    cudaMemset(d_diversity, 0, sizeof(float));
    
    int blockSize = 256;
    int gridSize = (popSize + blockSize - 1) / blockSize;
    
    calculateDiversityKernel<<<gridSize, blockSize>>>(d_population, popSize, individualSize, d_diversity);
    
    cudaMemcpy(&h_diversity, d_diversity, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_diversity);
    
    // Normalize by maximum possible pairs
    int maxPairs = (popSize * (popSize - 1)) / 2;
    if (maxPairs > 0) {
        h_diversity /= maxPairs;
    }
    
    return h_diversity;
}

// Function to decide if population size should increase
bool shouldIncreasePopulationSize(ControlParameters* params) {
    // Check if we're stagnating but still have room to grow
    if (params->stagnantGenerationCounter >= params->maxStagnantGenerations / 2 && 
        params->currentPopSize < params->maxPopSize) {
        return true;
    }
    
    // Check if diversity is too low
    if (params->populationDiversity < 0.1f && params->currentPopSize < params->maxPopSize) {
        return true;
    }
    
    return false;
}

// Function to adapt mutation and crossover rates based on progress
void adaptOperatorRates(ControlParameters* params, float* mutationRate, float* crossoverRate) {
    // Increase mutation when stagnation occurs
    if (params->stagnantGenerationCounter > 0) {
        *mutationRate = params->baseMutationRate * (1.0f + params->operatorAdaptationRate * params->stagnantGenerationCounter);
        
        // Cap mutation rate at reasonable level
        if (*mutationRate > 0.5f) {
            *mutationRate = 0.5f;
        }
    } else {
        *mutationRate = params->baseMutationRate;
    }
    
    // Adjust crossover rate based on diversity
    if (params->populationDiversity < 0.3f) {
        // Lower diversity, increase crossover to mix genes more
        *crossoverRate = params->baseCrossoverRate + 
                       (1.0f - params->baseCrossoverRate) * (1.0f - params->populationDiversity);
    } else {
        *crossoverRate = params->baseCrossoverRate;
    }
}

// Update control parameters after generation
void updateControlParameters(ControlParameters* params, float newBestFitness, 
                            float avgFitness, float fitnessVariance) {

    params->generationsCompleted++;
    
    // Check for improvement
    if (newBestFitness > params->currentBestFitness + 0.0001f) {
        // Progress made
        params->stagnantGenerationCounter = 0;
    } else {
        // Stagnation
        params->stagnantGenerationCounter++;
    }
    
    // Update fitness tracking
    params->previousBestFitness = params->currentBestFitness;
    params->currentBestFitness = newBestFitness;
    
    // Update moving average with exponential weighting
    float alpha = 2.0f / (params->fitnessHistoryWindow + 1);
    params->fitnessMovingAverage = alpha * avgFitness + (1 - alpha) * params->fitnessMovingAverage;
    params->fitnessVariance = fitnessVariance;
    
    // Dynamic population sizing logic
    if (shouldIncreasePopulationSize(params)) {
        int newPopSize = (int)(params->currentPopSize * params->popSizeGrowthFactor);
        if (newPopSize > params->maxPopSize) {
            newPopSize = params->maxPopSize;
        }
        
        if (newPopSize > params->currentPopSize) {
            printf("Generation %d: Increasing population size from %d to %d\n", 
                   params->generationsCompleted, params->currentPopSize, newPopSize);
            
            params->currentPopSize = newPopSize;
            params->stagnantGenerationCounter = 0; // Reset stagnation counter after adaptation
        }
    }
    
    // Record adaptation time
    params->lastAdaptationTime = clock() / (double)CLOCKS_PER_SEC;
}

// Termination check
bool checkTermination(ControlParameters* params) {
    // Check if we've reached maximum generations
    if (params->generationsCompleted >= params->maxGenerations) {
        printf("Terminating: Reached maximum number of generations (%d)\n", params->maxGenerations);
        return true;
    }
    
    // Check for stagnation
    if (params->stagnantGenerationCounter >= params->maxStagnantGenerations && 
        params->currentPopSize >= params->maxPopSize) {
        printf("Terminating: Stagnation detected with maximum population size reached\n");
        return true;
    }
    
    // Check for fitness convergence
    if (params->currentBestFitness >= params->convergenceThreshold) {
        printf("Terminating: Reached fitness convergence threshold (%.4f)\n", params->convergenceThreshold);
        return true;
    }
    
    return false;
}

void reallocateDeviceMemory(float** d_population, float** d_offspring, float** d_fitness, 
                            curandState** d_randStates, int oldPopSize, int newPopSize, int individualSize) {
    
    // Allocate new device memory
    float* new_d_population;
    float* new_d_offspring;
    float* new_d_fitness;
    curandState* new_d_randStates;

    cudaMalloc(&new_d_population, newPopSize * individualSize * sizeof(float));
    cudaMalloc(&new_d_offspring, newPopSize * individualSize * sizeof(float));
    cudaMalloc(&new_d_fitness, newPopSize * sizeof(float));
    cudaMalloc(&new_d_randStates, newPopSize * sizeof(curandState));

    // Copy old data to new memory
    cudaMemcpy(new_d_population, *d_population, oldPopSize * individualSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_d_offspring, *d_offspring, oldPopSize * individualSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_d_fitness, *d_fitness, oldPopSize * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(new_d_randStates, *d_randStates, oldPopSize * sizeof(curandState), cudaMemcpyDeviceToDevice);

    // Free old memory
    cudaFree(*d_population);
    cudaFree(*d_offspring);
    cudaFree(*d_fitness);
    cudaFree(*d_randStates);
    
    // Update pointers
    *d_population = new_d_population;
    *d_offspring = new_d_offspring;
    *d_fitness = new_d_fitness;
    *d_randStates = new_d_randStates;
    
    // Re-initialize random states for new individuals
    int blockSize = 256;
    int gridSize = (newPopSize - oldPopSize + blockSize - 1) / blockSize;
    
    initializeRandomStatesKernel<<<gridSize, blockSize>>>(*d_randStates + oldPopSize, time(NULL));
}

// Main AUTO+PDMS evolutionary function for KDD Cup dataset
void evolveWithAutomaticControl(Dataset trainDataset, Dataset testDataset) {
    // Initialize control parameters
    ControlParameters params = initializeControlParameters();
    srand(time(NULL));
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    // Size calculations
    int individualSize = trainDataset.numFeatures * 2;
    int popSize = params.currentPopSize;
    int maxPopSize = params.maxPopSize;
    
    // Allocate host memory
    float* h_population = (float*)malloc(maxPopSize * individualSize * sizeof(float));
    float* h_offspring = (float*)malloc(maxPopSize * individualSize * sizeof(float));
    float* h_fitness = (float*)malloc(maxPopSize * sizeof(float));
    
    // Create initial population
    createInitialPopulation(h_population, popSize, trainDataset.numFeatures);
    
    // Allocate device memory
    float* d_population;
    float* d_offspring;
    float* d_fitness;
    float* d_trainFeatures;
    int* d_trainLabels;
    float* d_testFeatures;
    int* d_testLabels;
    curandState* d_randStates;
    
    cudaMalloc(&d_population, maxPopSize * individualSize * sizeof(float));
    cudaMalloc(&d_offspring, maxPopSize * individualSize * sizeof(float));
    cudaMalloc(&d_fitness, maxPopSize * sizeof(float));
    cudaMalloc(&d_trainFeatures, trainDataset.numSamples * trainDataset.numFeatures * sizeof(float));
    cudaMalloc(&d_trainLabels, trainDataset.numSamples * sizeof(int));
    cudaMalloc(&d_testFeatures, testDataset.numSamples * testDataset.numFeatures * sizeof(float));
    cudaMalloc(&d_testLabels, testDataset.numSamples * sizeof(int));
    cudaMalloc(&d_randStates, maxPopSize * sizeof(curandState));
    
    // Copy data to device
    cudaMemcpy(d_trainFeatures, trainDataset.features, 
               trainDataset.numSamples * trainDataset.numFeatures * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_trainLabels, trainDataset.labels, 
               trainDataset.numSamples * sizeof(int), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_testFeatures, testDataset.features, 
               testDataset.numSamples * testDataset.numFeatures * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_testLabels, testDataset.labels, 
               testDataset.numSamples * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    // Initialize random states
    int blockSize = 256;
    int gridSize = (popSize + blockSize - 1) / blockSize;
    
    initializeRandomStatesKernel<<<gridSize, blockSize>>>(d_randStates, time(NULL));
    
    printf("Starting AUTO+PDMS evolution for KDD Cup dataset\n");
    printf("Initial population size: %d\n", popSize);
    printf("Training samples: %d, Test samples: %d\n", trainDataset.numSamples, testDataset.numSamples);
    printf("Features: %d, Classes: %d\n", trainDataset.numFeatures, trainDataset.numClasses);
    
    bool terminate = false;
    
    // For tracking best individual on test set
    float bestTestAccuracy = 0.0f;
    float* h_testFitness = (float*)malloc(1 * sizeof(float)); // Only need to copy one individual's fitness
    float* d_testFitness;
    cudaMalloc(&d_testFitness, 1 * sizeof(float));
    
    // Main evolutionary loop
    while (!terminate) {
        // If population size increased, we need to reallocate and copy
        if (params.currentPopSize > popSize) {
            int oldPopSize = popSize;
            popSize = params.currentPopSize;
            reallocateDeviceMemory(&d_population, &d_offspring, &d_fitness, &d_randStates, oldPopSize, popSize, individualSize);
            
            // Create new individuals for expanded population
            createInitialPopulation(h_population + oldPopSize * individualSize, 
                                   popSize - oldPopSize, trainDataset.numFeatures);
        }

        // Adjust current grid size based on population size
        gridSize = (popSize + blockSize - 1) / blockSize;
        
        // Copy current population to device
        cudaMemcpy(d_population, h_population, popSize * individualSize * sizeof(float), 
                  cudaMemcpyHostToDevice);
        
        // Evaluate fitness on training set
        float accuracyWeight = 0.8f;
        float complexityWeight = 0.2f;
        
        evaluateFitnessKernel<<<gridSize, blockSize>>>(
            d_population, d_fitness, 
            d_trainFeatures, d_trainLabels,
            popSize, trainDataset.numSamples, trainDataset.numFeatures, trainDataset.numClasses,
            accuracyWeight, complexityWeight);
        
        // Copy fitness back to host
        cudaMemcpy(h_fitness, d_fitness, popSize * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Calculate fitness statistics
        float totalFitness = 0.0f;
        float bestFitness = -INFINITY;
        int bestIndex = -1;
        
        for (int i = 0; i < popSize; i++) {
            totalFitness += h_fitness[i];
            if (h_fitness[i] > bestFitness) {
                bestFitness = h_fitness[i];
                bestIndex = i;
            }
        }
        
        float avgFitness = totalFitness / popSize;
        
        // Calculate fitness variance
        float sumSquaredDiff = 0.0f;
        for (int i = 0; i < popSize; i++) {
            float diff = h_fitness[i] - avgFitness;
            sumSquaredDiff += diff * diff;
        }
        float fitnessVariance = sumSquaredDiff / popSize;
        
        // Measure population diversity
        params.populationDiversity = updateDiversity(d_population, popSize, individualSize);
        
        // Every 10 generations, evaluate on test set
        if (params.generationsCompleted % 10 == 0 || checkTermination(&params)) {
            // Evaluate fitness on test set for best individual
            evaluateFitnessKernel<<<1, 1>>>(
                d_population + bestIndex * individualSize, d_testFitness, 
                d_testFeatures, d_testLabels,
                1, testDataset.numSamples, trainDataset.numFeatures, trainDataset.numClasses,
                1.0f, 0.0f);  // No complexity penalty for evaluation
            
            cudaMemcpy(h_testFitness, d_testFitness, sizeof(float), cudaMemcpyDeviceToHost);
            
            if (h_testFitness[0] > bestTestAccuracy) {
                bestTestAccuracy = h_testFitness[0];
                printf("New best test accuracy: %.4f\n", bestTestAccuracy);
            }
        }
        
        // Print statistics
        printf("Generation %d: Best Fitness = %.6f, Avg = %.6f, Diversity = %.6f\n",
               params.generationsCompleted, bestFitness, avgFitness, params.populationDiversity);
        
        // Determine adaptive operator rates
        float mutationRate, crossoverRate;
        adaptOperatorRates(&params, &mutationRate, &crossoverRate);
        
        // Apply genetic operations with adaptive rates
        adaptiveGeneticOperationsKernel<<<gridSize, blockSize>>>(
            d_population, d_offspring, d_fitness,
            popSize, individualSize,
            mutationRate, crossoverRate,
            d_randStates);
        
        // Copy offspring back to host
        cudaMemcpy(h_offspring, d_offspring, popSize * individualSize * sizeof(float), 
                  cudaMemcpyHostToDevice);
        
        // Replace population with offspring
        memcpy(h_population, h_offspring, popSize * individualSize * sizeof(float));
        
        // Update control parameters and check termination
        updateControlParameters(&params, bestFitness, avgFitness, fitnessVariance);
        terminate = checkTermination(&params);
        
        params.totalEvaluations += popSize;
    }
    
    // Print final statistics
    printf("\nEvolution completed\n");
    printf("Generations: %d\n", params.generationsCompleted);
    printf("Total evaluations: %d\n", params.totalEvaluations);
    printf("Final population size: %d\n", popSize);
    printf("Best fitness on training set: %.6f\n", params.currentBestFitness);
    printf("Best accuracy on test set: %.6f\n", bestTestAccuracy);
    
    // Get best individual
    int bestIndividualIndex = 0;
    float bestIndividualFitness = h_fitness[0];
    
    for (int i = 1; i < popSize; i++) {
        if (h_fitness[i] > bestIndividualFitness) {
            bestIndividualFitness = h_fitness[i];
            bestIndividualIndex = i;
        }
    }
    
    // Print best individual details
    printf("\nBest Individual Features:\n");
    for (int j = 0; j < trainDataset.numFeatures; j++) {
        float lowerBound = h_population[bestIndividualIndex * individualSize + j * 2];
        float upperBound = h_population[bestIndividualIndex * individualSize + j * 2 + 1];
        printf("Feature %d: [%.4f, %.4f]\n", j, lowerBound, upperBound);
    }
    
    // Clean up
    free(h_population);
    free(h_offspring);
    free(h_fitness);
    free(h_testFitness);
    
    cudaFree(d_population);
    cudaFree(d_offspring);
    cudaFree(d_fitness);
    cudaFree(d_trainFeatures);
    cudaFree(d_trainLabels);
    cudaFree(d_testFeatures);
    cudaFree(d_testLabels);
    cudaFree(d_testFitness);
    cudaFree(d_randStates);
}

// Main function
int main() {
    // Load datasets
    printf("Loading KDD Cup training dataset...\n");
    Dataset trainDataset = loadKDDCupDataset("kdd_cup_train.csv");
    
    if (trainDataset.numSamples == 0) {
        printf("Failed to load training dataset. Exiting.\n");
        return 1;
    }
    
    printf("Loading KDD Cup test dataset...\n");
    Dataset testDataset = loadKDDCupDataset("kdd_cup_test.csv");
    
    if (testDataset.numSamples == 0) {
        printf("Failed to load test dataset. Exiting.\n");
        freeDataset(&trainDataset);
        return 1;
    }
    
    // Normalize datasets
    printf("Normalizing datasets...\n");
    normalizeDataset(&trainDataset);
    normalizeDataset(&testDataset);
    
    // Run evolution with automatic parameter control
    evolveWithAutomaticControl(trainDataset, testDataset);
    
    // Clean up
    freeDataset(&trainDataset);
    freeDataset(&testDataset);
    
    return 0;
}