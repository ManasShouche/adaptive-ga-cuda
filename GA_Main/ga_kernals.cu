#include "ga.h"

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
__global__ void evaluateFitnessKernel(float* population, float* fitness, float* data, int* labels, int popSize, int dataSize, int features, int numClasses, float accuracyWeight, float complexityWeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < popSize) {
        float* individual = &population[idx * features * 2];
        
        int correctPredictions = 0;
        float totalRange = 0.0f;
        
        for (int i = 0; i < dataSize; i++) {
            int prediction = -1;
            float maxActivation = -1.0f;
            
            for (int c = 0; c < numClasses; c++) {
                float activation = 0.0f;
                
                for (int j = 0; j < features; j++) {
                    float value = data[i * features + j];
                    float lowerBound = individual[j * 2];
                    float upperBound = individual[j * 2 + 1];
                    
                    if (value >= lowerBound && value <= upperBound) {
                        activation += 1.0f;
                    }
                }
                
                if (activation > maxActivation) {
                    maxActivation = activation;
                    prediction = c;
                }
            }
            
            if (prediction == labels[i]) {
                correctPredictions++;
            }
        }
        
        for (int j = 0; j < features; j++) {
            totalRange += individual[j * 2 + 1] - individual[j * 2];
        }
        
        float complexityPenalty = 0.0f;
        if (features > 0) {
            complexityPenalty = totalRange / features;
        }

        float accuracy = (float)correctPredictions / dataSize;
        fitness[idx] = accuracyWeight * accuracy - complexityWeight * complexityPenalty;
    }
}

// Kernel for adaptive genetic operations
__global__ void adaptiveGeneticOperationsKernel(float* population, float* offspring, float* fitness, int popSize, int individualSize, float mutationRate, float crossoverRate, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= popSize) return;
    
    curandState localState = randStates[idx];
    
    int parent1Idx = 0, parent2Idx = 0;
    
    int candidate1 = curand(&localState) % popSize;
    int candidate2 = curand(&localState) % popSize;
    parent1Idx = (fitness[candidate1] > fitness[candidate2]) ? candidate1 : candidate2;
    
    candidate1 = curand(&localState) % popSize;
    candidate2 = curand(&localState) % popSize;
    parent2Idx = (fitness[candidate1] > fitness[candidate2]) ? candidate1 : candidate2;
    
    for (int i = 0; i < individualSize; i++) {
        offspring[idx * individualSize + i] = population[parent1Idx * individualSize + i];
    }
    
    if (curand_uniform(&localState) < crossoverRate) {
        int crossPoint1 = curand(&localState) % individualSize;
        int crossPoint2 = curand(&localState) % individualSize;
        
        if (crossPoint1 > crossPoint2) {
            int temp = crossPoint1;
            crossPoint1 = crossPoint2;
            crossPoint2 = temp;
        }
        
        for (int i = crossPoint1; i <= crossPoint2; i++) {
            offspring[idx * individualSize + i] = population[parent2Idx * individualSize + i];
        }
    }
    
    for (int i = 0; i < individualSize; i++) {
        if (curand_uniform(&localState) < mutationRate) {
            if (i % 2 == 0) {
                float upperBound = offspring[idx * individualSize + i + 1];
                float mutation = curand_normal(&localState) * 0.1f;
                float newValue = offspring[idx * individualSize + i] + mutation;
                
                if (newValue < 0.0f) newValue = 0.0f;
                if (newValue > upperBound) newValue = upperBound;
                
                offspring[idx * individualSize + i] = newValue;
            } else {
                float lowerBound = offspring[idx * individualSize + i - 1];
                float mutation = curand_normal(&localState) * 0.1f;
                float newValue = offspring[idx * individualSize + i] + mutation;
                
                if (newValue < lowerBound) newValue = lowerBound;
                if (newValue > 1.0f) newValue = 1.0f;
                
                offspring[idx * individualSize + i] = newValue;
            }
        }
    }
    
    randStates[idx] = localState;
}