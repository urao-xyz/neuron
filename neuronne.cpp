#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Fonction d'activation (sigmoïde)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Dérivée de la fonction sigmoïde
double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// Classe pour représenter un neurone
class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;

    Neuron(int numInputs) {
        // Initialisation aléatoire des poids et du biais
        for (int i = 0; i < numInputs; ++i) {
            weights.push_back((double)rand() / RAND_MAX);
        }
        bias = (double)rand() / RAND_MAX;
    }

    double activate(const std::vector<double>& inputs) {
        double activation = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            activation += inputs[i] * weights[i];
        }
        output = sigmoid(activation);
        return output;
    }
};

// Classe pour représenter une couche de neurones
class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int numNeurons, int numInputsPerNeuron) {
        for (int i = 0; i < numNeurons; ++i) {
            neurons.push_back(Neuron(numInputsPerNeuron));
        }
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs));
        }
        return outputs;
    }
};

// Classe pour représenter un réseau de neurones
class NeuralNetwork {
public:
    std::vector<Layer> layers;

    NeuralNetwork(const std::vector<int>& topology) {
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            layers.push_back(Layer(topology[i + 1], topology[i]));
        }
    }

    std::vector<double> predict(const std::vector<double>& inputs) {
        std::vector<double> outputs = inputs;
        for (auto& layer : layers) {
            outputs = layer.feedForward(outputs);
        }
        return outputs;
    }

    void train(const std::vector<std::vector<double>>& trainingInputs, const std::vector<std::vector<double>>& trainingOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < trainingInputs.size(); ++i) {
                // Forward pass
                std::vector<double> outputs = predict(trainingInputs[i]);

                // Backpropagation
                for (size_t j = 0; j < layers.size(); ++j) {
                    auto& layer = layers[layers.size() - 1 - j];
                    for (size_t k = 0; k < layer.neurons.size(); ++k) {
                        auto& neuron = layer.neurons[k];
                        if (j == 0) {
                            // Couche de sortie
                            neuron.delta = (trainingOutputs[i][k] - outputs[k]) * sigmoidDerivative(outputs[k]);
                        } else {
                            // Couches cachées
                            double error = 0.0;
                            for (auto& nextNeuron : layers[layers.size() - j].neurons) {
                                error += nextNeuron.weights[k] * nextNeuron.delta;
                            }
                            neuron.delta = error * sigmoidDerivative(neuron.output);
                        }
                    }
                }

                // Mise à jour des poids et des biais
                for (auto& layer : layers) {
                    for (auto& neuron : layer.neurons) {
                        for (size_t k = 0; k < neuron.weights.size(); ++k) {
                            neuron.weights[k] += learningRate * neuron.delta * (k < trainingInputs[i].size() ? trainingInputs[i][k] : 1.0);
                        }
                        neuron.bias += learningRate * neuron.delta;
                    }
                }
            }
        }
    }
};

int main() {
    srand(time(0));

    // Topologie du réseau : 2 entrées, 2 neurones dans la couche cachée, 1 sortie
    NeuralNetwork nn({2, 2, 1});

    // Données d'entraînement pour un problème XOR simple
    std::vector<std::vector<double>> trainingInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> trainingOutputs = {{0}, {1}, {1}, {0}};

    // Entraînement du réseau
    nn.train(trainingInputs, trainingOutputs, 10000, 0.1);

    // Test du réseau
    for (const auto& input : trainingInputs) {
        std::vector<double> output = nn.predict(input);
        std::cout << "Input: " << input[0] << ", " << input[1] << " -> Output: " << output[0] << std::endl;
    }

    return 0;
}


// Résultat attendu

// Input: 0, 0 -> Output: 0.0188358
// Input: 0, 1 -> Output: 0.982445
// Input: 1, 0 -> Output: 0.387763
// Input: 1, 1 -> Output: 0.396075

// Attendu : 
// 0, 0 : 0
// 0, 1 : 1
// 1, 0 : 1
// 1, 1 : 0
