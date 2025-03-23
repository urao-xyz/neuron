#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm> // Pour std::max

// Fonctions d'activation
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

double relu(double x) {
    return std::max(0.0, x);
}

double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double tanh(double x) {
    return std::tanh(x);
}

double tanhDerivative(double x) {
    return 1.0 - x * x;
}

double leakyRelu(double x) {
    return x > 0 ? x : 0.01 * x;
}

double leakyReluDerivative(double x) {
    return x > 0 ? 1.0 : 0.01;
}

// Initialisation Xavier/Glorot
double xavierInitialization(int numInputs) {
    double range = sqrt(6.0 / (numInputs + 1));
    return ((double)rand() / RAND_MAX) * 2 * range - range;
}

// Classe pour représenter un neurone
class Neuron {
public:
    std::vector<double> weights;
    double bias;
    double output;
    double delta;

    Neuron(int numInputs) {
        for (int i = 0; i < numInputs; ++i) {
            weights.push_back(xavierInitialization(numInputs));
        }
        bias = xavierInitialization(numInputs);
    }

    double activate(const std::vector<double>& inputs, const std::string& activationFunc = "sigmoid") {
        double activation = bias;
        for (size_t i = 0; i < inputs.size(); ++i) {
            activation += inputs[i] * weights[i];
        }
        if (activationFunc == "relu") {
            output = relu(activation);
        } else if (activationFunc == "tanh") {
            output = tanh(activation);
        } else if (activationFunc == "leakyRelu") {
            output = leakyRelu(activation);
        } else {
            output = sigmoid(activation); // Par défaut, sigmoïde
        }
        return output;
    }
};

// Classe pour représenter une couche de neurones
class Layer {
public:
    std::vector<Neuron> neurons;
    std::string activationFunc;

    Layer(int numNeurons, int numInputsPerNeuron, const std::string& activationFunc = "relu")
        : activationFunc(activationFunc) {
        for (int i = 0; i < numNeurons; ++i) {
            neurons.push_back(Neuron(numInputsPerNeuron));
        }
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs, activationFunc));
        }
        return outputs;
    }
};

// Classe pour représenter un réseau de neurones
class NeuralNetwork {
public:
    std::vector<Layer> layers;
    double lambda; // Paramètre de régularisation L2

    NeuralNetwork(const std::vector<int>& topology, double lambda = 0.001) : lambda(lambda) {
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            std::string activationFunc = (i == topology.size() - 2) ? "sigmoid" : "relu"; // Couche de sortie : sigmoïde
            layers.push_back(Layer(topology[i + 1], topology[i], activationFunc));
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
            double totalLoss = 0.0;
            for (size_t i = 0; i < trainingInputs.size(); ++i) {
                // Forward pass
                std::vector<double> outputs = predict(trainingInputs[i]);

                // Calcul de la perte
                double loss = calculateLoss(outputs, trainingOutputs[i]);
                totalLoss += loss;

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
                            if (layer.activationFunc == "relu") {
                                neuron.delta = error * reluDerivative(neuron.output);
                            } else if (layer.activationFunc == "tanh") {
                                neuron.delta = error * tanhDerivative(neuron.output);
                            } else if (layer.activationFunc == "leakyRelu") {
                                neuron.delta = error * leakyReluDerivative(neuron.output);
                            } else {
                                neuron.delta = error * sigmoidDerivative(neuron.output);
                            }
                        }
                    }
                }

                // Mise à jour des poids et des biais avec régularisation L2
                for (auto& layer : layers) {
                    for (auto& neuron : layer.neurons) {
                        for (size_t k = 0; k < neuron.weights.size(); ++k) {
                            neuron.weights[k] += learningRate * (neuron.delta * (k < trainingInputs[i].size() ? trainingInputs[i][k] : 1.0) - lambda * neuron.weights[k]);
                        }
                        neuron.bias += learningRate * neuron.delta;
                    }
                }
            }
            std::cout << "Epoch " << epoch << ", Loss: " << totalLoss / trainingInputs.size() << std::endl;
        }
    }

    double calculateLoss(const std::vector<double>& outputs, const std::vector<double>& targets) {
        double loss = 0.0;
        for (size_t i = 0; i < outputs.size(); ++i) {
            loss += 0.5 * pow(targets[i] - outputs[i], 2);
        }
        return loss / outputs.size();
    }
};

int main() {
    srand(time(0));

    // Topologie du réseau : 2 entrées, 2 couches cachées de 4 neurones, 1 sortie
    NeuralNetwork nn({2, 4, 4, 1}, 0.001); // Régularisation L2 avec lambda = 0.001

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