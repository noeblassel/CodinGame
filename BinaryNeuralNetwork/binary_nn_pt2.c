/*
Solution to Binary Neural Network- part 2:https://www.codingame.com/ide/puzzle/binary-neural-network---part-2
This reuses the code in part 1, adding dynamically defined activation functions.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#define errlog(args...) fprintf(stderr, args)

typedef unsigned char byte;

#define SIGMOID 0
#define RELU 1
#define TANH 2
#define LINEAR 3
#define LK_RELU 4

typedef struct neuron_t
{
    float weighted_input_value;
    float activation_value;
    float delta;
} Neuron;

typedef float (*neural_function_t)(Neuron *);

typedef struct layer_t
{
    int dim;
    struct layer_t *prev;
    struct layer_t *next;
    Neuron *units;

    float **weights;
    float *bias;

    float **weights_grad;
    float *bias_grad;

    neural_function_t activation;
    neural_function_t dactivation;
} Layer;

typedef float (*loss_function_t)(Layer *, float *);

typedef struct mlp_t
{
    int n_layers;
    int *layer_dims;
    Layer *layers;
    int *activation_types;
} MultiLayerPerceptron;

float eta = .5;
long r_state = 1103527590;

long lcg();
float randf(float min, float max);

float sigmoid(Neuron *unit);
float dsigmoid(Neuron *unit);
float relu(Neuron *unit);
float drelu(Neuron *unit);
float tanh_neural(Neuron *unit);
float dtanh_neural(Neuron *unit);
float id(Neuron *unit);
float did(Neuron *unit);

void setup(MultiLayerPerceptron *net, int n_layers, int *layer_dimensions, int *activation_types);
void initialize_weights(MultiLayerPerceptron *net);

void forward(MultiLayerPerceptron *net, float *input);
void backward(MultiLayerPerceptron *net, float *input, float *output, bool recompute_activations);

void gradient_step(MultiLayerPerceptron *net);
void predict(MultiLayerPerceptron *net, float *input);
void cleanup(MultiLayerPerceptron *net);

int main()
{
    MultiLayerPerceptron net;
    int layer_dims[3] = {8, 9, 8};
    int activation_funcs[3] = {LINEAR, TANH, SIGMOID};
    setup(&net, 3, layer_dims, activation_funcs);
    initialize_weights(&net);

    int n_tests, n_trains;
    scanf("%d%d", &n_tests, &n_trains);
    float **test_X = malloc(n_tests * sizeof(float *));
    float **train_X = malloc(n_trains * sizeof(float *));
    float **train_Y = malloc(n_trains * sizeof(float *));

    char x[9];
    char y[9];
    for (int i = 0; i < n_tests; ++i)
    {
        scanf("%s", x);
        test_X[i] = (float *)malloc(net.layer_dims[0] * sizeof(float));
        for (int j = 0; j < 8; j++)
        {
            test_X[i][j] = (float)(x[j] - '0');
        }
    }

    for (int i = 0; i < n_trains; ++i)
    {
        scanf("%s%s", x, y);
        train_X[i] = malloc(net.layer_dims[0] * sizeof(float));
        train_Y[i] = malloc(net.layer_dims[net.n_layers - 1] * sizeof(float));
        for (int j = 0; j < 8; j++)
        {
            train_X[i][j] = (float)(x[j] - '0');
            train_Y[i][j] = (float)(y[j] - '0');
        }
    }
    for (int i = 0; i < 800; ++i)
    {
        for (int j = 0; j < n_trains; j++)
        {
            backward(&net, train_X[j], train_Y[j], 1);
            gradient_step(&net);
        }
    }
    for (int j = 0; j < n_tests; j++)
    {
        predict(&net, test_X[j]);
    }
    cleanup(&net);

    for (int i = 0; i < n_tests; ++i)
        free(test_X[i]);
    free(test_X);
    for (int i = 0; i < n_trains; ++i)
        free(train_X[i]), free(train_Y[i]);
    free(train_X), free(train_Y);
    return 0;
}

long lcg()
{
    r_state = (0x41c64e6d * r_state + 0x3039) % 0x80000000;
    return r_state;
}
float randf(float min, float max)
{
    return min + (max - min) * (((float)lcg()) / 0x7fffffff);
}
float sigmoid(Neuron *unit)
{
    if (unit->weighted_input_value > 0)
    {
        return 1. / (1 + (float)exp(-unit->weighted_input_value));
    }
    else
    {
        float s = (float)exp(unit->weighted_input_value);
        return s / (1 + s);
    }
}
float dsigmoid(Neuron *unit)
{
    return (unit->activation_value) * (1 - unit->activation_value);
}

float relu(Neuron *unit)
{
    return (unit->weighted_input_value > 0) ? unit->weighted_input_value : 0.0;
}
float drelu(Neuron *unit)
{
    return (float)(unit->weighted_input_value > 0);
}

float tanh_neural(Neuron *unit)
{
    return tanhf(unit->weighted_input_value);
}
float dtanh_neural(Neuron *unit)
{
    return 1 - (unit->activation_value) * (unit->activation_value);
}
float id(Neuron *unit)
{
    return unit->weighted_input_value;
}
float did(Neuron *unit)
{
    return 1.0;
}
float lk_relu(Neuron *unit){
    return (unit->weighted_input_value>0)?unit->weighted_input_value:0.01*unit->weighted_input_value;
}
float dlk_relu(Neuron *unit){
    return (unit->weighted_input_value>0)?1:0.01;
}

void setup(MultiLayerPerceptron *net, int n_layers, int *layer_dimensions, int *activation_types)
{
    neural_function_t activation_functions[5] = {sigmoid, relu, tanh_neural, id,lk_relu};
    neural_function_t derivatives[5]={dsigmoid,drelu,dtanh_neural,did,dlk_relu};

    net->n_layers = n_layers;
    net->layers = (Layer *)malloc(net->n_layers * sizeof(Layer));
    net->layer_dims = (int *)malloc((net->n_layers) * sizeof(int));
    net->activation_types = (int *)malloc((net->n_layers) * sizeof(int));

    for (int i = 0; i <= net->n_layers; ++i)
    {
        net->layer_dims[i] = layer_dimensions[i];
    }
    for (int i = 0; i < net->n_layers; ++i)
    {
        net->activation_types[i] = activation_types[i];
    }

    for (int i = 0; i < net->n_layers; ++i)
    {
        net->layers[i].units = (Neuron *)malloc(net->layer_dims[i] * sizeof(Neuron));
        if (i > 0)
        {
            net->layers[i].prev = net->layers + i - 1;
            net->layers[i].weights = (float **)malloc(net->layer_dims[i - 1] * sizeof(float *));
            net->layers[i].weights_grad = (float **)malloc(net->layer_dims[i - 1] * sizeof(float *));

            for (int j = 0; j < net->layer_dims[i - 1]; j++)
            {
                net->layers[i].weights[j] = (float *)malloc(net->layer_dims[i] * sizeof(float));
                net->layers[i].weights_grad[j] = (float *)malloc(net->layer_dims[i] * sizeof(float));
            }

            net->layers[i].bias = (float *)malloc(net->layer_dims[i] * sizeof(float));
            net->layers[i].bias_grad = (float *)malloc(net->layer_dims[i] * sizeof(float));
        }
        net->layers[i].activation = activation_functions[net->activation_types[i]];
        net->layers[i].dactivation= derivatives[net->activation_types[i]];
        if (i < net->n_layers - 1)
            net->layers[i].next = net->layers + i + 1;
    }
}

void cleanup(MultiLayerPerceptron *net)
{
    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int j = 0; j < net->layer_dims[i - 1]; j++)
        {
            free(net->layers[i].weights[j]);
            free(net->layers[i].weights_grad[j]);
        }
        free(net->layers[i].weights);
        free(net->layers[i].weights_grad);

        free(net->layers[i].bias);
        free(net->layers[i].bias_grad);

        free(net->layers[i].units);
    }

    free(net->layers[0].units);
    free(net->layers);
    free(net->layer_dims);
    free(net->activation_types);
}

void initialize_weights(MultiLayerPerceptron *net)
{
    for (int i = 1; i < net->n_layers; ++i)
    {
        float alpha = (float)sqrt(3. / ((float)net->layer_dims[i - 1]));
        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            for (int j = 0; j < net->layer_dims[i - 1]; j++)
            {
                net->layers[i].weights[j][k] = randf(-alpha, alpha);
            }
            net->layers[i].bias[k] = 0;
        }
    }
}

void forward(MultiLayerPerceptron *net, float *input)
{
    for (int k = 0; k < net->layer_dims[0]; k++)
    {
        net->layers[0].units[k].weighted_input_value = input[k];
        net->layers[0].units[k].activation_value = (*(net->layers[0].activation))(&(net->layers[0].units[k]));
    }
    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            net->layers[i].units[k].weighted_input_value = 0;
            for (int j = 0; j < net->layer_dims[i - 1]; j++)
            {
                net->layers[i].units[k].weighted_input_value += (net->layers[i].weights[j][k]) * (net->layers[i - 1].units[j].activation_value);
            }
            net->layers[i].units[k].weighted_input_value += net->layers[i].bias[k];
            net->layers[i].units[k].activation_value = (*(net->layers[i].activation))(&(net->layers[i].units[k]));
        }
    }
}

void backward(MultiLayerPerceptron *net, float *input, float *output, bool recompute_activations)
{
    if (recompute_activations)
        forward(net, input);
    for (int k = 0; k < net->layer_dims[net->n_layers - 1]; ++k)
    {
        net->layers[net->n_layers - 1].units[k].delta = (*(net->layers[net->n_layers-1].dactivation))(&(net->layers[net->n_layers - 1].units[k])) * (net->layers[net->n_layers - 1].units[k].activation_value - output[k]);
    }

    for (int i = net->n_layers - 2; i > 0; i--)
    {
        for (int j = 0; j < net->layer_dims[i]; j++)
        {
            float _delta = 0;
            for (int k = 0; k < net->layer_dims[i + 1]; k++)
            {
                _delta += (net->layers[i + 1].units[k].delta) * (net->layers[i + 1].weights[j][k]);
            }
            _delta *= (*(net->layers[i].dactivation))(&(net->layers[i].units[j]));
            net->layers[i].units[j].delta = _delta;
        }
    }

    for (int i = net->n_layers - 1; i > 0; i--)
    {
        for (int j = 0; j < net->layer_dims[i - 1]; j++)
        {
            for (int k = 0; k < net->layer_dims[i]; k++)
            {
                net->layers[i].weights_grad[j][k] = (net->layers[i - 1].units[j].activation_value) * (net->layers[i].units[k].delta);
            }
        }

        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            net->layers[i].bias_grad[k] = net->layers[i].units[k].delta;
        }
    }
}

void gradient_step(MultiLayerPerceptron *net)
{
    for (int i = 1; i < net->n_layers; i++)
    {
        for (int j = 0; j < net->layer_dims[i - 1]; j++)
        {
            for (int k = 0; k < net->layer_dims[i]; k++)
            {
                net->layers[i].weights[j][k] -= eta * net->layers[i].weights_grad[j][k];
            }
        }
        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            net->layers[i].bias[k] -= eta * net->layers[i].bias_grad[k];
        }
    }
}

void predict(MultiLayerPerceptron *net, float *input)
{
    forward(net, input);
    for (int i = 0; i < net->layer_dims[net->n_layers - 1]; ++i)
    {
        printf("%d", (net->layers[net->n_layers - 1].units[i].activation_value > .5) ? 1 : 0);
    }
    printf("\n");
}
