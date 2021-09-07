#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define log(args...) fprintf(stderr, args)
#define true 1
#define false 0

typedef double real;
typedef char bool;


typedef struct neuron_t
{
    real activation;
    real delta;
} Neuron;

typedef struct layer_t
{
    int dim;
    struct layer_t *prev;
    struct layer_t *next;
    Neuron *units;

    real **weights;
    real *bias;

    real **weights_grad;
    real *bias_grad;
} Layer;

typedef struct mlp_t
{
    int n_layers;
    int *layer_dims;
    Layer *layers;
} MultiLayerPerceptron;

real eta = .5;
long r_state = 1103527590;

int dim_in, dim_out, n_hidden_layers;
int *dims;

int n_epochs;
int n_training_pairs, n_test_inputs;
char **test_inputs;
char ***training_pairs;

MultiLayerPerceptron net;

long lcg();
real randf(real min, real max);
real sigmoid(real l);

void setup();
void initialize_weights();

void forward(char *input);
void backward(char *input, char *output, bool recompute_activations);

void gradient_step();
void predict(char *input);
void cleanup();

int main()
{

    setup();
    initialize_weights();

    for (int i = 0; i < n_epochs; ++i)
    {
        for (int j = 0; j < n_training_pairs; ++j)
        {
            backward(training_pairs[j][0], training_pairs[j][1], true);
            gradient_step();
        }
    }
    for (int i = 0; i < n_test_inputs; i++)
    {
        predict(test_inputs[i]);
    }
    cleanup();
    return 0;
}

long lcg()
{
    r_state = (0x41c64e6d * r_state + 0x3039) % 0x80000000;
    return r_state;
}
real randf(real min, real max)
{
    return min + (max - min) * (((real)lcg()) / 0x7fffffff);
}
real sigmoid(real x)
{
    if (x > 0)
    {
        return 1. / (1 + (real)exp(-x));
    }
    else
    {
        real s = (real)exp(x);
        return s / (1 + s);
    }
}
void setup()
{
    scanf("%d%d%d%d%d%d", &dim_in, &dim_out, &n_hidden_layers, &n_test_inputs, &n_training_pairs, &n_epochs);

    net.n_layers = 2 + n_hidden_layers;
    net.layers = (Layer *)malloc(net.n_layers * sizeof(Layer));
    net.layer_dims = (int *)malloc((2 + n_hidden_layers) * sizeof(int));
    net.layer_dims[0] = dim_in;
    net.layer_dims[1 + n_hidden_layers] = dim_out;

    for (int i = 1; i <= n_hidden_layers; ++i)
    {
        scanf("%d", net.layer_dims + i);
    }

    test_inputs = (char **)malloc(n_test_inputs * sizeof(char *));
    training_pairs = (char ***)malloc(n_training_pairs * sizeof(char **));
    for (int i = 0; i < n_test_inputs; i++)
    {
        test_inputs[i] = (char *)malloc(dim_in * sizeof(char));
        scanf("%s", test_inputs[i]);
    }

    for (int i = 0; i < n_training_pairs; ++i)
    {
        training_pairs[i] = (char **)malloc(2 * sizeof(char *));
        training_pairs[i][0] = (char *)malloc(dim_in * sizeof(char));
        training_pairs[i][1] = (char *)malloc(dim_out * sizeof(char));
        scanf("%s%s", training_pairs[i][0], training_pairs[i][1]);
    }

    for (int i = 0; i < net.n_layers; ++i)
    {
        net.layers[i].units = (Neuron *)malloc(net.layer_dims[i] * sizeof(Neuron));
        if (i > 0)
        {
            net.layers[i].prev = net.layers + i - 1;
            net.layers[i].weights = (real **)malloc(net.layer_dims[i - 1] * sizeof(real *));
            net.layers[i].weights_grad = (real **)malloc(net.layer_dims[i - 1] * sizeof(real *));

            for (int j = 0; j < net.layer_dims[i - 1]; j++)
            {
                net.layers[i].weights[j] = (real *)malloc(net.layer_dims[i] * sizeof(real));
                net.layers[i].weights_grad[j] = (real *)malloc(net.layer_dims[i] * sizeof(real));
            }

            net.layers[i].bias = (real *)malloc(net.layer_dims[i] * sizeof(real));
            net.layers[i].bias_grad = (real *)malloc(net.layer_dims[i] * sizeof(real));
        }
        if (i < net.n_layers - 1)
            net.layers[i].next = net.layers + i + 1;
    }
}

void cleanup()
{
    for (int i = 1; i < net.n_layers; ++i)
    {
        for (int j = 0; j < net.layer_dims[i - 1]; j++)
        {
            free(net.layers[i].weights[j]);
            free(net.layers[i].weights_grad[j]);
        }
        free(net.layers[i].weights);
        free(net.layers[i].weights_grad);

        free(net.layers[i].bias);
        free(net.layers[i].bias_grad);

        free(net.layers[i].units);
    }

    free(net.layers[0].units);
    free(net.layers);

    for (int i = 0; i < n_training_pairs; i++)
    {
        free(training_pairs[i][0]);
        free(training_pairs[i][1]);
        free(training_pairs[i]);
    }
    free(training_pairs);

    for (int i = 0; i < n_test_inputs; i++)
    {
        free(test_inputs[i]);
    }

    free(test_inputs);
}

void initialize_weights()
{
    for (int i = 1; i < net.n_layers; ++i)
    {
        real alpha = (real)sqrt(3. / ((real)net.layer_dims[i - 1]));
        for (int k = 0; k < net.layer_dims[i]; k++)
        {
            for (int j = 0; j < net.layer_dims[i - 1]; j++)
            {
                net.layers[i].weights[j][k] = randf(-alpha, alpha);
            }
            net.layers[i].bias[k] = 0;
        }
    }
}

void forward(char *input)
{
    for (int k = 0; k < net.layer_dims[0]; k++)
    {
        net.layers[0].units[k].activation = (real)(input[k] - '0');
    }
    for (int i = 1; i < net.n_layers; ++i)
    {
        for (int k = 0; k < net.layer_dims[i]; k++)
        {
            real l = 0;
            for (int j = 0; j < net.layer_dims[i - 1]; j++)
            {
                l += (net.layers[i].weights[j][k]) * (net.layers[i - 1].units[j].activation);
            }
            l += net.layers[i].bias[k];
            net.layers[i].units[k].activation = sigmoid(l);
        }
    }
}

void backward(char *input, char *output, bool recompute_activations)
{
    if (recompute_activations)
        forward(input);
    for (int k = 0; k < net.layer_dims[net.n_layers - 1]; ++k)
    {
        real h = net.layers[net.n_layers - 1].units[k].activation;
        net.layers[net.n_layers - 1].units[k].delta = h * (1 - h) * (h - (real)(output[k] - '0'));
    }

    for (int i = net.n_layers - 2; i > 0; i--)
    {
        for (int j = 0; j < net.layer_dims[i]; j++)
        {
            real _delta = 0;
            for (int k = 0; k < net.layer_dims[i + 1]; k++)
            {
                _delta += (net.layers[i + 1].units[k].delta) * (net.layers[i + 1].weights[j][k]);
            }
            real h = net.layers[i].units[j].activation;
            _delta *= (h) * (1 - h);
            net.layers[i].units[j].delta = _delta;
        }
    }

    for (int i = net.n_layers - 1; i > 0; i--)
    {
        for (int j = 0; j < net.layer_dims[i - 1]; j++)
        {
            for (int k = 0; k < net.layer_dims[i]; k++)
            {
                net.layers[i].weights_grad[j][k] = (net.layers[i - 1].units[j].activation) * (net.layers[i].units[k].delta);
            }
        }

        for (int k = 0; k < net.layer_dims[i]; k++)
        {
            net.layers[i].bias_grad[k] = net.layers[i].units[k].delta;
        }
    }
}

void gradient_step()
{
    for (int i = 1; i < net.n_layers; i++)
    {
        for (int j = 0; j < net.layer_dims[i - 1]; j++)
        {
            for (int k = 0; k < net.layer_dims[i]; k++)
            {
                net.layers[i].weights[j][k] -= eta * net.layers[i].weights_grad[j][k];
            }
        }

        for (int k = 0; k < net.layer_dims[i]; k++)
        {
            net.layers[i].bias[k] -= eta * net.layers[i].bias_grad[k];
        }
    }
}

void predict(char *input)
{
    forward(input);
    for (int i = 0; i < net.layer_dims[net.n_layers - 1]; ++i)
    {
        printf("%d", (net.layers[net.n_layers - 1].units[i].activation > .5) ? 1 : 0);
    }
    printf("\n");
}
