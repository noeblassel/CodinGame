#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define log(args...) fprintf(stderr, args)

typedef double real;

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
long lcg();
real randf(real min, real max);
real sigmoid(real l);

void setup(MultiLayerPerceptron* net,int n_layers,int* layer_dimensions);
void init_from_b85(MultiLayerPerceptron* net,char* b85_data);
void save_to_b85(MultiLayerPerceptron* net,char* filename);
void initialize_weights(MultiLayerPerceptron* net);

void forward(MultiLayerPerceptron* net,real *input);
void backward(MultiLayerPerceptron* net,real *input, real *output, bool recompute_activations);

void gradient_step(MultiLayerPerceptron* net);
void predict(MultiLayerPerceptron* net,real *input);
void cleanup(MultiLayerPerceptron* net);

int main()
{

    setup();
    initialize_weights();
    forward(test_inputs[1]);

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
void setup(MultiLayerPerceptron* net,int n_layers,int* layer_dimensions)
{
    net->n_layers = n_layers;
    net->layers = (Layer *)malloc(net->n_layers * sizeof(Layer));
    net->layer_dims = (int *)malloc((net->n_layers) * sizeof(int));

    for (int i = 0; i <= net->n_layers; ++i)
    {
        net->layer_dims[i]=layer_dimensions[i];
    }


    for (int i = 0; i < net->n_layers; ++i)
    {
        net->layers[i].units = (Neuron *)malloc(net->layer_dims[i] * sizeof(Neuron));
        if (i > 0)
        {
            net->layers[i].prev = net->layers + i - 1;
            net->layers[i].weights = (real **)malloc(net->layer_dims[i - 1] * sizeof(real *));
            net->layers[i].weights_grad = (real **)malloc(net->layer_dims[i - 1] * sizeof(real *));

            for (int j = 0; j < net->layer_dims[i - 1]; j++)
            {
                net->layers[i].weights[j] = (real *)malloc(net->layer_dims[i] * sizeof(real));
                net->layers[i].weights_grad[j] = (real *)malloc(net->layer_dims[i] * sizeof(real));
            }

            net->layers[i].bias = (real *)malloc(net->layer_dims[i] * sizeof(real));
            net->layers[i].bias_grad = (real *)malloc(net->layer_dims[i] * sizeof(real));
        }
        if (i < net->n_layers - 1)
            net->layers[i].next = net->layers + i + 1;
    }
}

void cleanup(MultiLayerPerceptron* net)
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
}

void initialize_weights(MultiLayerPerceptron* net)
{
    for (int i = 1; i < net->n_layers; ++i)
    {
        real alpha = (real)sqrt(3. / ((real)net->layer_dims[i - 1]));
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

void forward(MultiLayerPerceptron* net,real* input)
{
    for (int k = 0; k < net->layer_dims[0]; k++)
    {
        net->layers[0].units[k].activation = input[k];
    }
    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            real l = 0;
            for (int j = 0; j < net->layer_dims[i - 1]; j++)
            {
                l += (net->layers[i].weights[j][k]) * (net->layers[i - 1].units[j].activation);
            }
            l += net->layers[i].bias[k];
            net->layers[i].units[k].activation = sigmoid(l);
        }
    }
}

void backward(MultiLayerPerceptron* net, real *input, real *output, bool recompute_activations)
{
    if (recompute_activations)
        forward(net,input);
    for (int k = 0; k < net->layer_dims[net->n_layers - 1]; ++k)
    {
        real h = net->layers[net.n_layers - 1].units[k].activation;
        net->layers[net->n_layers - 1].units[k].delta = h * (1 - h) * (h - output[k]);
    }

    for (int i = net->n_layers - 2; i > 0; i--)
    {
        for (int j = 0; j < net->layer_dims[i]; j++)
        {
            real _delta = 0;
            for (int k = 0; k < net->layer_dims[i + 1]; k++)
            {
                _delta += (net->layers[i + 1].units[k].delta) * (net->layers[i + 1].weights[j][k]);
            }
            real h = net->layers[i].units[j].activation;
            _delta *= (h) * (1 - h);
            net->layers[i].units[j].delta = _delta;
        }
    }

    for (int i = net->n_layers - 1; i > 0; i--)
    {
        for (int j = 0; j < net->layer_dims[i - 1]; j++)
        {
            for (int k = 0; k < net->layer_dims[i]; k++)
            {
                net->layers[i].weights_grad[j][k] = (net->layers[i - 1].units[j].activation) * (net->layers[i].units[k].delta);
            }
        }

        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            net->layers[i].bias_grad[k] = net->layers[i].units[k].delta;
        }
    }
}

void gradient_step(MultiLayerPerceptron* net)
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

void predict(MultiLayerPerceptron* net,char *input)
{
    forward(input);
    for (int i = 0; i < net->layer_dims[net->n_layers - 1]; ++i)
    {
        printf("%d", (net->layers[net->n_layers - 1].units[i].activation > .5) ? 1 : 0);
    }
    printf("\n");
}

void init_from_b85(MultiLayerPerceptron* net,char* b85_data){

}

void save_to_b85(MultiLayerPerceptron* net, char* filename){
    FILE* fp=fopen(filename,"w");

    int n_topology_bytes=net->n_layers;
    for(int i=0;i<net->n_layers;++i){
        n_topology_bytes+=net->layer_dims[i];
    }
    n_topology_bytes*=sizeof(int);

    int n_parameter_bytes=0;
    for(int i=1;i<net->n_layers;++i){
        n_parameter_bytes+=net->layer_dims[i]*(1+net->layer_dims[i-1])//weight matrix + bias vector
    }
    n_parameter_bytes*=sizeof(real);
    n_encoded_bytes=(int)(1.2*(n_parameter_bytes+n_topology_bytes))+1;//base85 encodes 4 binary bytes into 5 ascii bytes

    char* raw_byte_buffer=(char*)malloc(n_topology_bytes+n_parameter_bytes);
    char* encoded_buffer=(char*)malloc(n_encoded_bytes);

    char* reading_ptr=raw_byte_buffer;
    char* writing_ptr=encoded_buffer;

    memcpy(reading_ptr,&(net->n_layers),sizeof(int));
    reading_ptr+=sizeof(int);

    for(int i=0;i<net->n_layers;++i){
        memcpy(reading_ptr,&(net->layer_dims[i]),sizeof(int));
        writing_ptr+=sizeof(int);
    }

    for(int i=1;i<net->n_layers;++i){
        for(int k=0;k<net->layer_dims[i];k++){
            for(int j=0;j<net->layer_dims[i-1];j++){
                memcpy(reading_ptr,&(net->layers[i].weights[j][k]),sizeof(real));
                reading_ptr+=sizeof(real);
            }
            for(int k=0;k<net->layer_dims[i];k++){
                memcpy(reading_ptr,&(net->n_layers),sizeof(real));
                reading_ptr+=sizeof(real);
            }
        }
    }

    

    fwrite(encoded_buffer,1,writing_ptr-encoded_buffer,fp);
    free(raw_byte_buffer);
    free(encoded_buffer);
    fclose(fp);
}