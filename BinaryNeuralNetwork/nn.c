#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#define errlog(args...) fprintf(stderr, args)

typedef unsigned char byte;
//modifying these values requires updating some arrays in function setup()
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
static char z85_charset[86] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
static byte z85_decoding_table[93] = { //z85_charset has ascii codes between 33 and 125, so we offset chars by 33, and only map bytes up to 125
    0x44, 0x00, 0x54, 0x53, 0x52, 0x48, 0x00,
    0x4B, 0x4C, 0x46, 0x41, 0x00, 0x3F, 0x3E, 0x45,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x40, 0x00, 0x49, 0x42, 0x4A, 0x47,
    0x51, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A,
    0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32,
    0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A,
    0x3B, 0x3C, 0x3D, 0x4D, 0x00, 0x4E, 0x43, 0x00,
    0x00, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
    0x21, 0x22, 0x23, 0x4F, 0x00, 0x50};

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

void read_from_z85(MultiLayerPerceptron *net, char *z85_string);
char *write_to_z85(MultiLayerPerceptron *net);
void save_to_file(MultiLayerPerceptron *net, char *filename);
void load_from_file(MultiLayerPerceptron *net, char *filename);

void cleanup(MultiLayerPerceptron *net);
char *z85_encode(byte *src, size_t len);
byte *z85_decode(char *src);

neural_function_t derivative[5] = {dsigmoid, drelu, dtanh_neural, did};

int main()
{
    MultiLayerPerceptron net;
    int layer_dims[5] = {8, 64, 32, 64, 8};
    int activation_funcs[5] = {LINEAR, SIGMOID, SIGMOID, RELU, RELU};
    setup(&net, 4, layer_dims, activation_funcs);
    initialize_weights(&net);
    save_to_file(&net, "a");
    load_from_file(&net, "a");
    save_to_file(&net, "b");
    cleanup(&net);
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
float lk_relu(Neuron *unit)
{
    return (unit->weighted_input_value) * (0.01 + 0.99 * (unit->weighted_input_value > 0));
}
float dlk_relu(Neuron *unit)
{
    return 0.01 + 0.99 * (unit->weighted_input_value > 0);
}

void setup(MultiLayerPerceptron *net, int n_layers, int *layer_dimensions, int *activation_types)
{
    neural_function_t activation_functions[5] = {sigmoid, relu, tanh_neural, id, lk_relu};
    neural_function_t derivatives[5] = {dsigmoid, drelu, dtanh_neural, did, dlk_relu};

    net->n_layers = n_layers;
    net->layers = (Layer *)malloc(net->n_layers * sizeof(Layer));
    net->layer_dims = (int *)malloc((net->n_layers) * sizeof(int));
    net->activation_types = (int *)malloc((net->n_layers) * sizeof(int));

    memcpy(&(net->layer_dims),layer_dimensions,n_layers*sizeof(int));
    memcpy(&(net->activation_types),activation_types,n_layers*sizeof(int));

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
        net->layers[i].dactivation = derivatives[net->activation_types[i]];
        if (i < net->n_layers - 1)
            net->layers[i].next = net->layers + i + 1;
    }
}

void copy(MultiLayerPerceptron *net_from,MultiLayerPerceptron *net_to){
    
    setup(net_to,net_from->n_layers,net_from->layer_dims,net_from->activation_types);
    
    for(int i=1;i<net_from->n_layers;++i){
        for(int k=0;k<net_from->layer_dims[i];k++){
            for(int j=0;j<net_from->layer_dims[i-1];j++){
                net_to->layers[i].weights[j][k]=net_from->layers[i].weights[j][k];
            }
            net_to->layers[i].bias[k]=net_from->layers[i].bias[k];
        }
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
        net->layers[net->n_layers - 1].units[k].delta = (*(net->layers[net->n_layers - 1].dactivation))(&(net->layers[net->n_layers - 1].units[k])) * (net->layers[net->n_layers - 1].units[k].activation_value - output[k]);
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

void save_to_file(MultiLayerPerceptron *net, char *filename)
{
    char *encoded_data = write_to_z85(net);
    FILE *fp = fopen(filename, "w");
    fwrite(encoded_data, 1, strlen(encoded_data), fp);
    fclose(fp);
    free(encoded_data);
}

char *write_to_z85(MultiLayerPerceptron *net)
{
    size_t n_bytes = 0;
    n_bytes += sizeof net->n_layers;
    for (int i = 0; i < net->n_layers; ++i)
        n_bytes += sizeof net->layer_dims[i];

    for (int i = 0; i < net->n_layers; ++i)
        n_bytes += sizeof net->activation_types[i];

    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int k = 0; k < net->layer_dims[i]; ++k)
        {
            for (int j = 0; j < net->layer_dims[i - 1]; ++j)
                n_bytes += sizeof net->layers[i].weights[j][k];
            n_bytes += sizeof net->layers[i].bias[k];
        }
    }

    byte *data = malloc(n_bytes);
    byte *copy_ptr = data;

    memcpy(copy_ptr, &(net->n_layers), sizeof net->n_layers);
    copy_ptr += sizeof net->n_layers;

    for (int i = 0; i < net->n_layers; ++i)
    {
        memcpy(copy_ptr, &(net->layer_dims[i]), sizeof net->layer_dims[i]);
        copy_ptr += sizeof net->layer_dims[i];
    }
    for (int i = 0; i < net->n_layers; ++i)
    {
        memcpy(copy_ptr, &(net->activation_types[i]), sizeof net->activation_types[i]);
        copy_ptr += sizeof net->activation_types[i];
    }
    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int k = 0; k < net->layer_dims[i]; ++k)
        {
            for (int j = 0; j < net->layer_dims[i - 1]; ++j)
            {
                memcpy(copy_ptr, &(net->layers[i].weights[j][k]), sizeof net->layers[i].weights[j][k]);
                copy_ptr += sizeof net->layers[i].weights[j][k];
            }
            memcpy(copy_ptr, &(net->layers[i].bias[k]), sizeof net->layers[i].bias[k]);
            copy_ptr += sizeof net->layers[i].bias[k];
        }
    }
    assert(copy_ptr == data + n_bytes);
    char *encoded_data = z85_encode(data, n_bytes);
    free(data);

    return encoded_data;
}
void read_from_z85(MultiLayerPerceptron *net, char *z85_string)
{
    byte *decoded_data = z85_decode(z85_string);
    byte *read_ptr = decoded_data;
    int num_layers;
    memcpy(&num_layers, read_ptr, sizeof(int));
    read_ptr += sizeof(int);
    int *layer_dimensions = (int *)malloc(num_layers * sizeof(int));
    int *activation_types = (int *)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; ++i)
    {
        memcpy(layer_dimensions + i, read_ptr, sizeof(int));
        read_ptr += sizeof(int);
    }
    for (int i = 0; i < num_layers; ++i)
    {
        memcpy(activation_types + i, read_ptr, sizeof(int));
        read_ptr += sizeof(int);
    }
    setup(net, num_layers, layer_dimensions, activation_types);

    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int k = 0; k < net->layer_dims[i]; ++k)
        {
            for (int j = 0; j < net->layer_dims[i - 1]; ++j)
            {
                memcpy(&(net->layers[i].weights[j][k]), read_ptr, sizeof(float));
                read_ptr += sizeof(float);
            }
            memcpy(&(net->layers[i].bias[k]), read_ptr, sizeof(float));
            read_ptr += sizeof(float);
        }
    }
    assert(read_ptr - decoded_data == strlen(z85_string) * 4 / 5);
    free(layer_dimensions);
    free(activation_types);
    free(decoded_data);
}

void load_from_file(MultiLayerPerceptron *net, char *filename)
{
    FILE *fp = fopen(filename, "r");
    fseek(fp, 0, SEEK_END);
    size_t len = ftell(fp);
    char *encoded_data = (char *)malloc((len+1) * sizeof(char));
    fseek(fp,0,SEEK_SET);
    fread(encoded_data, sizeof(char), len, fp);
    encoded_data[len]=0;
    read_from_z85(net, encoded_data);
    free(encoded_data);
    fclose(fp);
}

char *z85_encode(byte *data, size_t len)
{
    if (len % 4)
    {
        errlog("Error in Z85 encoding: data length is not a multiple of 4.\n");
        return NULL;
    }
    size_t num_chars = len * 5 / 4;
    char *encoded_data = (char *)malloc(num_chars + 1);
    uint char_idx = 0;
    uint byte_idx = 0;
    u_int32_t val = 0;
    uint pow;
    while (byte_idx < len)
    {
        for (int i = 0; i < 4; ++i)
            val = val * 256 + data[byte_idx++];
        pow = 52200625;
        for (int i = 0; i < 5; i++)
            encoded_data[char_idx++] = z85_charset[val / pow % 85], pow /= 85;
        val = 0;
    }
    assert(char_idx == num_chars);
    encoded_data[char_idx] = 0;

    return encoded_data;
}

byte *z85_decode(char *encoded_data)
{
    if (strlen(encoded_data) % 5)
    {
        printf("Error in Z85 decoding: encoded string length is not a multiple of 5.\n");
        return NULL;
    }

    size_t num_bytes = strlen(encoded_data) * 4 / 5;
    byte *data = (byte *)malloc(num_bytes);

    uint byte_idx = 0;
    uint char_idx = 0;
    u_int32_t val = 0;
    uint pow;

    while (char_idx < strlen(encoded_data))
    {
        for (int i = 0; i < 5; ++i)
            val = val * 85 + z85_decoding_table[(byte)encoded_data[char_idx++] - 33];
        pow = 16777216;
        for (int i = 0; i < 4; ++i)
            data[byte_idx++] = val / pow % 256, pow /= 256;
        val = 0;
    }
    assert(byte_idx == num_bytes);
    return data;
}
