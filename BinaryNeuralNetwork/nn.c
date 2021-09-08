#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

#define errlog(args...) fprintf(stderr, args)

typedef unsigned char byte;

typedef struct neuron_t
{
    float activation;
    float delta;
} Neuron;

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
} Layer;

typedef struct mlp_t
{
    int n_layers;
    int *layer_dims;
    Layer *layers;
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
float sigmoid(float l);

void setup(MultiLayerPerceptron *net, int n_layers, int *layer_dimensions);
void initialize_weights(MultiLayerPerceptron *net);

void forward(MultiLayerPerceptron *net, float *input);
void backward(MultiLayerPerceptron *net, float *input, float *output, bool recompute_activations);

void gradient_step(MultiLayerPerceptron *net);
void predict(MultiLayerPerceptron *net, float *input);

void save_to_z85(MultiLayerPerceptron *net, char *filename);
void load_from_z85(MultiLayerPerceptron *net, char *z85_string);
void load_from_file(MultiLayerPerceptron *net, char *filename);

void cleanup(MultiLayerPerceptron *net);
char *z85_encode(byte *src,size_t len);
byte *z85_decode(char *src);

int main()
{
    MultiLayerPerceptron net;
    int n_layers=4;
    int layer_dims[4]={8,16,16,8};
    setup(&net,n_layers,layer_dims);
    initialize_weights(&net);
    save_to_z85(&net,"a");
    MultiLayerPerceptron net2;
    load_from_file(&net2,"a");
    save_to_z85(&net2,"b");
    cleanup(&net);
    cleanup(&net2);
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
float sigmoid(float x)
{
    if (x > 0)
    {
        return 1. / (1 + (float)exp(-x));
    }
    else
    {
        float s = (float)exp(x);
        return s / (1 + s);
    }
}
void setup(MultiLayerPerceptron *net, int n_layers, int *layer_dimensions)
{
    net->n_layers = n_layers;
    net->layers = (Layer *)malloc(net->n_layers * sizeof(Layer));
    net->layer_dims = (int *)malloc((net->n_layers) * sizeof(int));

    for (int i = 0; i <= net->n_layers; ++i)
    {
        net->layer_dims[i] = layer_dimensions[i];
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
        net->layers[0].units[k].activation = input[k];
    }
    for (int i = 1; i < net->n_layers; ++i)
    {
        for (int k = 0; k < net->layer_dims[i]; k++)
        {
            float l = 0;
            for (int j = 0; j < net->layer_dims[i - 1]; j++)
            {
                l += (net->layers[i].weights[j][k]) * (net->layers[i - 1].units[j].activation);
            }
            l += net->layers[i].bias[k];
            net->layers[i].units[k].activation = sigmoid(l);
        }
    }
}

void backward(MultiLayerPerceptron *net, float *input, float *output, bool recompute_activations)
{
    if (recompute_activations)
        forward(net, input);
    for (int k = 0; k < net->layer_dims[net->n_layers - 1]; ++k)
    {
        float h = net->layers[net->n_layers - 1].units[k].activation;
        net->layers[net->n_layers - 1].units[k].delta = h * (1 - h) * (h - output[k]);
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
            float h = net->layers[i].units[j].activation;
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
        printf("%d", (net->layers[net->n_layers - 1].units[i].activation > .5) ? 1 : 0);
    }
    printf("\n");
}

void save_to_z85(MultiLayerPerceptron *net, char *filename)
{
    size_t n_bytes = 0;
    n_bytes += sizeof net->n_layers;
    for (int i = 0; i < net->n_layers; ++i)
        n_bytes += sizeof net->layer_dims[i];
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

    FILE *fp = fopen(filename, "w");
    fwrite(encoded_data, 1, strlen(encoded_data), fp);
    fclose(fp);
    free(data);
    free(encoded_data);
}

void load_from_z85(MultiLayerPerceptron *net, char *z85_string)
{
    byte *decoded_data = z85_decode(z85_string);
    byte *read_ptr = decoded_data;
    int num_layers;
    memcpy(&num_layers, read_ptr, sizeof(int));
    read_ptr += sizeof(int);
    int *layer_dimensions = (int *)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; ++i)
    {
        memcpy(layer_dimensions + i, read_ptr, sizeof(int));
        read_ptr += sizeof(int);
    }
    setup(net, num_layers, layer_dimensions);

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
    free(decoded_data);
}

void load_from_file(MultiLayerPerceptron *net, char *filename)
{
    FILE *fp = fopen(filename, "r");
    fseek(fp, 0, SEEK_END);
    size_t len = ftell(fp);
    char *encoded_data = (char *)malloc(len * sizeof(char));
    fread(encoded_data, sizeof(char), len, fp);

    load_from_z85(net, encoded_data);
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
        for(int i=0;i<4;++i)val=val * 256 + data[byte_idx++];
        pow=52200625;
        for(int i=0;i<5;i++)encoded_data[char_idx++]=z85_charset[val/pow%85],pow/=85;
        val=0;
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
        for(int i=0;i<5;++i)val= val * 85 + z85_decoding_table[(byte)encoded_data[char_idx++]-33];
        pow=16777216;
        for(int i=0;i<4;++i)data[byte_idx++]=val/pow%256,pow/=256;
        val=0;
    }
    assert(byte_idx == num_bytes);
    return data;
}
