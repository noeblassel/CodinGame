#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define log(args,...) fprintf(stderr,args)
/*THIS PROGRAM IMPLEMENTS A SIMPLE FEEDFORWARD MULTI-LAYERED NEURAL NETWORK ARCHITECTURE
THE NON LINEARITY IS SIGMOID, AND THE LOSS IS SQUARED ERROR.
NETWORK PARAMETERS ARE UPDATED USING SIMPLE ONLINE GRADIENT DESCENT.*/
/*==================DATA STRUCTURES==================================*/
        typedef double real;//SIMPLE TOGGLE BETWEEN DOUBLE OR FLOAT PRECISION FOR REAL NUMBERS

        typedef struct{
            real activation;
            real delta;//DERIVATIVE OF LOSS WITH RESPECT TO THE PRE-ACTIVATION VALUE (INFERED USING THE CHAIN RULE AND THE ACTIVATION VALUE)
        }Neuron;

        typedef struct{
            int dim;//NUMBER OF NEURONS IN LAYER
            Layer* prev;//POINTER TO PREVIOUS LAYER IN NETWORK (NULL if input layer)
            Layer* next;//POINTER TO NEXT LAYER (NULL if output layer)
            Neuron* units;//POINTER TO FIRST NEURON IN LAYER

            real** weights;//MATRIX OF CONNECTION WEIGHTS BETWEEN PREVIOUS AND CURRENT LAYER
            real* bias;//VECTOR OF BIASES (CONNECTIONS WEIGHTS BETWEEN BIAS NEURON AND CURRENT LAYER)

            real** weights_grad;//GRADIENT OF LOSS W.R.T. CONNECTION WEIGHTS (same shape as weights)
            real* bias_grad;//GRADIENT OF LOSS W.R.T. BIASES
        }Layer;

        typedef struct{
            int n_layers;//NUMBER OF LAYERS
            int* layer_dims;
            Layer* layers;//POINTER TO THE FIRST LAYER
        }MultiLayerPerceptron;


/*======================================================================*/

/*====================GLOBAL VARIABLES==============================*/
real eta=.5;//LEARNING RATE
long r_state=1103527590;//INITIAL RANDOM SEED

int dim_in,dim_out,n_hidden_layers;
int*dims;

int n_epochs;
int n_training_examples,n_test_inputs;
char** testing_inputs;
char*** training_examples;//TRAINING (X,Y) PAIRS

MultiLayerPerceptron net;

/*===========================================   =======================*/

/*=========================ALGO & UTILITY FUNCTIONS PROTOTYPES==============================*/
long lcg();//PSEUDORANDOM NUMBER GENERATOR
real randf(real min,real max);//FLOATING POINT RANDOM GENERATION (CALLS LCG())
real sigmoid(real l);//SIGMOID FUNCTION (numerically stable version)

void setup();//PARSES INPUT AND INITIALIZES THE NETWORK (MEMORY ALLOCATION HANDLED HERE)
void initialize_weights();//RANDOMLY INITIALIZES THE WEIGHTS OF THE NETWORK

void forward(char* input);//FEED AN INPUT THROUGH THE NETWORK. ACTIVATIONS ARE STORED IN INDIVIDUAL NEURONS
void backward(char* input,char* output,bool recompute_activations);//PERFORMS BACKWARD PASS (COMPUTING DELTAS)
void compute_gradient(char* input,char* expected_output);//COMPUTES PARAMETER GRADIENTS

void gradient_step();//UPDATES THE WEIGHTS OF NETWORK (FOR NOW JUST VANILLA GRADIENT DESCENT)
void predict(char* input);//FORWARDS INPUT THROUGH THE NETWORK AND OUTPUTS(PRINTS) A PREDICTION
void cleanup();//PERFORMS MEMORY LIBERATION
/*=================================================*/

int main(){

    setup();
    initialize_weights();

    for(int i=0;i<n_epochs;++i){
        for(int j=0;j<n_training_examples;++j){
            backward(training_examples[j][0],training_examples[j][1],true);
            compute_gradient(training_examples[j][0],training_examples[j][1]);
            gradient_step();
        }
    }
    return 0;
}

/*==================FUNCTION DEFINITIONS===============*/
/*=================NUMERIC========================*/
long lcg(){
    r_state=(0x41c64e6d*r_state+0x3039)%0x80000000;
    return r_state;
}
real randf(real min,real max){
    return min+(max-min)*(((real)lcg())/0x7fffffff);
}
real sigmoid(real x){//only deal with [0,1] floating points to avoid overflow errors
    if(x>0){
        return 1./(1+(real)exp(-x));
    }
    else{
        real s=(real)exp(x);
        return s/(1+s);
    }
}

/*=================MEMORY HANDLING AND INPUT PARSING========================*/
    void setup(){
        scanf("%d%d%d%d%d%d",&dim_in,&dim_out,&n_hidden_layers,&n_test_inputs,&n_training_examples,&n_epochs);

        net.n_layers=2+n_hidden_layers;
        net.layers=(Layer*)malloc(net.n_layers*sizeof(Layer));
        net.layer_dims=(int*)malloc((2+n_hidden_layers)*sizeof(int));

        net.layer_dims[0]=dim_in;
        net.layer_dims[1+n_hidden_layers]=dim_out;

        for(int i=1;i<=n_hidden_layers;++i){
            scanf("%d",net.layer_dims+i);
        }

        testing_inputs=(char**)malloc(n_test_inputs*sizeof(char*));
        training_examples=(char***)malloc(n_training_examples*sizeof(char**));

        for(int i=0;i<n_test_inputs;i++){
            testing_inputs[i]=(char*)malloc(dim_in*sizeof(char));
            scanf("%s",testing_inputs[i]);
        }

        for(int i=0;i<n_training_examples;++i){
            training_examples[i]=(char**)malloc(2*sizeof(char*));
            training_examples[i][0]=(char*)malloc(dim_in*sizeof(char));
            training_examples[i][1]=(char*)malloc(dim_out*sizeof(char));
            scanf("%s%s",training_examples[i][0],training_examples[i][1]);
        }

        for(int i=0;i<net.n_layers;++i){
            net.layers[i].units=(Neuron*)malloc(net.layer_dims[i]*sizeof(Neuron));
            if(i>0){
                net.layers[i].prev=net.layers+i-1;
                net.layers[i].weights=(real**)malloc(net.layer_dims[i-1]*sizeof(real*));
                net.layers[i].weights_grad=(real**)malloc(net.layer_dims[i-1]*sizeof(real*));

                for(int j=0;j<net.layer_dims[i-1];j++){
                    net.layers[i].weights[j]=(real*)malloc(net.layer_dims[i]*sizeof(real));
                    net.layers[i].weights_grad[j]=(real*)malloc(net.layer_dims[i]*sizeof(real));

                }

                net.layers[i].bias=(real*)malloc(net.layer_dims[i]*sizeof(real));
                net.layers[i].bias_grad=(real*)malloc(net.layer_dims[i]*sizeof(real));

            }
            if(i<net.n_layers-1)net.layers[i].next=net.layers+i+1;
        } 
    }