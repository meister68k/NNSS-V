//
// MNIST recognize for pure C
//
// 2021-01-23 programed by NAKAUE,T
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#define MODEL_PARAM_SIZE    13600
#define MODEL_INPUT_SIZE    28
#define MODEL_OUTPUT_SIZE   10
#define MODEL_CONV_DEPTH    8
#define MODEL_CONV_OUT      ((MODEL_INPUT_SIZE / 2) - 1)
#define MODEL_FEAT_SIZE      (MODEL_CONV_OUT * MODEL_CONV_OUT * MODEL_CONV_DEPTH)


// Neural Network Model
typedef struct {
    int8_t *conv_bias;
    int8_t (*conv_kernel)[3][MODEL_CONV_DEPTH];
    int8_t (*dense_kernel)[MODEL_OUTPUT_SIZE];

    uint8_t (*input)[MODEL_INPUT_SIZE];
    uint8_t (*output_1)[MODEL_CONV_OUT][MODEL_CONV_DEPTH];
    uint8_t *output;
} NeuralNetwork;


// NeuralNetwork class constructor
NeuralNetwork* model_initialize(NeuralNetwork* model)
{
    model->conv_bias = (int8_t*) malloc(MODEL_PARAM_SIZE);
    if(model->conv_bias == NULL) return NULL;

    int8_t *conv_kernel = &(model->conv_bias[MODEL_CONV_DEPTH]);
    model->conv_kernel = (int8_t (*)[3][MODEL_CONV_DEPTH]) conv_kernel;
    model->dense_kernel = (int8_t (*)[MODEL_OUTPUT_SIZE]) &(conv_kernel[3 * 3 * MODEL_CONV_DEPTH]);

    model->output = (uint8_t*) malloc(
        MODEL_OUTPUT_SIZE + 
        MODEL_INPUT_SIZE * MODEL_INPUT_SIZE +
        MODEL_FEAT_SIZE
    );
    if(model->output == NULL) {
        free(model->conv_bias);
        return NULL;
    }

    uint8_t *input = &(model->output[MODEL_OUTPUT_SIZE]);
    model->input = (uint8_t (*)[MODEL_INPUT_SIZE]) input;
    model->output_1 =
        (uint8_t (*)[MODEL_CONV_OUT][MODEL_CONV_DEPTH]) &(input[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE]);

    return model;
}


// load network weigths
NeuralNetwork* model_load(NeuralNetwork* model, char *fname)
{
    FILE *fp = fopen(fname, "rb");
    if(fp == NULL) return NULL;

    fread(model->conv_bias, 1, MODEL_PARAM_SIZE, fp);

    fclose(fp);

    return model;
}


void conv_layer(NeuralNetwork* model, uint8_t input[][MODEL_INPUT_SIZE], uint8_t output[][MODEL_CONV_OUT][MODEL_CONV_DEPTH])
{
    int8_t kernel[3][3];
    int8_t bias;

    for(uint_fast8_t k = 0; k < MODEL_CONV_DEPTH; k++) {
        // get kernel value
        for(uint_fast8_t y = 0; y < 3; y++) {
//            for(uint_fast8_t x = 0; x < 3; x++) kernel[y][x] = model->conv_kernel[y][x][k];
            for(uint_fast8_t x = 0; x < 3; x++) kernel[0][y*3+x] = model->conv_kernel[0][0][(y*3+x)*MODEL_CONV_DEPTH+k];
        }
        bias = model->conv_bias[k];

        for(uint_fast8_t yy = 0; yy < MODEL_CONV_OUT; yy++) {
            for(uint_fast8_t xx = 0; xx < MODEL_CONV_OUT; xx++) {
                int16_t logits = 0;
                for(uint_fast8_t y = 0; y < 3; y++) {
                    for(uint_fast8_t x = 0; x < 3; x++) {
//                        logits += ((int16_t)(input[yy * 2 + y][xx * 2 + x])) * ((int16_t)kernel[y][x]);
                        logits += ((int16_t)(input[0][((int_fast16_t)(yy * 2 + y))*MODEL_INPUT_SIZE+xx * 2 + x])) * ((int16_t)kernel[0][y*3+x]);
                    }
                }
                logits = (logits >> 8) + bias;
                if(logits < 0) logits = 0;
                if(logits > 255) logits = 255;
//                output[yy][xx][k] = (uint8_t) logits;
                output[0][0][((int_fast16_t)(yy*MODEL_CONV_OUT+xx))*MODEL_CONV_DEPTH+k] = (uint8_t) logits;
            }
        }
    }

    return;
}


void dense_layer(int8_t kernel[][MODEL_OUTPUT_SIZE], uint8_t input[], uint8_t output[])
{

    for(uint_fast8_t j = 0; j < MODEL_OUTPUT_SIZE; j++) {
        int16_t logits = 0;
        for(uint_fast16_t i = 0; i < (MODEL_FEAT_SIZE); i++) {
//            logits += ((int16_t)(input[i])) * ((int16_t)kernel[i][j]);
            logits += ((int16_t)(input[i])) * ((int16_t)kernel[0][i*MODEL_OUTPUT_SIZE+j]);
        }
        if(logits < 0) logits = 0;
        output[j] = (uint8_t)(logits >> 5);
    }

    return;
}


uint_fast8_t model_predict(NeuralNetwork* model)
{
    conv_layer(model, model->input, model->output_1);
    dense_layer(model->dense_kernel, (uint8_t*) model->output_1, model->output);

    uint_fast8_t result = 0;
    uint8_t val = 0;

    for(uint_fast8_t i = 0; i < MODEL_OUTPUT_SIZE; i++) {
        if(model->output[i] > val) {
            val = model->output[i];
            result = i;
        }
    }

    return result;
}


void draw_input_img_hex(uint8_t input[][MODEL_INPUT_SIZE])
{
    for(uint_fast16_t y = 0; y < MODEL_INPUT_SIZE; y++) {
        for(uint_fast16_t x = 0; x < MODEL_INPUT_SIZE; x++) {
//            printf("%02X", (unsigned char)input[y][x]);
            printf("%02X", (unsigned char)input[0][y*MODEL_INPUT_SIZE+x]);
        }
        puts("");
    }

    return;
}


#define draw_input_img(x) draw_input_img_hex(x)


void draw_feature_img_hex(uint8_t output[][MODEL_CONV_OUT][MODEL_CONV_DEPTH], uint8_t k)
{
    for(uint_fast8_t y = 0; y < MODEL_CONV_OUT; y++) {
        for(uint_fast8_t x = 0; x < MODEL_CONV_OUT; x++) {
//            printf("%02X", (unsigned char)output[y][x][k]);
            printf("%02X", (unsigned char)output[0][0][(y*MODEL_CONV_OUT+x)*MODEL_CONV_DEPTH+k]);
        }
        puts("");
    }

    return;
}


#define draw_feature_img(x, k) draw_feature_img_hex(x, k)


int main(int argc, char *argv[])
{
    NeuralNetwork model;
    if(model_initialize(&model) == NULL) {
        puts("not enough memory.");
        abort();
    }

    if(model_load(&model, "MNMODEL.BIN") == NULL) {
        puts("model binary not found.");
        abort();
    }
    puts("model initialize ok.");

    for(uint_fast8_t i = 1; i < argc; i++) {
        FILE *fp = fopen(argv[i], "rb");
        if(fp != NULL) {
            puts(argv[i]);
            fread(model.input, 1, MODEL_INPUT_SIZE * MODEL_INPUT_SIZE, fp);
            fclose(fp);
            draw_input_img(model.input);

            uint_fast8_t label = model_predict(&model);
            puts("");
            draw_feature_img(model.output_1, 1);

            puts("");
            for(uint_fast8_t i = 0; i < MODEL_OUTPUT_SIZE; i++) printf("%d ", (unsigned char)model.output[i]);
            puts("");

            printf("result = %d\n\n", label);
        } else {
            printf("%s not found.\n", argv[i]);
        }
    }

    return 0;
}
