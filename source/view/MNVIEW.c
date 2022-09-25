//
// MNIST viewer for X1
//
// 2021-03-08 programed by NAKAUE,T
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#define MODEL_INPUT_SIZE    28
#define OS_WORK_COLORF      0xec96
#define OS_WORK_TXADR       0xec8e


// PCGで1文字出力
void putchar_pcg(uint8_t c)
{
    const uint16_t* cursor = (uint8_t*) OS_WORK_TXADR;

    outp(0x2000 + (*cursor), 0x27);
    outp(0x3000 + (*cursor), c);
    putchar('\x1c');
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


void draw_input_img_pcg(uint8_t input[][MODEL_INPUT_SIZE])
{
    for(uint_fast16_t y = 0; y < MODEL_INPUT_SIZE; y += 2) {
        for(uint_fast16_t x = 0; x < MODEL_INPUT_SIZE; x++) {
            uint8_t v1 = input[0][ y      * MODEL_INPUT_SIZE + x];
            uint8_t v2 = input[0][(y + 1) * MODEL_INPUT_SIZE + x];
            uint8_t v = (v1 & 0xf0) | ((v2 >> 4) & 0x0f);

            putchar_pcg(v);
        }
        puts("");
    }

    return;
}


#define draw_input_img(x) draw_input_img_pcg(x)


int main(int argc, char *argv[])
{
    uint8_t *img = malloc(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

    for(uint_fast8_t i = 1; i < argc; i++) {
        FILE *fp = fopen(argv[i], "rb");
        if(fp != NULL) {
            puts(argv[i]);
            fread(img, 1, MODEL_INPUT_SIZE * MODEL_INPUT_SIZE, fp);
            fclose(fp);
            draw_input_img(img);
        } else {
            printf("%s not found.\n", argv[i]);
        }
    }

    return 0;
}
