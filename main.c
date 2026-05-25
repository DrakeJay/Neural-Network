#include <stdio.h>
#include <string.h>
#define _CRT_SECURE_NO_WARNINGS

#include "base.h"
#include "arena.h"
#include "prng.h"


#include "arena.c"
#include "prng.c"


typedef struct {
    u32 rows, cols;
    // row - major
    f32* data;
} matrix;



matrix* mat_creat(mem_arena* arena, u32 rows, u32 cols);
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 x);
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
u64 mat_argmax(matrix* mat);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);




//create matrix
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat -> rows = rows;
    mat -> cols = cols;
    mat -> data = PUSH_ARRAY(arena,f32, (u64)rows * cols);   // push array and others are from the arena.h files

    return mat;
}



//load matrix
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename ) {
    matrix* mat = mat_create(arena, rows, cols);
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f,0, SEEK_SET);

    size = MIN(size, sizeof(f32) * rows * cols);
    fread(mat->data, 1 , size, f);
    fclose(f);

    return mat;
}

b32 mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src -> rows || dst->cols != src -> cols) {
        return false;
    }
    
    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}

void mat_clear(matrix* mat){
    memset(mat->data, 0 , sizeof(f32) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;

}


void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    u64 size= (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }
}

void mat_scale(matrix* mat, f32 scale) {
    u64 size = (u64)mat->rows * mat-> cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;  
    }
}



f32 mat_sum(matrix* mat) {
    u64 size = (u64)mat-> rows * mat->cols;
    
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }
    return sum;
}   

u64 mat_argmax(matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;
    
    u64 max_i = 0;
    for (u64 i = 0; i < size; i++) {
        if (mat->data[i] > mat->data[max_i]){
            max_i = i;
        }
    }
}

b32 mat_add(matrix* out, const matrix* a , const matrix* b) {
    if (a->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }
    
}






