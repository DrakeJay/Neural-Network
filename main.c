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


typedef struct {
    u32 num_vars;
    
    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;

} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;
    
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model_var* mv_create(
    mem_arens* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

model_var* mv_relu(
    mem_arena* arena, model_xontext* model,
    model_var* input, u32 flags
); 

model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_sub(
    mem_aren* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);2

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
);

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);






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

//Activation function
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);
b32 mat_cross_entropy_add_grad(
    matrix* p_g, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);

//create matrix
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat -> rows = rows;
    mat -> cols = cols;
    mat -> data = PUSH_ARRAY(arena,f32, (u64)rows * cols);   // push array and others are from the arena.h files

    return mat;
}



//load matrix``
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
    
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return false;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b){
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }
    if (out->rows != a->rows || out->cols != a->cols) {
        return false;
    }

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return false;
}


void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[j + i * out->cols] += 
                    a->data[k + i * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[j + k * b->cols];
            }
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[j + i * out->cols] += 
                    a->data[i + k * a->cols] * 
                    b->data[k + j * b->cols];
            }
        }
    }
}

b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b)
    {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;
    
    if (a_cols != b_rows) {return false;}
    if (out->rows != a_rows || out->cols != b_cols) { return false; }
    
    if (zero_out) {
        mat_clear(out);
    
    }
    
    u32 transpose = (transpose_a << 1 ) | transpose_b;
    switch (transpose) {
        case 0b00: { _mat_mul_nn (out, a, b);  } break;
        case 0b01: { _mat_mul_nt (out, a, b);  } break;
        case 0b10: { _mat_mul_tn (out, a, b);  } break;
        case 0b11: { _mat_mul_nn (out, a, b);  } break;
    } 
    
    return true;
    
}

// -----------------ACTIVATION FUNCTIONS----------------------

b32 mat_relu(matrix* out, const matrix in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    u64 size = (64)out->rows * out->cols;
    for (u64 i = o; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
}
    return true;
}

b32 mat_softmax(matrix* out, const matrix in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    u64 size = (u64)out->rows * out-> cols;
    //softmax
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
}
    mat_scale(out, 1.0f / sum);

    return true;
}
