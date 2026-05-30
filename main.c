#include <stdio.h>
#include <string.h>
#include <unistd.h>
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





typedef enum {
    MV_FLAG_NONE = 0,

    MV_FLAG_REQUIRES_GRAD = (1 << 0),    //BACKPROPAGATION
    MV_FLAG_PARAMETER = (1 << 1),   //WEIGHTS AND BIAS
    MV_FLAG_INPUT = (1 << 2),       //INPUT 
    MV_FLAG_OUTPUT = (1 << 3),      //OUTPUT PREDICTION
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),  // ACTUAL TRUTH VALUE 
    MV_FLAG_COST =  (1 << 5),       //LOSS 
} model_var_flags;


typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,

    _MV_OP_UNARY_START,

    MV_OP_RELU,
    MV_OP_SOFTMAX,

    _MV_OP_BINARY_START,
    
    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op;

#define MODEl_VAR_MAX_INPUTS 2
  
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))




typedef struct model_var {
    u32 index;
    u32 flags;
    
    matrix* val;    //FORWARD VALUE
    matrix* grad;   //GRADIENT
    
    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;


typedef struct {
    model_var** vars;
    u32 size;
} model_program;





typedef struct {
    u32 num_vars;
    
    model_var* input;    //IMAGE 
    model_var* output;  //PREDICTION
    model_var* desired_output;  //ACTUAL ANSWER
    model_var* cost;            //LOSS USING CROSS-ENTROPY

    model_program forward_prog;   //COMPUTE OUTPUT
    model_program cost_prog;     // COMPUTES COST

} model_context;




//DATA AND HYPER PARAMETERS
typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;
    
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;







//DECLERATIONS OF TRAINING / FORWARD

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

model_var* mv_relu(
    mem_arena* arena, model_context* model,
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
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
);

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);


void model_prog_compute(model_program* prog);
void model_prog_compute_grads(model_program* prog);

model_context* model_create(mem_arena* arena);
void model_compile (mem_arena* arena, model_context* model);
void model_feedforward(model_context* model);
void model_train(
    model_context* model,
    const model_training_desc* training_desc
);


//mnist digit

void draw_mnist_digit(f32* data);
void create_mnist_model(mem_arena* arena, model_context* model);



// MATRIX DECLARATIONS
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

//ACTIVATION FUNCTIONS DECLERATIONS 
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);



//GRADIENT BACKWARD PROPAGATIONS
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(
    matrix* out, const matrix* softmax_out, const matrix* grad
);
b32 mat_cross_entropy_add_grad(
    matrix* p_g, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);







//lAYERS OF THE MATRIX 
//ALLOCATTES THE ROWS AND COLUMNS OF THE MATRIX, CREATES MATRIX WITH ALL ZEROS
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat -> rows = rows;
    mat -> cols = cols;
    mat -> data = PUSH_ARRAY(arena,f32, (u64)rows * cols);   // push array and others are from the arena.h files

    return mat;
}


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




//COPY, IF SHAPES DONT MATCH RETURN FALSE
b32 mat_copy(matrix* dst, matrix* src) {
    if (dst->rows != src -> rows || dst->cols != src -> cols) {
        return false;
    }
    
    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);

    return true;
}



// ALL ELEMENTS ARE ZERO
void mat_clear(matrix* mat){
    memset(mat->data, 0 , sizeof(f32) * (u64)mat->rows * mat->cols);
}



// EVERY ELEMENT IS SET TO X
void mat_fill(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;

}

//RANDOM VALUES 
void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    u64 size= (u64)mat->rows * mat->cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }
}
// ALL ELEMENTS MULTIPLIED IN PLACE
void mat_scale(matrix* mat, f32 scale) {
    u64 size = (u64)mat->rows * mat-> cols;

    for (u64 i = 0; i < size; i++) {
        mat->data[i] *= scale;  
    }
}


// SUMS ALL ELEMNETS, TOTAL COST INTO SCALAR
f32 mat_sum(matrix* mat) {
    u64 size = (u64)mat-> rows * mat->cols;
    
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        sum += mat->data[i];
    }
    return sum;
}   


//LARGEST ELEMENST, READ THE PREDICTED OUTPUT
u64 mat_argmax(matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;
    
    u64 max_i = 0;
    for (u64 i = 0; i < size; i++) {
        if (mat->data[i] > mat->data[max_i]){
            max_i = i;
        }
    }
}



//OUT = A + B
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
    return true;
}
//OUT = A - B
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

    return true;
}

// out = A @ B        A == ROWS x K   B = K x COLS    
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
// OUT = A @ Btranspose      
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
//out = Atranspose @ B
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
//OUT = Atranspose @ Btranspose
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
//gradient accross multiple paths during backproagation
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b)
    {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;
    
    if (a_cols != b_rows) {return false;}    //check dims match
    if (out->rows != a_rows || out->cols != b_cols) { return false; }   // output shape
    
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
// ReLU : out= max(0, input)
b32 mat_relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) {
        return false;
    }
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = MAX(0, in->data[i]);
}
    return true;
}

//softmax   :   out[i] =  exponent(input[i]) / sum_j exponent(input[j])
b32 mat_softmax(matrix* out, const matrix* in) {
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


// cross entropy :  output[i] = -p[i] * log(q[i])    p = target distribution / q = predicion
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) { return false; }
    if (out->rows != p->rows || out->cols != p->cols) {return false;}

    u64 size = (u64)out->rows * out->cols;
    for (u64 i =0; i < size; i++) {
        out->data[i] = p->data[i] == 0.0f ?
            0.0f : p->data[i] * -logf(q->data[i]);
    }
}





//TO DO

b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad){
    if(out->rows != in->rows || out->cols != in->cols){
        return false;
    }
    if(out->rows != grad->rows || out->cols != grad->cols) {
        return false;
    }
    u64 size = (u64) out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
    out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }
    return true;
}


b32 mat_softmax_add_grade(matrix* out, const matrix* softmax_out, const matrix* grad){
    if (softmax_out->rows != 1 && softmax_out->cols != 1){
        return false;
    }
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);
    
    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix* jacobian = mat_create(scratch.arena, size, size);
    
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j ++){
            jacobian->data[j + i * size] = 
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }

    mat_mul(out, jacobian, grad, 0, 0, 0);  // ACVCUMALATES J @ gradient into the output
    

    arena_scratch_release(scratch);
    
    return true;

}

b32 mat_cross_entropy_add_grad(matrix* p_grad, matrix* q_grad, const matrix* p, const matrix* q, const matrix* grad){
    if (p->rows != q->rows || p->cols != q->cols) {
    return false;
}
    u64 size = (u64)p->rows * p->cols;
    
    if (p_grad != NULL) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) {
            return false;
        }

        for (u64 i = 0; i < size; i++) {
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
        }
    }
    if (q_grad != NULL) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) {
            return false;
        }
        for (u64 i = 0; i < size; i++) {
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
        }
    }
    return true;
}






//construction of the graph    mb _ fucntions
model_var* mv_create(mem_arena* arena, model_context* model, u32 rows, u32 cols, u32 flags) {
    model_var* out = PUSH_STRUCT(arena, model_var);
    
    out->index = model->num_vars++;
    out->flags = flags;
    out->op = MV_OP_CREATE;
    out->val = mat_create(arena, rows, cols);

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        out->grad = mat_create(arena, rows, cols);
    }
    if (flags & MV_FLAG_INPUT) {model->input = out;} 
    if (flags & MV_FLAG_OUTPUT) {model->output = out;}
    if (flags & MV_FLAG_DESIRED_OUTPUT) {model->desired_output = out;}
    if (flags & MV_FLAG_COST) {model->cost = out;}
    
    return out;
}   

//TODO binary() relu() softmax() add() sub() matmul() crossEntropyu()

model_var* _mv_unary_impl(mem_arena* arena, model_context* model, model_var* input, u32 rows, u32 cols, u32 flags, model_var_op op){
    if (input->flags & MV_FLAG_REQUIRES_GRAD) {
        flags|= MV_FLAG_REQUIRES_GRAD;
    }
    model_var* out = mv_create(arena, model, rows, cols, flags);
    
    out->op = op;
    out->inputs[0] = input;

    return out;
}

model_var* _mv_binary_impl(mem_arena* arena, model_context* model, model_var* a ,model_var* b, u32 rows, u32 cols, u32 flags, model_var_op op){
    if (
        (a->flags & MV_FLAG_REQUIRES_GRAD) || (b->flags & MV_FLAG_REQUIRES_GRAD)) {
            flags |= MV_FLAG_REQUIRES_GRAD;
    }
    model_var* out = mv_create(arena,model, rows, cols,flags);
    
    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;
    
    return out;
}




model_var* mv_relu(mem_arena* arena, model_context* model, model_var* input, u32 flags) {
    return _mv_unary_impl(
        arena, model, input,
        input->val->rows, input->val->cols,
        flags, MV_OP_RELU
    );
}


model_var* mv_softmax(mem_arena* arena, model_context* model, model_var* input, u32 flags) {
    return _mv_unary_impl(arena, model, input, input->val->rows, input->val->cols,flags, MV_OP_SOFTMAX);

}

model_var* mv_add(mem_arena* arena, model_context* model, model_var* a, model_var* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(arena, model, a,b ,a->val->rows, a->val->cols, flags, MV_OP_ADD);

}


model_var* mv_sub(mem_arena* arena, model_context* model, model_var* a, model_var* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(arena, model, a,b ,a->val->rows, a->val->cols, flags, MV_OP_SUB);

}


model_var* mv_matmul(mem_arena* arena, model_context* model, model_var* a, model_var* b, u32 flags) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(arena, model, a,b ,a->val->rows, a->val->cols, flags, MV_OP_MATMUL);
}


model_var* mv_cross_entropy(mem_arena* arena, model_context* model, model_var* p, model_var* q, u32 flags) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) {
        return NULL;
    }
    return _mv_binary_impl(arena, model, p,q ,p->val->rows, q->val->cols, flags, MV_OP_CROSS_ENTROPY);
}

model_program model_prog_create(mem_arena* arena, model_context* model, model_var* out_var){
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);
    
    b8* visited = PUSH_ARRAY(scratch.arena, b8, model->num_vars);
    
    u32 stack_size = 0;
    u32 out_size = 0;
    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
    model_var** out = PUSH_ARRAY(scratch.arena,model_var*, model->num_vars);

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        model_var* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) {continue; }

        if (visited[cur->index]) {
            if (out_size < model->num_vars) {
                out[out_size++] = cur;
            }
           continue;
        }
        
        visited[cur->index] = true;
        
        if (stack_size < model->num_vars) {
            stack[stack_size++] = cur;
        }
        
        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i <num_inputs; i++) {
            model_var* input = cur->inputs[i];

            if (input->index >= model->num_vars || visited[input->index]) {
                continue;
            }

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size-1; k++) {
                        stack[k] = stack[k+1];

                    }
                    stack_size--;
                }
            }
            
            if (stack_size < model->num_vars) {
                stack[stack_size++] = input;
            }
        }
    }
    model_program prog = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, model_var*,out_size)
    };
    
    memcpy(prog.vars, out, sizeof(model_var*) * out_size);
    
    arena_scratch_release(scratch);
    
    return prog;
}

////TODO ------------ MODEL PROG COMPUTE 
void model_prog_compute_grads(model_program* prog){
    for (u32 i = 0; i < prog->size - 1; i++) {
        model_var* cur = prog->vars[i];
    
        if ((cur->FLAGS & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD){
            continue;
        }
        if (cur->flags * MV_FLAG_PARAMETER) {
            continue;
        }
        mat_clear(cur->grad);
    }

    mat_fill(prog->vars[prog->size-1]->grad, 1.0f);
    
    for (i64 i = (i64)prog->size - 1; i>= 0; i--){
        model_var* cur = prog->vars[i];
        
        if ((cur->flags & MV_FLAG_REQUIRES_GRAD) == 0){
            continue;
}
        model_var* a = cur->inputs[0];
        model_var* b = cur->inputs[1];
        
        u32 num_inputs = MV_NUM_INPUTS(cur->op);
        
        if (
            num_inputs == 1 &&
            (a->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD &&
            (b->flags & MV_FLAG_REQUIRES_GRAD) != MV_FLAG_REQUIRES_GRAD)
        {
            continue;}
}
        switch (cur->op) {
            case MV_OP_NULL;

}
    
)
}

//// TODO MODEL COMPUTE GRADS






