#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<string.h>

#define BUFFER_SIZE 1024

struct dim
{
    int rows;
    int cols;
};

struct LR_params{
    float alpha;
    float split_ratio;
};

struct theta{
    double *t;
};

struct TTSplit{
    float **training;
    float **testing;
};

int contains_int(int* arr, int n, int val)
{
    for(int i=0; i < n; i++)
    {
        if(arr[i] == val) return 1;
    }
    return 0;
}

int contains_float(float* arr, int n, float val)
{
    for(int i=0; i < n; i++)
    {
        if(arr[i] == val) return 1;
    }
    return 0;
}

float** create_array_2d(int n, int m)
{
    float* values = calloc(m * n, sizeof(float));
    float** rows = malloc(n * sizeof(float*));
    for(int i = 0; i < n; i++) {
        rows[i] = values + i * m;
    }
    return rows;
}

void destroy_array_2d(float** arr)
{
    free(*arr);
    free(arr);
}

void read_data(FILE *csv, float*** data, struct dim csv_dim)
{
    float temp;
    int i;
    int j;
    for(i = 0; i < csv_dim.rows; i++) {
        for(j = 0; j < csv_dim.cols; j++) {
            fscanf(csv, "%f", &temp);
            (*data)[i][j] = temp;
            fscanf(csv, ",");
        }
    }
    fclose(csv);
}

struct dim get_csv_dimensions(FILE *file)
{
    const char *delimeter = ",";
    char buffer[ BUFFER_SIZE ];
    char *token;

    int rows_count = 0;
    int cols_count = 0;

    while(fgets(buffer, BUFFER_SIZE, file) != NULL){
        token = strtok(buffer, delimeter);
        while(token != NULL) {
            if(strstr(token, "\n") != NULL) {
                rows_count++;
                if(cols_count != 0) {
                    cols_count = 0;
                }
            }
            cols_count++;
            token = strtok(NULL, delimeter);
        }
    }
    rows_count += 1;
    cols_count -= 1;
    fseek(file, 0, SEEK_SET);

    return (struct dim){.rows=rows_count, .cols=cols_count};
}

struct TTSplit train_test_split(float **data,struct LR_params params,int rows,int cols){
    struct TTSplit split;
    int sample_rows = (int)((float)rows * (1-params.split_ratio));
    split.training = (float**) malloc(sample_rows * sizeof(float) * cols);
    split.testing = (float**) malloc((rows-sample_rows) * sizeof(float) * cols);
    int* indecies = malloc(rows * sizeof(int));
    int count = 0;
    for(int i=0; i < sample_rows; i++) indecies[i] = -1;

    while(count < sample_rows)
    {
        int random_index = rand() % rows;
        if(!contains_int(indecies, sample_rows, random_index))
        {
            split.training[count] = data[random_index];
            indecies[count] = random_index;
            count++;
        }
    }
    int index=0;
    while(count < rows)
    {
        if(!contains_int(indecies, sample_rows, index))
        {
            split.testing[count] = data[index];
            indecies[count] = index;
            count++;
        }
        index++;
    }
    return split;
}

double cross_validation(float **data,struct LR_params params,int rows,int cols){
    struct TTSplit train=train_test_split(data,params,rows,cols);
    //struct theta res=fit_model(train.training,params,rows,cols);
    return 1.0;
}

int main(){
    char *fn="data.csv";
    FILE *csv_file;
    csv_file = fopen(fn,"r");
    if(csv_file == NULL) {
        printf("Error: can't open file \n");
        return -1;
    }
    struct dim csv_dim = get_csv_dimensions(csv_file);
    int rows = csv_dim.rows;
    int cols = csv_dim.cols;
    float** data = create_array_2d(rows,cols);
    read_data(csv_file, &data, csv_dim);
    struct LR_params params = {.alpha=0.1,.split_ratio=0.25};
    clock_t begin = clock();
    double cv_accuracy = cross_validation(data,params,rows,cols);
    clock_t end = clock();
    printf("\ntime taken: %fs | accuracy: %.20f\n", (double)(end - begin) / CLOCKS_PER_SEC, cv_accuracy);
    destroy_array_2d(data);
}