#include "ggml.h"
#include "ggml-opt.h"

#include "mnist-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    fprintf(stdout, "Step 1: Setting up paths...\n");

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s mnist-fc-f32.gguf data/MNIST/raw/t10k-images-idx3-ubyte data/MNIST/raw/t10k-labels-idx1-ubyte [CPU/CUDA0]\n", argv[0]);
        exit(1);
    }
    
    fprintf(stdout, "Step 2: Initializing dataset...\n");
    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(MNIST_NINPUT, MNIST_NCLASSES, MNIST_NTEST, MNIST_NBATCH_PHYSICAL);

    fprintf(stdout, "Step 3: Loading images...\n");
    if (!mnist_image_load(argv[2], dataset)) {
        fprintf(stderr, "Failed to load images from %s\n", argv[2]);
        return 1;
    }

    fprintf(stdout, "Step 4: Loading labels...\n");
    if (!mnist_label_load(argv[3], dataset)) {
        fprintf(stderr, "Failed to load labels from %s\n", argv[3]);
        return 1;
    }

    fprintf(stdout, "Step 5: Generating random index...\n");
    const int iex = rand() % MNIST_NTEST;
    mnist_image_print(stdout, dataset, iex);

    fprintf(stdout, "Step 6: Setting up backend...\n");
    const std::string backend = argc >= 5 ? argv[4] : "";

    fprintf(stdout, "Step 7: Loading model...\n");
    const int64_t t_start_us = ggml_time_us();
    mnist_model model = mnist_model_init_from_file(argv[1], backend, MNIST_NBATCH_LOGICAL, MNIST_NBATCH_PHYSICAL);
    mnist_model_build(model);
    const int64_t t_load_us = ggml_time_us() - t_start_us;
    fprintf(stdout, "%s: loaded model in %.2lf ms\n", __func__, t_load_us / 1000.0);

    fprintf(stdout, "Step 8: Evaluating model...\n");
    ggml_opt_result_t result_eval = mnist_model_eval(model, dataset);
    fprintf(stdout, "Model evaluation completed\n");

    fprintf(stdout, "Step 9: Getting predictions...\n");
    std::vector<int32_t> pred(MNIST_NTEST);
    ggml_opt_result_pred(result_eval, pred.data());
    fprintf(stdout, "%s: predicted digit is %d\n", __func__, pred[iex]);

    fprintf(stdout, "Step 10: Calculating loss...\n");
    double loss;
    double loss_unc;
    ggml_opt_result_loss(result_eval, &loss, &loss_unc);
    fprintf(stdout, "%s: test_loss=%.6lf+-%.6lf\n", __func__, loss, loss_unc);

    fprintf(stdout, "Step 11: Calculating accuracy...\n");
    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result_eval, &accuracy, &accuracy_unc);
    fprintf(stdout, "%s: test_acc=%.2lf+-%.2lf%%\n", __func__, 100.0*accuracy, 100.0*accuracy_unc);

    fprintf(stdout, "Step 12: Cleanup...\n");
    ggml_opt_result_free(result_eval);

    fprintf(stdout, "Program completed successfully\n");
    return 0;
}