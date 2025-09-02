from Loop_EMAP_train import loop_train_test
import warnings
import time

# remove abundant output
warnings.filterwarnings('ignore')

## global constant value
run_times = 10
patch_size = 13
train_ratio = 0.05
pca_components = 30
emap_components = 1
BATCH_SIZE_TRAIN = 128
epochs = 250
cuda = [0]

def Run_experiment():
    hp = {
        'run_times': run_times,
        'patch_size': patch_size,
        'train_ratio': train_ratio,
        'pca_components': pca_components,
        'emap_components': emap_components,
        'BATCH_SIZE_TRAIN': BATCH_SIZE_TRAIN,
        'epochs': epochs,
        'cuda': cuda
    }

    loop_train_test(hp)



if __name__ == '__main__':
    Run_experiment()
    print(time.asctime(time.localtime()))