# FPGA4HEP-Brevitas
Jet Classification and Regression training in Brevitas


## Data
Place the file below in /path/to/FPGA4HEP-Brevitas/data/

https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v

## Training and evaluation

For training or testing:
``` 
python train.py --batch-size      batchsize     (int)     \
                --test-batch-size testbatchsize (int)     \
                --epochs          epochs        (int)     \ 
                --lr              lr            (float)   \
                --momentum        momentum      (float)   \
                --seed            seed          (int)     \
                --log-interval    loginterval   (int)     \
                --cuda            cuda          (bool)    \
                --name            name          (string)  \
                --test            test          (bool)
```
