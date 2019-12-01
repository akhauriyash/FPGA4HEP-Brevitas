# FPGA4HEP-Brevitas
Jet Classification and Regression training in Brevitas


## Data
Place the file below in /path/to/FPGA4HEP-Brevitas/data/

https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v

## Training and evaluation

For training or testing:
``` 
python main.py  [--batch-size      batchsize     (int)   ]  \
                [--test-batch-size testbatchsize (int)   ]  \
                [--epochs          epochs        (int)   ]  \ 
                [--lr              lr            (float) ]  \
                [--momentum        momentum      (float) ]  \
                [--seed            seed          (int)   ]  \
                [--log-interval    loginterval   (int)   ]  \
                [--cuda            cuda          (bool)  ]  \
                [--name            name          (string)]  \
                [--test            test          (bool)  ]
```

## Requirements
1. Python3
2. PyTorch
3. Brevitas
4. Matplotlib>3.1.1 (For correct plotting of Confusion Matrix)
5. Pandas
6. scikit-learn 
7. yaml
