# Learning Loss for Active Learning
NUM_TRAIN = 65862 # 50000 # N
NUM_VAL   = 16466 # 50000 - NUM_TRAIN
BATCH     = 32 # B
SUBSET    = 1000 # 10000 M
ADDENDUM  = 100 # 1000 K

MARGIN = 1.0 # xi
WEIGHT = 0.1 # lambda

TRIALS = 1 # 3
CYCLES = 10

EPOCH = 100 # 200
LR = 1e-3
MILESTONES = [160]
EPOCHL = 80 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4