# Learning Loss for Active Learning
NUM_TRAIN = 65862 # 50000 # N
NUM_VAL   = 16466 # 50000 - NUM_TRAIN
BATCH     = 16 # B
SUBSET    = 10000 # 10000 M
ADDENDUM  = 1000 # 1000 K

MARGIN = 1.0 # xi
WEIGHT = 0.1 # lambda

TRIALS = 1 # 3
CYCLES = 10

EPOCH = 100 # 200
LR = 5e-5
MILESTONES = [160]
EPOCHL = 80 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4