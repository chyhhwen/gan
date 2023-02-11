import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SIZE = 128
BATCH_SIZE = 16
EPOCH_LR = [(30, 0.01), (30, 0.001), (30, 0.0001)]
CHECKPOINT = "/data/image_compress"
DATA_FOLDER = "/data/pubfig_faces"