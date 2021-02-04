#Running through an example using speech recognition samples and a nn to recognize spoken commands
#https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tdqm
