import torch.multiprocessing

from .criterions import *
from .datasets import *
from .models import *
from .tasks import *

torch.multiprocessing.set_sharing_strategy("file_system")

print("fairseq plugins loaded...")
