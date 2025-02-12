import time

from torchray.attribution.rise import rise
from torchray.benchmark import get_example_data, plot_example
from torchray.utils import get_device


# Obtain example data.
model, x, category_id, _ = get_example_data() # arch='resnet50'

# Run on GPU if available.
device = get_device()
model.to(device)
x = x.to(device)

start = time.time()
# RISE method.
saliency = rise(model, x, num_masks=8000)
saliency = saliency[:, category_id].unsqueeze(0)
print('RISE took', time.time() - start)

# Plots.
plot_example(x, saliency, 'RISE', category_id, save_path='/home/etsykuno/TorchRay/outputs/rise_output.png')
