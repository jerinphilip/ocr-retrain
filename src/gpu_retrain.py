from gpu_ocr.model import create_network
from warpctc_pytorch import CTCLoss

Net = create_network(input_size=30, hidden_size=50, hidden_depth=3, output_classes=108)
print(Net)
Criterion = CTCLoss()

# Load inputs as images, truths
inputs = #Load here
targets = #Load here
max_epochs = 1000
for epoch in range(max_epochs):
    print(epoch)

