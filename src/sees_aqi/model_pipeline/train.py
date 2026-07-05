from neuralop.models import FNO

# Initialize the network backbone
model = FNO(
    n_modes=(16, 16), 
    in_channels=5, 
    out_channels=1, 
    hidden_channels=64
)