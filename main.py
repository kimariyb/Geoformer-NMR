from loader.carbon import CarbonDataset
from network.decoder import GeoformerPretrainedModel, GeoformerConfig

dataset = CarbonDataset(root='./data')
data = dataset[0]

# config = GeoformerConfig()
# net = GeoformerPretrainedModel()

# y = net(data.atom_x, data.mask)

print(data.atom_y)
print(data.mask)
