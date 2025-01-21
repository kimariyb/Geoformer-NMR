from loader.carbon import CarbonDataset

dataset = CarbonDataset(root='./data')
data = dataset[0]
print(data.atom_y)
print(data.mask)
