# print explained energy of each layer
#/root/username/distillm-2/outputs/representation_basis/math5k/explained_energy_layer7.pt
#/root/username/distillm-2/outputs/representation_basis/math5k/explained_energy_layer14.pt
#/root/username/distillm-2/outputs/representation_basis/math5k/explained_energy_layer28.pt

import torch

with open("outputs/representation_basis/math5k/explained_energy_layer7.pt", "rb") as f:
    explained_energy_layer7 = torch.load(f)
with open("outputs/representation_basis/math5k/explained_energy_layer14.pt", "rb") as f:
    explained_energy_layer14 = torch.load(f)    
with open("outputs/representation_basis/math5k/explained_energy_layer28.pt", "rb") as f:
    explained_energy_layer28 = torch.load(f)


teacher_hidden_dim = 3584
for d in [
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
]:

    if d <= teacher_hidden_dim:
        energy_7 = explained_energy_layer7[d - 1].item()
        energy_14 = explained_energy_layer14[d - 1].item()
        energy_28 = explained_energy_layer28[d - 1].item()

        print(
            f"d={d:4d} : "
            f"layer 7: {energy_7 * 100:7.3f}% | "
            f"layer 14: {energy_14 * 100:7.3f}% | "
            f"layer 28: {energy_28 * 100:7.3f}%"
        )
