#%%
import torch
import torch.nn as nn
# %%
a = torch.rand((1, 512, 20, 20))
# %%
up1 = nn.Upsample(scale_factor=(2,2))
# %%
b = up1(a)
# %%
b.size()
# %%
up2 = nn.Upsample(scale_factor=(0.5, 2, 2))
# %%
c = up2(b)
# %%
c.size()
# %%
b = b.unsqueeze(1)
# %%
b.size()
# %%
c = c.squeeze(1)
# %%
c.size()
# %%
