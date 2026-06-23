path = '/home/ya4v/.cache/huggingface/modules/transformers_modules/nvidia/MambaVision_hyphen_T_hyphen_1K/b1de77e17599566d98efb701c0231b1095dc3a67/modeling_mambavision.py'
with open(path) as f:
    content = f.read()
old = 'dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]'
new = 'dpr = torch.linspace(0, drop_path_rate, sum(depths), device="cpu").tolist()'
content = content.replace(old, new)
with open(path, 'w') as f:
    f.write(content)
print('patched')
