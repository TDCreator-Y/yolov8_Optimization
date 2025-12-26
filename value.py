from ultralytics.nn.modules.dcn_conv import DCNConv

cnt = 0
for m in model.model.modules():
    if isinstance(m, DCNConv):
        cnt += 1
print("DCNConv count:", cnt)
