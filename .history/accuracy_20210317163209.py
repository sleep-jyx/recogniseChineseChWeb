import torch
ch_path = 'checkpoint_3_epoch.pkl'
checkpoint = torch.load(ch_path)
model.load_state_dict(checkpoint['model_state_dict'])
