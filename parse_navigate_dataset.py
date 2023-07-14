import numpy as np
import torch

# for task_number_str in ['50', '500', '5k', '50k']:
    # data = torch.load(f'data/NavigateTo_full_obs_{task_number_str}tasks.pt')
    # s = data['states'][:, 20*20*3:]
    # shelf_mask = s[:, 2] >= 5
    # assert 0.49 <= shelf_mask.mean() <= 0.51, f'Mean was {shelf_mask.mean()} but should be 0.5'
    # shelf_mask_checker = s[:, 3] >= 2
    # assert shelf_mask[shelf_mask_checker].all()

    # data_shelf = {k: v[shelf_mask] for k, v in data.items()}
    # data_book = {k: v[~shelf_mask] for k, v in data.items()}

    # assert (data_shelf['states'][:, 20*20*3 + 2] >= 5).all()
    # assert (data_book['states'][:, 20*20*3 + 2] <= 1).all()

    # torch.save(data_shelf, f'data/NavigateToShelf_full_obs_{task_number_str}tasks.pt', pickle_protocol=4)
    # torch.save(data_book, f'data/NavigateToBook_full_obs_{task_number_str}tasks.pt', pickle_protocol=4)

# CUPBOARD
# data = torch.load(f'data_cupboard/data/NavigateTo_full_obs_50ktasks.pt')
# s = data['states']
# cupboard_mask = s[:, 2] >= 5
# assert 0.49 <= cupboard_mask.mean() <= 0.51, f'Mean was {cupboard_mask.mean()} but should be 0.5'
# cupboard_mask_checker = s[:, 3] >= 2
# assert cupboard_mask[cupboard_mask_checker].all()

# data_cupboard = {k: v[cupboard_mask] for k, v in data.items()}
# data_cup = {k: v[~cupboard_mask] for k, v in data.items()}

# assert (data_cupboard['states'][:, 2] >= 5).all()
# assert (data_cup['states'][:, 2] <= 1).all()

# torch.save(data_cupboard, f'data_cupboard/data/NavigateToCupboard_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(data_cup, f'data_cupboard/data/NavigateToCup_full_obs_50ktasks.pt', pickle_protocol=4)

# BALLBIN
# data = torch.load(f'data_ballbin/data/NavigateTo_full_obs_50ktasks.pt')
# s = data['states']
# bin_mask = s[:, 2] >= 4
# assert 0.49 <= bin_mask.mean() <= 0.51, f'Mean was {bin_mask.mean()} but should be 0.5'
# bin_mask_checker = s[:, 3] >= 4
# assert bin_mask[bin_mask_checker].all()

# data_bin = {k: v[bin_mask] for k, v in data.items()}
# data_ball = {k: v[~bin_mask] for k, v in data.items()}

# assert (data_bin['states'][:, 2] >= 4).all()
# assert (data_ball['states'][:, 2] <= 0.5).all()

# torch.save(data_bin, f'data_ballbin/data/NavigateToBin_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(data_ball, f'data_ballbin/data/NavigateToBall_full_obs_50ktasks.pt', pickle_protocol=4)

# STICKBASKET
data = torch.load(f'data_stickbasket/data/NavigateTo_full_obs_50ktasks.pt')
s = data['states']
basket_mask = s[:, 2] >= 7
assert 0.49 <= basket_mask.mean() <= 0.51, f'Mean was {basket_mask.mean()} but should be 0.5'
basket_mask_checker = s[:, 3] <= 3
assert basket_mask[basket_mask_checker].all()

data_basket = {k: v[basket_mask] for k, v in data.items()}
data_stick = {k: v[~basket_mask] for k, v in data.items()}

assert (data_basket['states'][:, 2] >= 7).all()
assert (data_stick['states'][:, 2] <= 1).all()

torch.save(data_basket, f'data_stickbasket/data/NavigateToBasket_full_obs_50ktasks.pt', pickle_protocol=4)
torch.save(data_stick, f'data_stickbasket/data/NavigateToStick_full_obs_50ktasks.pt', pickle_protocol=4)

# BOXTRAY
# data = torch.load(f'data_boxtray/data/NavigateTo_full_obs_50ktasks.pt')
# s = data['states']
# tray_mask = s[:, 2] >= 11
# assert 0.49 <= tray_mask.mean() <= 0.51, f'Mean was {tray_mask.mean()} but should be 0.5'
# tray_mask_checker = s[:, 3] >= 3
# assert tray_mask[tray_mask_checker].all()

# data_tray = {k: v[tray_mask] for k, v in data.items()}
# data_box = {k: v[~tray_mask] for k, v in data.items()}

# assert (data_tray['states'][:, 2] >= 11).all()
# assert (data_box['states'][:, 2] <= 1).all()

# torch.save(data_tray, f'data_boxtray/data/NavigateToTray_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(data_box, f'data_boxtray/data/NavigateToBox_full_obs_50ktasks.pt', pickle_protocol=4)
