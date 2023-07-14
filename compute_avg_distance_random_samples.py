import torch
import numpy as np
from predicators import utils

datasets_per_domain = {
    'bookshelf': ['NavigateTo', 'NavigateToBook', 'NavigateToShelf', 'PickBook', 'PlaceBookOnShelf'],
    'cupboard': ['NavigateTo', 'NavigateToCup', 'NavigateToCupboard', 'PickCup', 'PlaceCupOnCupboard'],
    'stickbasket': ['NavigateTo', 'NavigateToStick', 'NavigateToBasket', 'PickStick', 'PlaceStickOnBasket'],
    'ballbin': ['NavigateTo', 'NavigateToBall', 'NavigateToBin', 'PickBall', 'PlaceBallOnBin'],
    'boxtray': ['NavigateTo', 'NavigateToBox', 'NavigateToTray', 'PickBox', 'PlaceBoxOnTray']
}

# for domain in ['cupboard', 'stickbasket', 'ballbin', 'boxtray']:
for domain in ['stickbasket']:
    for dataset in datasets_per_domain[domain]:
        # d = torch.load('data_grid/data/'+dataset+'_full_obs_50ktasks.pt')
        d = torch.load(f'data_{domain}/data/{dataset}_full_obs_50ktasks.pt')
        x = d['states']#[:, 20*20*3:]
        shuffle = np.random.permutation(x.shape[0])
        x_1 = x[shuffle[::2]]
        x_2 = x[shuffle[1::2]]
        if x_1.shape[0] > x_2.shape[0]:
            x_1 = x_1[1:]
        dist = np.linalg.norm(x_1 - x_2, axis=1).mean()

        torch.save({'dist': dist}, f'data_{domain}/data/{dataset}_distance_random_samples.pt')

        if dataset.startswith('NavigateTo'):
            lo = np.array([-8, -4])
            hi = np.array([9, 5])
            a = np.random.rand(x.shape[0], lo.shape[0]) * (hi - lo) + lo

            obj_x = x[:, 0]
            obj_y = x[:, 1]
            obj_w = x[:, 2]
            obj_h = x[:, 3]
            obj_yaw = x[:, 4]

            offset_x = a[:, 0]
            offset_y = a[:, 1]
            pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
            obj_h * offset_y * np.sin(obj_yaw)
            pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                    obj_h * offset_y * np.cos(obj_yaw)

            pos_x = np.clip(pos_x, 0, 20 - 1e-6)
            pos_y = np.clip(pos_y, 0, 20 - 1e-6)

            aux_labels = np.empty((x.shape[0], 1))
            for i in range(x.shape[0]):
                rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                if dataset == 'NavigateToCup':
                    aux_labels[i] = rect.line_segments[0].distance_nearest_point(pos_x[i], pos_y[i])
                elif dataset == 'NavigateToTray':
                    x1 = obj_x[i]
                    y1 = obj_y[i]
                    x2 = x1 + (obj_w[i] - obj_h[i]) * np.cos(obj_yaw[i])
                    y2 = y1 + (obj_w[i] - obj_h[i]) * np.sin(obj_yaw[i])
                    rect1 = utils.Rectangle(x1, y1, obj_h[i], obj_h[i], obj_yaw[i])
                    rect2 = utils.Rectangle(x2, y2, obj_h[i], obj_h[i], obj_yaw[i])
                    aux_labels[i] = min(rect1.distance_nearest_point(pos_x[i], pos_y[i]),
                                            rect2.distance_nearest_point(pos_x[i], pos_y[i]))
                else:
                    aux_labels[i] = rect.distance_nearest_point(pos_x[i], pos_y[i])

        elif dataset.startswith('Pick'):
            lo = np.array([0, -np.pi])
            hi = np.array([1, np.pi])
            a = np.random.rand(x.shape[0], lo.shape[0]) * (hi - lo) + lo

            obj_x = x[:, 0]
            obj_y = x[:, 1]
            obj_w = x[:, 2]
            obj_h = x[:, 3]
            obj_yaw = x[:, 4]

            robby_x = x[:, 6]
            robby_y = x[:, 7]
            robby_yaw = x[:, 8]

            offset_gripper = a[:, 0]

            tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
            tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

            aux_labels = np.empty((x.shape[0], 1))
            for i in range(x.shape[0]):
                rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                aux_labels[i] = rect.distance_nearest_point(tip_x[i], tip_y[i])


        elif dataset.startswith('Place'):
            lo = np.array([0])
            hi = np.array([1])
            a = np.random.rand(x.shape[0], lo.shape[0]) * (hi - lo) + lo

            obj_x = x[:, 6]
            obj_y = x[:, 7]
            obj_w = x[:, 8]
            obj_h = x[:, 9]
            obj_yaw = x[:, 10]

            robby_x = x[:, 12]
            robby_y = x[:, 13]
            robby_yaw = x[:, 14]

            offset_gripper = a[:, 0]

            tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
            tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

            aux_labels = np.empty((x.shape[0], 1))
            for i in range(x.shape[0]):
                rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                aux_labels[i] = rect.distance_nearest_point(tip_x[i], tip_y[i])

        shuffle = np.random.permutation(aux_labels.shape[0])
        y_1 = aux_labels[shuffle[::2]]
        y_2 = aux_labels[shuffle[1::2]]
        if y_1.shape[0] > y_2.shape[0]:
            y_1 = y_1[1:]
        dist = np.linalg.norm(y_1 - y_2, axis=1).mean()        

        torch.save({'dist': dist}, f'data_{domain}/data/{dataset}_distance_random_samples_geometry.pt')

        if dataset.startswith('NavigateTo'):
            lo = np.array([-8, -4])
            hi = np.array([9, 5])
            a = np.random.rand(x.shape[0], lo.shape[0]) * (hi - lo) + lo

            obj_x = x[:, 0]
            obj_y = x[:, 1]
            obj_w = x[:, 2]
            obj_h = x[:, 3]
            obj_yaw = x[:, 4]

            offset_x = a[:, 0]
            offset_y = a[:, 1]
            pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
            obj_h * offset_y * np.sin(obj_yaw)
            pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                    obj_h * offset_y * np.cos(obj_yaw)

            pos_x = np.clip(pos_x, 0, 20 - 1e-6)
            pos_y = np.clip(pos_y, 0, 20 - 1e-6)

            aux_labels = np.empty((x.shape[0], 7))
            for i in range(x.shape[0]):
                rect = utils.Rectangle(obj_x[i], obj_y[i], obj_w[i], obj_h[i], obj_yaw[i])
                if dataset == 'NavigateToCup':
                    aux_labels[i, 0] = rect.line_segments[0].distance_nearest_point(pos_x[i], pos_y[i])
                elif dataset == 'NavigateToTray':
                    x1 = obj_x[i]
                    y1 = obj_y[i]
                    x2 = x1 + (obj_w[i] - obj_h[i]) * np.cos(obj_yaw[i])
                    y2 = y1 + (obj_w[i] - obj_h[i]) * np.sin(obj_yaw[i])
                    rect1 = utils.Rectangle(x1, y1, obj_h[i], obj_h[i], obj_yaw[i])
                    rect2 = utils.Rectangle(x2, y2, obj_h[i], obj_h[i], obj_yaw[i])
                    aux_labels[i, 0] = min(rect1.distance_nearest_point(pos_x[i], pos_y[i]),
                                            rect2.distance_nearest_point(pos_x[i], pos_y[i]))
                else:
                    aux_labels[i, 0] = rect.distance_nearest_point(pos_x[i], pos_y[i])
                aux_labels[i, 1], aux_labels[i, 2] = rect.nearest_point(pos_x[i], pos_y[i])
                aux_labels[i, 3], aux_labels[i, 4] = pos_x[i], pos_y[i]
                aux_labels[i, 5], aux_labels[i, 6] = rect.relative_reoriented_coordinates(pos_x[i], pos_y[i])

        elif dataset.startswith('Pick'):
            lo = np.array([0, -np.pi])
            hi = np.array([1, np.pi])
            a = np.random.rand(x.shape[0], lo.shape[0]) * (hi - lo) + lo

            book_x = x[:, 0]
            book_y = x[:, 1]
            book_w = x[:, 2]
            book_h = x[:, 3]
            book_yaw = x[:, 4]

            robby_x = x[:, 6]
            robby_y = x[:, 7]
            robby_yaw = x[:, 8]

            offset_gripper = a[:, 0]

            tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
            tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

            aux_labels = np.empty((x.shape[0], 4))
            for i in range(x.shape[0]):
                rect = utils.Rectangle(book_x[i], book_y[i], book_w[i], book_h[i], book_yaw[i])
                aux_labels[i, 0], aux_labels[i, 1] = tip_x[i], tip_y[i]
                aux_labels[i, 2], aux_labels[i, 3] = rect.relative_reoriented_coordinates(tip_x[i], tip_y[i])


        elif dataset.startswith('Place'):
            lo = np.array([0])
            hi = np.array([1])
            a = np.random.rand(x.shape[0], lo.shape[0]) * (hi - lo) + lo

            book_relative_x = x[:, 0]
            book_relative_y = x[:, 1]
            book_w = x[:, 2]
            book_h = x[:, 3]
            book_relative_yaw = x[:, 4]

            shelf_x = x[:, 6]
            shelf_y = x[:, 7]
            shelf_w = x[:, 8]
            shelf_h = x[:, 9]
            shelf_yaw = x[:, 10]

            robby_x = x[:, 12]
            robby_y = x[:, 13]
            robby_yaw = x[:, 14]

            offset_gripper = a[:, 0]

            tip_x = robby_x + (2 + offset_gripper * 2) * np.cos(robby_yaw)
            tip_y = robby_y + (2 + offset_gripper * 2) * np.sin(robby_yaw)

            place_x = tip_x + book_relative_x * np.sin(
                robby_yaw) + book_relative_y * np.cos(robby_yaw)
            place_y = tip_y + book_relative_y * np.sin(
                robby_yaw) - book_relative_x * np.cos(robby_yaw)
            place_yaw = book_relative_yaw + robby_yaw

            aux_labels = np.empty((x.shape[0], 4))
            for i in range(x.shape[0]):
                shelf_rect = utils.Rectangle(shelf_x[i], shelf_y[i], shelf_w[i], shelf_h[i], shelf_yaw[i])
                
                book_yaw = place_yaw[i]
                while book_yaw > np.pi:
                    book_yaw -= (2 * np.pi)
                while book_yaw < -np.pi:
                    book_yaw += (2 * np.pi)
                book_rect = utils.Rectangle(place_x[i], place_y[i], book_w[i], book_h[i], book_yaw)
                com_x, com_y = book_rect.center

                aux_labels[i, 0], aux_labels[i, 1] = com_x, com_y
                aux_labels[i, 2], aux_labels[i, 3] = shelf_rect.relative_reoriented_coordinates(com_x, com_y)

        # Note: I was initially normalizing these assuming that all models would have the same normalization
        # constants, but that's actually not true bc they're trained on different data sets. Therefore, I'm
        # leaving these unnormalized and using them to normalize (across "geometric tests") all losses
        # shift = np.min(aux_labels, axis=0)
        # scale = np.max(aux_labels - shift, axis=0)
        # scale = np.clip(scale, 1e-6, None)
        # aux_labels = ((aux_labels - shift) / scale) * 2 - 1

        # print(aux_labels.min(axis=0), aux_labels.max(axis=0))

        shuffle = np.random.permutation(aux_labels.shape[0])
        y_1 = aux_labels[shuffle[::2]]
        y_2 = aux_labels[shuffle[1::2]]
        if y_1.shape[0] > y_2.shape[0]:
            y_1 = y_1[1:]

        norm_vec = np.sqrt(np.mean((y_1 - y_2) ** 2, axis=0))
        torch.save({'norm_vec': norm_vec}, f'data_{domain}/data/{dataset}_distance_random_samples_geometry+.pt')



