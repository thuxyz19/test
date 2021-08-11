import numpy as np
from common.skeleton import Skeleton
h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])




subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
actions = ['Photo 1', 'Purchases 1', 'Phoning 1', 'Sitting 1',
           'SittingDown 2', 'Walking 1', 'Eating 2', 'WalkDog',
           'Sitting 2', 'Eating', 'Waiting 1', 'WalkDog 1',
           'Phoning', 'Discussion 1', 'Posing', 'Greeting 1',
           'Smoking', 'Photo', 'Waiting', 'Purchases', 'Walking',
           'SittingDown', 'Greeting', 'Directions', 'WalkTogether',
           'Discussion', 'WalkTogether 1', 'Smoking 1', 'Posing 1', 'Directions 1']


data = np.load('./data/data_3d_h36m.npz', allow_pickle=True)['positions_3d'].item()

gt = {}
for subject, actions in data.items():
    gt[subject] = {}
    for action_name, positions in actions.items():
        gt[subject][action_name] = {
            'positions': positions,
        }

# Bring the skeleton to 17 joints instead of the original 32
joints_to_remove = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]

kept_joints = h36m_skeleton.remove_joints(joints_to_remove)
for subject in gt.keys():
    for action in gt[subject].keys():
        s = gt[subject][action]
        if 'positions' in s:
            s['positions'] = s['positions'][:, kept_joints]


# Rewire shoulders to the correct parents
h36m_skeleton._parents[11] = 8
h36m_skeleton._parents[14] = 8


bone = {}

for s in subjects:
    bone[s] = 0.0
    count = 0
    for a in gt[s].keys():
        joints = gt[s][a]['positions']
        b = np.mean(np.sqrt(np.sum((joints[:, 12, :] - joints[:, 13, :]) ** 2, -1)))
        bone[s] += b
        count += 1
    bone[s] = bone[s] / count
print(bone)
