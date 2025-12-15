import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os
import heapq

# 计算余弦相似度
def calculate_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
class CrossTaskDataset(Dataset):
    def __init__(
            self,
            anot_info,
            feature_dir,
            #prompt_features,
            video_list,
            horizon=3,
            num_action=105,
            aug_range=0,
            M=2,
            mode="train",
    ):
        super().__init__()
        self.anot_info = anot_info
        self.feature_dir = feature_dir
        #self.prompt_features = prompt_features
        self.aug_range = aug_range
        self.horizon = horizon
        self.video_list = video_list
        self.max_duration = 0
        self.mode = mode
        self.M = M
        self.num_action = num_action
        self.transition_matrix = np.zeros((num_action, num_action))

        self.task_info = {"Make Jello Shots": 23521,
                          "Build Simple Floating Shelves": 59684,
                          "Make Taco Salad": 71781,
                          "Grill Steak": 113766,
                          "Make Kimchi Fried Rice": 105222,
                          "Make Meringue": 94276,
                          "Make a Latte": 53193,
                          "Make Bread and Butter Pickles": 105253,
                          "Make Lemonade": 44047,
                          "Make French Toast": 76400,
                          "Jack Up a Car": 16815,
                          "Make Kerala Fish Curry": 95603,
                          "Make Banana Ice Cream": 109972,
                          "Add Oil to Your Car": 44789,
                          "Change a Tire": 40567,
                          "Make Irish Coffee": 77721,
                          "Make French Strawberry Cake": 87706,
                          "Make Pancakes": 91515}

        self.data = []
        self.load_data()
        if self.mode == "train":
            self.transition_matrix = self.cal_transition(self.transition_matrix)

    def cal_transition(self, matrix):
        ''' Cauculate transition matrix

        Args:
            matrix:     [num_action, num_action]

        Returns:
            transition: [num_action, num_action]
        '''
        transition = matrix / np.sum(matrix, axis=1, keepdims=True)
        return transition

    def load_data(self):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)

        for video_info in video_info_dict:
            video_id = video_info["id"]["vid"]
            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]

            try:
                saved_features = \
                    np.load(os.path.join(self.feature_dir, "{}_{}.npy". \
                                         format(self.task_info[task], video_id)),
                            allow_pickle=True)["frames_features"]
                step_features = \
                    np.load(os.path.join(self.feature_dir, "{}_{}.npy". \
                                         format(self.task_info[task], video_id)),
                            allow_pickle=True)["steps_features"]
            except:
                continue

            # Remove repeated actions. Intuitively correct, but do not work well on dataset.
            # legal_video_anot = []
            # for i in range(len(video_anot)):
            #     if i == 0 or video_anot[i]["action_id"] != video_anot[i-1]["action_id"]:
            #         legal_video_anot.append(video_anot[i])
            # video_anot = legal_video_anot

            ## update transition matrix
            if self.mode == "train":
                for i in range(len(video_anot) - 1):
                    cur_action = video_anot[i]["reduced_action_id"] - 1
                    next_action = video_anot[i + 1]["reduced_action_id"] - 1
                    self.transition_matrix[cur_action, next_action] += 1

            for i in range(len(video_anot) - self.horizon + 1):
                all_features = []
                all_action_ids = []

                for j in range(self.horizon):
                    cur_video_anot = video_anot[i + j]
                    cur_action_id = cur_video_anot["reduced_action_id"] - 1
                    features = []
                    # j = 0
                    ## Using adjacent frames for data augmentation
                    for frame_offset in range(-self.aug_range, self.aug_range + 1):
                        s_time = cur_video_anot["start"] + frame_offset
                        e_time = cur_video_anot["end"] + frame_offset
                        if j == 0 or e_time - s_time + 1 <= 6:

                            if s_time < 0 or e_time >= saved_features.shape[0]:
                                continue
                            # if e_time - s_time + 1 <= 6:
                            if s_time + self.M <= saved_features.shape[0]:
                                if s_time == 0:
                                    image_start = saved_features[s_time: s_time + self.M + 1]
                                else:
                                    image_start = saved_features[s_time - 1: s_time + self.M]
                            else:
                                image_start = saved_features[saved_features.shape[0] - self.M: saved_features.shape[0]]
                            image_start_cat = image_start[0]
                            for w in range(len(image_start) - 1):
                                image_start_cat = np.concatenate((image_start_cat, image_start[w + 1]), axis=0)
                                # image_start_cat = np.mean((image_start_cat, image_start[w + 1]), axis=0)
                            # features.append(image_start_cat)

                            e_time = max(2, e_time)
                            if e_time >= saved_features.shape[0] - 1:
                                image_end = saved_features[e_time - 2:e_time + self.M - 1]
                            else:
                                image_end = saved_features[e_time - 1:e_time + self.M]
                            # image_end = saved_features[e_time - 2:e_time + self.M - 1]
                            image_end_cat = image_end[0]
                            for w in range(len(image_end) - 1):
                                image_end_cat = np.concatenate((image_end_cat, image_end[w + 1]), axis=0)
                                # image_end_cat = np.mean((image_end_cat, image_end[w + 1]), axis=0)
                            # features.append(image_end_cat)
                            features.append(np.stack((image_start_cat, image_end_cat)))
                        else:

                            s_time = cur_video_anot["start"] + frame_offset
                            e_time = cur_video_anot["end"] + frame_offset

                            if s_time < 0 or e_time >= saved_features.shape[0]:
                                continue
                            # if e_time - s_time + 1 <= 6:
                            cosine = []
                            for j in range(s_time, e_time):
                                cosine.append(calculate_cosine_similarity(step_features[i], saved_features[j]))
                            indices = heapq.nlargest(2 * self.horizon, range(len(cosine)), cosine.__getitem__)
                            indices = sorted(indices)
                            # interval = (image_end_idx1 - image_start_idx1 + 1) // 6
                            # print(interval,image_start_idx1,image_end_idx1)
                            image_start_cat = saved_features[s_time + indices[0]]
                            # for w in range(len(image_start) - 1):
                            image_start_cat = np.concatenate((image_start_cat,
                                                              saved_features[s_time + indices[1]],
                                                              saved_features[s_time + indices[2]]), axis=0)
                            # image_start_cat = np.mean((image_start_cat, image_start[w + 1]), axis=0)
                            # features.append(image_start_cat)

                            e_time = max(2, e_time)
                            if e_time >= saved_features.shape[0] - 1:
                                image_end = saved_features[e_time - 2:e_time + self.M - 1]
                            else:
                                image_end = saved_features[e_time - 1:e_time + self.M]
                            # image_end = saved_features[e_time - 2:e_time + self.M - 1]
                            image_end_cat = image_end[0]
                            for w in range(len(image_end) - 1):
                                image_end_cat = np.concatenate((image_end_cat, image_end[w + 1]), axis=0)
                                # image_end_cat = np.mean((image_end_cat, image_end[w + 1]), axis=0)
                            # features.append(image_end_cat)
                            features.append(np.stack((image_start_cat, image_end_cat)))
                        

                    all_features.append(features)
                    all_action_ids.append(cur_action_id)

                task_id = cur_video_anot["task_id"]

                ## permutation of augmented features, action ids and prompts
                aug_features = itertools.product(*all_features)

                self.data.extend([{"states": np.stack(f),
                                   "actions": np.array(all_action_ids),
                                   "tasks": np.array(task_id)}
                                  for f in aug_features])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = self.data[idx]["states"]
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions,
                                                                             dtype=torch.long), torch.as_tensor(tasks,
                                                                                                                dtype=torch.long)