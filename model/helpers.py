import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
import logging
from tensorboardX import SummaryWriter


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 2, 1, 0)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 2, 1, 0)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=32, drop_out=0.0, if_zero=False):
        super().__init__()
        if drop_out > 0.0:
            self.block = nn.Sequential(
                zero_module(
                    nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                ),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
                nn.Dropout(p=drop_out),
            )
        elif if_zero:
            self.block = nn.Sequential(
                zero_module(
                    nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                ),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),

            )
        else:
            self.block = nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def condition_projection(x, conditions, action_dim, class_dim, abc = False):
    if not abc:
        for t, val in conditions.items():
            if t != 'task':
                x[:, t, class_dim + action_dim:] = val.clone()

        #x[:, 1:-1, class_dim + action_dim:] =  0.
        x[:, :, :class_dim] = conditions['task']
    else:
        for t, val in conditions.items():
            if t != 'task' and t != 'mid_s'and t != 'mid_g':
                x[:, t, class_dim + action_dim:] = val.clone()

        x[:, 1:-1, class_dim + action_dim:] = 0.
        x[:, :, :class_dim] = conditions['task']
        x[:, 0, class_dim:class_dim + action_dim] = conditions['mid_s']
        x[:, -1, class_dim:class_dim + action_dim] = conditions['mid_g']

    return x



def mask_topk_per_sample(x: torch.Tensor,y:torch.Tensor, k: int = 5, xyz: bool = False) -> torch.Tensor:
    """
    保留每个样本中前k大的元素，其他位置置零
    Args:
        x: 输入张量，形状为 (batch_size, ...)
        k: 保留的最大元素数量
    Returns:
        masked_x: 掩码后的张量
    """
    if xyz ==True:
        batch_size, features = y.shape
    else:
        batch_size, seq_len, features = y.shape
    # 展平除batch维度外的所有维度
    x_flat = x.view(-1,features)
    y_flat = y.view(-1,features)
    # 获取每个样本的topk索引（按值从大到小排列）
    _, indices = torch.topk(y_flat, k=k, dim=1)
    # 生成布尔掩码
    mask_flat = torch.zeros_like(y_flat, dtype=torch.bool)
    mask_flat.scatter_(1, indices, True)
    # 恢复原始形状并与输入相乘
    mask = mask_flat.view_as(x)
    return x * mask

def mask_topk_per_sample0(x: torch.Tensor, k: int = 5,xyz: bool = False) -> torch.Tensor:
    """
    保留每个样本中前k大的元素，其他位置置零
    Args:
        x: 输入张量，形状为 (batch_size, ...)
        k: 保留的最大元素数量
    Returns:
        masked_x: 掩码后的张量
    """
    if xyz == True:
        batch_size, features = x.shape
    else:
        batch_size, seq_len, features = x.shape
    # 展平除batch维度外的所有维度
    x_flat = x.view(-1, features)
    # 获取每个样本的topk索引（按值从大到小排列）
    _, indices = torch.topk(x_flat, k=k, dim=1)
    # 生成布尔掩码
    mask_flat = torch.zeros_like(x_flat, dtype=torch.bool)
    mask_flat.scatter_(1, indices, True)
    # 恢复原始形状并与输入相乘
    mask = mask_flat.view_as(x)
    return x * mask




def condition_projection1(x, conditions, action_dim, class_dim, epoch,bcd):
    if epoch<=200:


        x[:, 0, class_dim + action_dim:] = conditions[0]
        x[:, -1, class_dim + action_dim:] = conditions[2]
        x[:, :, class_dim :class_dim+ action_dim] = 0

    elif epoch>200 and epoch<=700:
        x[:, 0, class_dim + action_dim:] = conditions[0]
        x[:, 1, class_dim + action_dim:] = conditions[1]

        x[:, -1, class_dim + action_dim:] = conditions[2]

        x[:, :, :class_dim] = conditions['task']

    else :#epoch>=800 and epoch<1200:


        x[:, 0, class_dim + action_dim:] = conditions[0]
        x[:, 1, class_dim + action_dim:] = conditions[1]

        x[:, -1, class_dim + action_dim:] = conditions[2]

        x[:, :, :class_dim] = conditions['task']
       
        if bcd == True:
            x[:, :, class_dim:class_dim + action_dim] = mask_topk_per_sample(x[:, :, class_dim:class_dim + action_dim],conditions['actions'], k=10)
        else:
            x[:, :, class_dim:class_dim + action_dim] = mask_topk_per_sample0(conditions['actions'], k=10)
        x[:, 0, class_dim:class_dim + action_dim] = conditions['mid_s']
        x[:, -1, class_dim:class_dim + action_dim] = conditions['mid_g']
    return x



# -----------------------------------------------------------------------------#
# ---------------------------------- Loss -------------------------------------#
# -----------------------------------------------------------------------------#
def calculate_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity




class NormalizedCosineLoss(nn.Module):
    def __init__(self):
        super(NormalizedCosineLoss, self).__init__()

    def forward(self, x1, x2):
        # 对输入进行L2归一化
        x1_normalized = F.normalize(x1, p=2, dim=-1)
        x2_normalized = F.normalize(x2, p=2, dim=-1)

        # 计算余弦相似度（归一化后的点积）
        cosine_similarity = (x1_normalized * x2_normalized).sum(dim=-1)

        # 计算损失：1 - 平均余弦相似度
        loss = 1-cosine_similarity.mean()
        return loss#.sum()


class NormalizedCosineLoss1(nn.Module):
    def __init__(self):
        super(NormalizedCosineLoss1, self).__init__()

    def forward(self, x1, x2,y1,y2):
        # 对输入进行L2归一化
        x1_normalized = F.normalize(x1, p=2, dim=-1)
        x2_normalized = F.normalize(x2, p=2, dim=-1)
        y1_normalized = F.normalize(y1, p=2, dim=-1)
        y2_normalized = F.normalize(y2, p=2, dim=-1)

        # 计算余弦相似度（归一化后的点积）
        cosine_similarity1 = (x1_normalized * y1_normalized).sum(dim=-1)
        cosine_similarity2 = (x2_normalized * y2_normalized).sum(dim=-1)


        # 计算损失：1 - 平均余弦相似度
        loss = 1 - cosine_similarity1.sum()/cosine_similarity2.sum()
        return loss

class replanloss(nn.Module):
    def __init__(self):
        super(replanloss, self).__init__()

    def forward(self, x1, x2,x3):
        # 对输入进行L2归一化
        x1_normalized = F.normalize(x1, p=2, dim=-1)
        x2_normalized = F.normalize(x2, p=2, dim=-1)
        x3_normalized = F.normalize(x3, p=2, dim=-1)

        # 计算余弦相似度（归一化后的点积）
        cosine_similarity1 = (x1_normalized * x2_normalized).sum(dim=-1)
        cosine_similarity2 = (x3_normalized * x2_normalized).sum(dim=-1)#整体固定，样本不同，对应的值不同
        cosine_similarity3 = (x1_normalized * x3_normalized).sum(dim=-1)


        # 计算损失：1 - 平均余弦相似度
        loss = 1 - cosine_similarity2.sum()/cosine_similarity1.sum()
        return loss


class Weighted_MSE(nn.Module):

    def __init__(self, weights, action_dim, class_dim):
        super().__init__()
        # self.register_buffer('weights', weights)
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.loss_fn = NormalizedCosineLoss()
        self.loss_fn1 = NormalizedCosineLoss1()
        self.loss_cond = replanloss()

    def forward(self, pred, targ,epoch,video_label,cond):
        """
        :param pred: [B, T, task_dim+action_dim+observation_dim]
        :param targ: [B, T, task_dim+action_dim+observation_dim]
        :return:
        """

        
        if epoch<=200:

          



            pred0 = pred[:, :, self.class_dim + self.action_dim:].reshape(-1, 1536)
            targ0 = targ[:, :, self.class_dim + self.action_dim:].reshape(-1, 1536)
            loss_ob=self.loss_fn(pred0,targ0)

            loss_action = F.mse_loss(pred, targ, reduction='none')
            loss_action[:, :, self.class_dim + self.action_dim:] *= 10.
            # loss_action[:, 0, self.class_dim:self.class_dim + self.action_dim] *= 10.
            # loss_action[:, -1, self.class_dim:self.class_dim + self.action_dim] *= 10.
            loss_action = loss_action.sum()
            loss_action = loss_action+loss_ob*10


        elif epoch>200 and epoch<=700:
            pred0 = pred[:, :, self.class_dim:self.class_dim + self.action_dim].reshape(-1, self.action_dim)
            pred1 = pred[:, :, self.class_dim + self.action_dim:].reshape(-1, 1536)
            targ0 = targ[:, :, self.class_dim:self.class_dim + self.action_dim].reshape(-1, self.action_dim)
            targ1 = targ[:, :, self.class_dim + self.action_dim:].reshape(-1, 1536)
            loss_ob1 = self.loss_fn1(pred0, pred1, targ0, targ1)

            loss_action = F.mse_loss(pred, targ, reduction='none')
            #loss_action[:, :, self.class_dim + self.action_dim:] *= 10.
            loss_action[:, 0, self.class_dim:self.class_dim + self.action_dim] *= 10.
            loss_action[:, -1, self.class_dim:self.class_dim + self.action_dim] *= 10.
            loss_action = loss_action.sum()+loss_ob1*10
           
        else:
            
            pred0 = pred[:, :, self.class_dim:self.class_dim + self.action_dim].reshape(-1, self.action_dim)
            pred1 = pred[:, :, self.class_dim + self.action_dim:].reshape(-1, 1536)
            targ0 = targ[:, :, self.class_dim:self.class_dim + self.action_dim].reshape(-1, self.action_dim)
            targ1 = targ[:, :, self.class_dim + self.action_dim:].reshape(-1, 1536)
            loss_ob1 = self.loss_fn1(pred0, pred1, targ0, targ1)


            loss_action = F.mse_loss(pred, targ, reduction='none')
            loss_action[:, 1:-1, self.class_dim:self.class_dim + self.action_dim] *= 10.
            
            loss_action = loss_action.sum()

           

            pred = pred[:, :, self.class_dim:self.class_dim + self.action_dim]
            targ = targ[:, :, self.class_dim:self.class_dim + self.action_dim]

            probs = F.softmax(pred, dim=-1)
            conds = F.softmax(cond['actions'], dim=-1)

            true_indices = targ.argmax(dim=-1)  # (batch_size, horizon)


            pred_indices = probs.argmax(dim=-1)  # (batch_size, horizon)
            cond_indices = conds.argmax(dim=-1)  # (batch_size, horizon)

            cond_diffs_1 = torch.abs(cond_indices - true_indices).float()
            cond_diffs_1 = (cond_diffs_1 != 0).int()


            cond_diffs_2 = torch.abs(cond_indices - true_indices).float()
            cond_diffs_2 = (cond_diffs_2 == 0).int()  ##A

            pos_diffs = torch.abs(pred_indices - true_indices).float()
            pos_diffs = (pos_diffs == 0).int()  ##B
            mask_flat = torch.zeros_like(pos_diffs, dtype=torch.bool)

            cond_a = cond_diffs_2 & ((1 - pos_diffs))
            targ_a = (1 - pos_diffs) & (1 - cond_diffs_2)  # (1-pos_diffs | cond_diffs_2))


            fp_loss_1 = self.loss_fn(targ_a.unsqueeze(-1) * pred, targ)
            fp_loss_2 = self.loss_fn(cond_a.unsqueeze(-1) * pred, cond['actions'])


            #fp_loss = self.loss_fn(pos_diffs.unsqueeze(-1) * pred, targ)
            


            loss_action =loss_action+fp_loss_1*1+ fp_loss_2*1
            

        return loss_action


Losses = {
    'Weighted_MSE': Weighted_MSE,
}

# -----------------------------------------------------------------------------#
# -------------------------------- lr_schedule --------------------------------#
# -----------------------------------------------------------------------------#

def get_lr_schedule_with_warmup(optimizer, num_training_steps, last_epoch=-1):
    num_warmup_steps = num_training_steps * 20 / 120
    decay_steps = num_training_steps * 30 / 120

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return max(0., float(current_step) / float(max(1, num_warmup_steps)))
        else:
            return max(0.5 ** ((current_step - num_warmup_steps) // decay_steps), 0.)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -----------------------------------------------------------------------------#
# ---------------------------------- logging ----------------------------------#
# -----------------------------------------------------------------------------#

# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=SummaryWriter, if_exist=False):
        self._log_dir = log_dir
        print('logging outputs to ', log_dir)
        self._n_logged_samples = n_logged_samples
        self._summ_writer = summary_writer(log_dir, flush_secs=120, max_queue=10)
        if not if_exist:
            log = logging.getLogger(log_dir)
            if not log.handlers:
                log.setLevel(logging.DEBUG)
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
                fh.setFormatter(formatter)
                log.addHandler(fh)
            self.log = log

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def flush(self):
        self._summ_writer.flush()

    def log_info(self, info):
        self.log.info("{}".format(info))

    def log_info1(self, value, key, epoch):
        self.log.info("Epoch: {} key: {} value: {:.2f}%" \
                      .format(epoch, key, value))


class ValueInit(nn.Module):
    """ValueMulti."""

    def __init__(self, img_shape):
        super(ValueInit, self).__init__()
        self.lin1 = nn.Linear(int(np.prod(img_shape)) + 768, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, 256)
        self.lin4 = nn.Linear(256, 1)
        torch.nn.init.xavier_normal_(self.lin4.weight)

    def forward(self, img, txt_emb):
        # pdb.set_trace()
        x = img.view(img.shape[0], -1)

        x = torch.cat([x, txt_emb], dim=1)
        # x = torch.cat([x, txt_emb], dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)



def success_rate(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: The predicted intermediate action sequence, numpy [batch, seq];
    gt  : The ground-truth action label sequence    , numpy [batch, seq];

    Metric Procedure:
    "All" prediction steps has to match with gt steps
    """
    rst = np.all(np.equal(pred, gt), axis=(1))

    if aggregate:
        return np.mean(rst) * 100
    else:
        return rst


def mean_category_acc(pred, gt):
    """required format
    Action space is a single integer
    pred: List [batch * seq]
    gt  : List [batch * seq]
    """
    # rst = precision_score(gt, pred, average="macro", zero_division=0)
    rst = (gt == pred).mean() * 100
    return rst


def acc_iou(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: Numpy [batch, seq]
    gt  : Numpy [batch, seq]
    """
    epsn = 1e-6

    if aggregate:
        intersection = (pred & gt).sum((0, 1))
        union = (pred | gt).sum((0, 1))
    else:
        intersection = (pred & gt).sum((1))
        union = (pred | gt).sum((1))

    return (intersection + epsn) / (union + epsn) * 100


def viterbi_path(transition, emission, prior=None, observation=None, return_likelihood=False):
    ''' Viterbi algorithm

    Search the most likely sequence of hidden states given the observations.

    Args:
        transition:     Transition matrix, where A[i][j] is the probability of
                        transitioning from state i to state j.  (num_action, num_action)
        emission:       Emission matrix, where B[i][j] is the probability of
                        emitting observation j from state i.    (num_action, horizon)
        prior:          Prior probabilities, where pi[i] is the probability of
                        starting in state i.    (num_action)
        observation:    Sequence of observations.   (horizon)
        return_likelihood:  Whether to return the likelihood of the best path.  (default: False)

    Returns:
        best_path:      The most likely action sequence.    (horizon)
        best_path_prob: The likelihood of the best path.
    '''

    # Initialize trellis
    T = emission.shape[1]  # time horizon
    N = transition.shape[0]  # number of actions

    if observation is None:
        observation = np.arange(T)

    if prior is None:
        prior = np.ones((N,), dtype=np.float32) / N

    trellis = np.zeros((T, N), dtype=np.float32)  # store the probabilities of each state at each time step
    backpointers = np.zeros((T, N),
                            dtype=np.int32)  # store the indices of the most likely previous state at each time step

    # Calculate probabilities for first time step
    trellis[0] = prior * emission[:, observation[0]]

    # Calculate probabilities for subsequent time steps
    for t in range(1, T):
        temp = trellis[t - 1].reshape((N, 1)) * transition
        trellis[t] = emission[:, observation[t]] * np.max(temp, axis=0)
        backpointers[t] = np.argmax(temp, axis=0)

    # Backtrack to find most likely sequence of hidden states
    best_path_prob = np.max(trellis[-1])
    best_path_pointer = np.argmax(trellis[-1])
    best_path = [best_path_pointer]
    for t in range(T - 1, 0, -1):
        best_path_pointer = backpointers[t][best_path_pointer]
        best_path.insert(0, best_path_pointer)

    best_path = np.array(best_path)

    if return_likelihood:
        return best_path, best_path_prob
    else:
        return best_path


class MLA(nn.Module):
    def __init__(self, d_model=512, down_dim=128, up_dim=128, num_heads=8, rope_head_dim=64, dropout_prob=0.2):
        super(MLA, self).__init__()

        self.d_model = d_model
        self.down_dim = down_dim
        self.up_dim = up_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = up_dim // num_heads
        # 初始化kv联合以及q对应的dow,up projection
        self.down_proj_kv = nn.Linear(d_model, down_dim)  # W^{DKV}
        self.up_proj_k = nn.Linear(down_dim, up_dim)  # W^{UK}
        self.up_proj_v = nn.Linear(down_dim, up_dim)  # W^{UV}
        self.down_proj_q = nn.Linear(d_model, down_dim)  # W^{DQ}
        self.up_proj_q = nn.Linear(down_dim, up_dim)  # W^{UQ}
        # 初始化解耦的q,k进行MQA计算的映射矩阵
        self.proj_qr = nn.Linear(down_dim, rope_head_dim * num_heads)
        self.proj_kr = nn.Linear(d_model, rope_head_dim * 1)
        # 初始化解耦的q,k对应的rope类，因为头的数量不同，初始化2个实例
        self.rope_q = RotaryEmbedding(rope_head_dim * num_heads, num_heads)
        self.rope_k = RotaryEmbedding(rope_head_dim, 1)
        # Dropout and final linear layer
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_heads * self.v_head_dim, d_model)
        self.res_dropout = nn.Dropout(dropout_prob)

    def forward(self, h, mask=None):
        bs, seq_len, _ = h.size()
        # setp1 :低秩转换
        c_t_kv = self.down_proj_kv(h)
        k_t_c = self.up_proj_k(c_t_kv)
        v_t_c = self.up_proj_v(c_t_kv)
        c_t_q = self.down_proj_q(h)
        q_t_c = self.up_proj_q(c_t_q)

        # step2:解耦的q,k进行MQA计算，同时引入ROPE
        # q_t_r,k_t_r施加rope时均扩展了n_h_r维度->[bs,n_h_r,seq_len,rope_head_dim]
        q_t_r = self.rope_q(self.proj_qr(c_t_q),seq_len)
        k_t_r = self.rope_k(self.proj_kr(h),seq_len)

        # step3:拼接step1，step2得到的q,k,进行sdpa计算
        # q_t_c扩展出num_heads为4维，以便于和q_t_r拼接
        q_t_c = q_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        # head_dim,rope_head_dim拼接
        q = torch.cat([q_t_c, q_t_r], dim=-1)
        # k_t_c扩展出num_heads为4维，以便于和k_t_r拼接
        k_t_c = k_t_c.reshape(bs, seq_len, self.num_heads, -1).transpose(1, 2)
        # k_t_r为MQA,n_h_k_r=1,为了和q_t_r计算，需要在n_h_k_r维度复制
        # k_t_r:[bs,n_h_r_k,seq_len,rope_head_dim]->[bs,num_heads,seq_len,rope_head_dim]
        k_t_r = k_t_r.repeat(1, self.num_heads, 1, 1)
        # head_dim,rope_head_dim拼接
        k = torch.cat([k_t_c, k_t_r], dim=-1)
        # 注意力计算,[bs,num_heads,seq_len,seq_len]
        scores = torch.matmul(q, k.transpose(-1, -2))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores / (math.sqrt(self.head_dim) + math.sqrt(self.rope_head_dim)), dim=-1)
        scores = self.dropout(scores)
        # v_t_c和scores计算，扩展出num_heads维度
        v_t_c = v_t_c.reshape(bs, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)
        output = torch.matmul(scores, v_t_c)
        # 压缩num_head,送入最终统一映射层
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.fc(output)
        output = self.res_dropout(output)
        return output





class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, num_heads, base=10000, max_len=512):
        super().__init__()
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.base = base
        self.max_len = max_len
        # 初始化时计算一次
        self.cos_pos_cache, self.sin_pos_cache = self._compute_pos_emb()

    def _compute_pos_emb(self):
        # theta_i=1/(10000^(2i/head_dim))i的范围是[0,head_dim//2]
        theta_i = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        # 根据最大长度创建位置索引序列，元素m对应第 m个位置
        positions = torch.arange(self.max_len)
        # max_len个位置对应m*theta->[max_len,head_dim//2]
        pos_emb = positions.unsqueeze(1) * theta_i.unsqueeze(0)
        # cos相邻位置复制一次，eg：123-》112233
        cos_pos = pos_emb.sin().repeat_interleave(2, dim=-1)
        # sin相邻位置复制一次,[max_len,head_dim]
        sin_pos = pos_emb.cos().repeat_interleave(2, dim=-1)

        return cos_pos, sin_pos

    def forward(self, q,seq_len):
        bs, q_len = q.shape[0], q.shape[1]
        self.cos_pos = self.cos_pos_cache[:q_len].cuda()
        self.sin_pos = self.sin_pos_cache[:q_len].cuda()
        # q压缩出num_heads以便于在head_dim上施加rope位置编码
        q = q.reshape(bs, q_len, self.num_heads, -1).transpose(1, 2)

        # repeat沿着指定位置复制，bs,num_head纬度上复制以便和q,k计算，其他纬度为1不复制->[bs,num_heads,max_len,head_dim]
        self.cos_pos = self.cos_pos.repeat(bs, self.num_heads, *([1] * len(self.cos_pos.shape))).cuda()
        self.sin_pos = self.sin_pos.repeat(bs, self.num_heads, *([1] * len(self.sin_pos.shape))).cuda()

        # 有了sin_pos,还需对q进行负奇，偶交替处理,先抽取q的奇偶元素再stack扩展最后一个维度让奇偶相邻
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).cuda()
        # 再reshape压缩最后一个维度实现负奇，偶交替
        q2 = q2.reshape(bs, self.num_heads, seq_len, -1).cuda()
        # q与位置编码相乘
        r_q = q * self.cos_pos + q2 * self.sin_pos

        return r_q
