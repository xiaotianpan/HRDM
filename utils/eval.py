from .accuracy import *
from model.helpers import AverageMeter,success_rate,mean_category_acc,acc_iou,viterbi_path
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
def calculate_cosine_similarity(vector1, vector2):
    if vector1.is_cuda:
        vector1 = vector1.cpu().numpy()
    if vector2.is_cuda:
        vector2 = vector2.cpu().numpy()
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

Output1 = namedtuple('Output1', 'output')
def validate(val_loader, model, args,epoch,save_max_output,transition_matrix):
    model.eval()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    trajectory_success_rate_meter = AverageMeter()
    MIoU1_meter = AverageMeter()
    MIoU2_meter = AverageMeter()

    A0_acc = AverageMeter() #acc1
    AT_acc = AverageMeter() #task class

    Cosine = AverageMeter()

    viterbi_sr = AverageMeter()
    viterbi_acc1 = AverageMeter()
    viterbi_miou = AverageMeter()

    j = 0
    abc = []
    for i_batch, sample_batch in enumerate(val_loader):

        video_label = sample_batch[1].cuda()
        batch_size_current, T = video_label.size()
        sample_batch[0] = sample_batch[0].reshape(batch_size_current, T * 2, 1536)
        # compute output
        global_img_tensors = sample_batch[0].cuda().contiguous().float()
        task_class = sample_batch[2].view(-1).cuda()
        cond = {}
        cond0 = {}
        with torch.no_grad():
            if epoch > 200:
                max_output = save_max_output['output'][j][0]
                max_output = torch.from_numpy(max_output).cuda()

                cond[0] = global_img_tensors[:, 0, :]
                cond[1] = max_output[:, 1, args.class_dim + args.action_dim:].float()
                cond[T - 1] = global_img_tensors[:, -1, :]
                cond['task'] = max_output[:, :, :args.class_dim].float()
                cond['mid_s'] = max_output[:, 0, args.class_dim:args.class_dim + args.action_dim].float()
                cond['mid_g'] = max_output[:, -1, args.class_dim:args.class_dim + args.action_dim].float()
                cond['actions'] = max_output[:, :, args.class_dim:args.class_dim + args.action_dim].float()
            else:

                cond0[0] = global_img_tensors[:, 0, :]
                cond0[T - 1] = global_img_tensors[:, -1, :]




            task_onehot = torch.zeros((task_class.size(0), args.class_dim))  # [bs*T, ac_dim]
            ind = torch.arange(0, len(task_class))
            task_onehot[ind, task_class] = 1.
            task_onehot = task_onehot.cuda()
            temp = task_onehot.unsqueeze(1)
            task_class_ = temp.repeat(1, T, 1)  # [bs, T, args.class_dim]


            video_label_reshaped = video_label.view(-1)

            action_label_onehot = torch.zeros((video_label_reshaped.size(0), args.action_dim))
            ind = torch.arange(0, len(video_label_reshaped))
            action_label_onehot[ind, video_label_reshaped] = 1.
            action_label_onehot = action_label_onehot.reshape(batch_size_current, T, -1).cuda()

            x_start = torch.zeros((batch_size_current, T, args.class_dim + args.action_dim + args.observation_dim))
            x_start[:, 0, args.class_dim + args.action_dim:] = global_img_tensors[:, 0, :]
            x_start[:, 1, args.class_dim + args.action_dim:] = global_img_tensors[:, 2, :]
            x_start[:, -1, args.class_dim + args.action_dim:] = global_img_tensors[:, -1, :]


            x_start[:, :, args.class_dim:args.class_dim + args.action_dim] = action_label_onehot
            x_start[:, :, :args.class_dim] = task_class_
            if epoch > 200:
                output = model(cond, epoch)
            else:
                output = model(cond0, epoch)

            actions_pred = output.contiguous()
            loss = model.module.loss_fn(actions_pred, x_start.cuda(),epoch,video_label_reshaped,cond)
            a = actions_pred[:, 1:-1, args.class_dim + args.action_dim:]
            b = x_start[:, 1:-1, args.class_dim + args.action_dim:].cuda()

            cosine = F.cosine_similarity(a, b, dim=2)
            cosine = cosine.sum()

            actions_pred = actions_pred[:, :, args.class_dim:args.class_dim + args.action_dim].contiguous()
            actions_pred = actions_pred.view(-1, args.action_dim)  # [bs*T, action_dim]

            (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc= \
                accuracy(actions_pred.cpu(), video_label_reshaped.cpu(), topk=(1, 5), max_traj_len=args.horizon,dfg=False)



            task_pred = output[:, :, :args.class_dim].contiguous()

            task_pred = torch.argmax(task_pred, dim=2)

            task_pred = task_pred[:, 0]


            correct = task_pred.eq(task_class.view(-1))
            task_acc = torch.sum(correct) / batch_size_current * 100




            output1 = output[:, :, :].contiguous()
            output1 = output1.cpu()
            output1 = output1.numpy()
            abc.append(Output1(output=output1))

            viterbi_pre = output[:, :, args.class_dim:args.class_dim + args.action_dim].contiguous()
            if transition_matrix is not None:
                pred_viterbi = []
                for i in range(batch_size_current):
                    viterbi_rst = viterbi_path(transition_matrix,
                                               viterbi_pre[i].permute(1, 0).detach().cpu().numpy())
                    pred_viterbi.append(torch.from_numpy(viterbi_rst))
                pred_viterbi = torch.stack(pred_viterbi).cuda()
            else:
                pred_viterbi = None

            output_viterbi = pred_viterbi.cpu().numpy()
            labels_viterbi = video_label.reshape(batch_size_current, -1).cpu().numpy().astype("int")
            sr_viterbi = success_rate(output_viterbi, labels_viterbi, True)
            miou_viterbi = acc_iou(output_viterbi, labels_viterbi, False).mean()
            acc_viterbi = mean_category_acc(output_viterbi, labels_viterbi)
            trajectory_success_rate =sr_viterbi
            MIoU1_v = miou_viterbi
            acc1_v = acc_viterbi




        j = j + 1
        losses.update(loss.item(), batch_size_current)
        acc_top1.update(acc1_v.item(), batch_size_current)
        acc_top5.update(acc5.item(), batch_size_current)
        trajectory_success_rate_meter.update(trajectory_success_rate.item(), batch_size_current)
        MIoU1_meter.update(MIoU1_v, batch_size_current)
        MIoU2_meter.update(MIoU2, batch_size_current)
        Cosine.update(cosine, batch_size_current)
        A0_acc.update(acc1.item(), batch_size_current)

        AT_acc.update(task_acc, batch_size_current)



    return torch.tensor(losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg), \
        torch.tensor(trajectory_success_rate_meter.avg), \
        torch.tensor(MIoU1_meter.avg), torch.tensor(MIoU2_meter.avg), \
        torch.tensor(A0_acc.avg), torch.tensor(AT_acc.avg), torch.tensor(Cosine.avg), abc#, \
        