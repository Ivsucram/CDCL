from itertools import cycle
import torch
import numpy as np
import tqdm

class CenterAwarePseudoModule(torch.nn.Module):

    def __init__(self, model, loader_target, loader=None, distance='cosine', threshold=0., task=0, args=None):
        super(CenterAwarePseudoModule, self).__init__()

        self.args = args

        model.eval()
        
        counter = 0
        with torch.no_grad():
            if loader is None:
                for n_iter, (input, _, _) in enumerate(tqdm.tqdm(loader_target)):
                    input = input.cuda()
                    injection_outputs, accumulator_outputs, feas, _, _ = model(input, task=task)

                    if n_iter == 0:
                        all_fea = torch.zeros(len(loader_target.dataset), feas.flatten(1).size(1))
                        all_accumulator_output = torch.zeros(len(loader_target.dataset), accumulator_outputs.size(1))
                        all_injection_output = torch.zeros(len(loader_target.dataset), injection_outputs.size(1))
                    
                    for _, (injection_output, accumulator_output, fea) in enumerate(zip(injection_outputs, accumulator_outputs, feas)):
                        all_fea[counter] = fea.flatten().detach().clone().float().cpu()
                        all_accumulator_output[counter] = accumulator_output.clone().detach().float().cpu()
                        all_injection_output[counter] = injection_output.clone().detach().float().cpu()
                        counter += 1
            else:
                for n_iter, (input, _, _) in enumerate(tqdm.tqdm(loader)):
                    input = input.cuda()
                    injection_outputs, accumulator_outputs, feas, _, _ = model(input, return_features=True, task=task)

                    if n_iter == 0:
                        all_fea = torch.zeros(len(loader_target.dataset) + len(loader.dataset), feas.flatten(1).size(1))
                        all_accumulator_output = torch.zeros(len(loader_target.dataset) + len(loader.dataset), accumulator_outputs.size(1))
                        all_injection_output = torch.zeros(len(loader_target.dataset) + len(loader.dataset), accumulator_outputs.size(1))

                    for _, (injection_output, accumulator_output, fea) in enumerate(zip(injection_outputs, accumulator_outputs, feas)):
                        all_fea[counter] = fea.flatten().detach().clone().float().cpu()
                        all_accumulator_output[counter] = accumulator_output.clone().detach().float().cpu()
                        all_injection_output[counter] = injection_output.clone().detach().float().cpu()
                        counter += 1

                for _, (input, _, _) in enumerate(tqdm.tqdm(loader_target)):
                    input = input.cuda()
                    injection_outputs, accumulator_outputs, feas, _, _ = model(input, return_features=True, task=task)

                    for _, (injection_output, accumulator_output, fea) in enumerate(zip(injection_outputs, accumulator_outputs, feas)):
                        all_fea[counter] = fea.flatten().detach().clone().float().cpu()
                        all_accumulator_output[counter] = accumulator_output.clone().detach().float().cpu()
                        all_injection_output[counter] = injection_output.clone().detach().float().cpu()
                        counter += 1

            all_accumulator_output = torch.nn.Softmax(dim=1)(all_accumulator_output)
            all_injection_output = torch.nn.Softmax(dim=1)(all_injection_output)
            _, accumulator_predict = all_accumulator_output.max(1)
            _, injection_predict = all_injection_output.max(1)

            if distance == 'cosine':
                all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
                all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

            all_fea = all_fea.float().cpu()
            accumulator_K = all_accumulator_output.size(1)
            accumulator_aff = all_accumulator_output.float().cpu()
            injection_K = all_injection_output.size(1)
            injection_aff = all_injection_output.float().cpu()

            accumulator_initc = accumulator_aff.t().matmul(all_fea)
            accumulator_initc = accumulator_initc / (1e-8 + accumulator_aff.sum(axis=0)[:, None])
            injection_initc = injection_aff.t().matmul(all_fea)
            injection_initc = injection_initc / (1e-8 + injection_aff.sum(axis=0)[:, None])
            accumulator_cls_count = torch.eye(accumulator_K)[accumulator_predict].sum(axis=0)
            injection_cls_count = torch.eye(injection_K)[injection_predict].sum(axis=0)
            accumulator_labelset = np.where(accumulator_cls_count > threshold)
            accumulator_labelset = accumulator_labelset[0]
            accumulator_labelset = torch.LongTensor(accumulator_labelset)
            injection_labelset = np.where(injection_cls_count > threshold)
            injection_labelset = injection_labelset[0]
            injection_labelset = torch.LongTensor(injection_labelset)

            self.accumulator_labelset = accumulator_labelset
            self.injection_labelset = injection_labelset
            self.accumulator_initc = accumulator_initc.cuda()
            self.injection_initc = injection_initc.cuda()

    def forward(self, model, x, distance='cosine', task=0):
        model.eval()
        with torch.no_grad():
            _, _, all_fea, _, _ = model(x, task=task)
            all_fea = all_fea.flatten(1)
            if distance == 'cosine':
                all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).cuda()), 1)
                all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

            accumulator_dd = torch.cdist(all_fea, self.accumulator_initc[self.accumulator_labelset])
            accumulator_pred_label = accumulator_dd.argmin(axis=1)
            accumulator_pred_label = self.accumulator_labelset[accumulator_pred_label]

            injection_dd = torch.cdist(all_fea, self.injection_initc[self.injection_labelset])
            injection_pred_label = injection_dd.argmin(axis=1)
            injection_pred_label = self.injection_labelset[injection_pred_label]

            return injection_pred_label, accumulator_pred_label

    def reorder_datasets(self, model, x_s, y_s, x_t, task=0):
        model.eval()
        with torch.no_grad():
            _, C, H, W = x_s.size()

            inj_y_t, acc_y_t = self(model, x_t, task=task)
            if len(y_s.size()) > 1:
                y_s = y_s.argmax()

            cx_s, cy_s, cx_t, cinj_y_t, cacc_y_t = x_s, y_s, x_t, inj_y_t, acc_y_t
            if x_s.size(0) > x_t.size(0):
                cx_t, cinj_y_t, cacc_y_t = cycle(x_t), cycle(inj_y_t), cycle(acc_y_t)
            elif x_s.size(0) < x_t.size(0):
                cx_s, cy_s = cycle(x_s), cycle(y_s)

            _, _, feat_s, _, _ = model(x_s, task=task)
            _, _, feat_t, _, _ = model(x_t, task=task)
            distmat = torch.cdist(feat_s, feat_t)

            injection_pairs, accumulator_pairs = [], []
            for idx, (xs, xt, ys, iyt, ayt) in enumerate(zip(cx_s, cx_t, cy_s, cinj_y_t, cacc_y_t)):
                if ys == acc_y_t[distmat[idx % distmat.size(0)].argmin()]:
                    accumulator_pairs.append((xs, x_t[distmat[idx % distmat.size(0)].argmin()], ys))
                if y_s[distmat.t()[idx % distmat.t().size(0)].argmin()] == ayt:
                    accumulator_pairs.append((x_s[distmat.t()[idx % distmat.t().size(0)].argmin()], xt, y_s[distmat.t()[idx % distmat.t().size(0)].argmin()]))

                if ys == inj_y_t[distmat[idx % distmat.size(0)].argmin()]:
                    injection_pairs.append((xs, x_t[distmat[idx % distmat.size(0)].argmin()], ys))
                if y_s[distmat.t()[idx % distmat.t().size(0)].argmin()] == iyt:
                    injection_pairs.append((x_s[distmat.t()[idx % distmat.t().size(0)].argmin()], xt, y_s[distmat.t()[idx % distmat.t().size(0)].argmin()]))

            acc_s, acc_t, acc_y = [None] * len(accumulator_pairs), [None] * len(accumulator_pairs), [None] * len(accumulator_pairs)
            for i, (xs, xt, ys) in enumerate(accumulator_pairs):
                acc_s[i], acc_t[i], acc_y[i]  = xs, xt, ys

            inj_s, inj_t, inj_y = [None] * len(injection_pairs), [None] * len(injection_pairs), [None] * len(injection_pairs)
            for i, (xs, xt, ys) in enumerate(injection_pairs):
                inj_s[i], inj_t[i], inj_y[i]  = xs, xt, ys

            if len(acc_s) == 0:
                acc_s, acc_t, acc_y = None, None, None
            else:
                acc_s, acc_t, acc_y = torch.cat(acc_s).view(-1, C, H, W), torch.cat(acc_t).view(-1, C, H, W), torch.stack(acc_y)

            if len(inj_s) == 0:
                inj_s, inj_t, inj_y = None, None, None
            else:
                inj_s, inj_t, inj_y = torch.cat(inj_s).view(-1, C, H, W), torch.cat(inj_t).view(-1, C, H, W), torch.stack(inj_y)

            return (inj_s, inj_t, inj_y), (acc_s, acc_t, acc_y)

    def reorder_datasets2(self, model, x_s, y_s, x_t, task=0):
        model.eval()
        with torch.no_grad():
            _, C, H, W = x_s.size()

            _, _, feat_s, _, _ = model(x_s, task=task)
            _, _, feat_t, _, _ = model(x_t, task=task)

            sim_mat = torch.matmul(feat_s, feat_t.T)
            _, knn_idx = torch.max(sim_mat, 1)
            _, target_knn_idx = torch.max(sim_mat, 0)
            del sim_mat
            iy_t, ay_t = self(model, x_t, task=task)

            injection_pairs, accumulator_pairs = [], []
            for idx, (xt, yt) in enumerate(zip(x_t, ay_t)):
                cur_idx = target_knn_idx[idx]
                if cur_idx < 0: continue
                xs = x_s[cur_idx]
                ys = y_s[cur_idx]
                if ys == yt:
                    accumulator_pairs.append((xs, xt, ys))

            for idx, (xs, ys) in enumerate(zip(x_s, y_s)):
                cur_idx = knn_idx[idx]
                if cur_idx < 0: continue
                xt = x_t[cur_idx]
                yt = ay_t[cur_idx]
                if ys == yt:
                    accumulator_pairs.append((xs, xt, ys))

            for idx, (xt, yt) in enumerate(zip(x_t, iy_t)):
                cur_idx = target_knn_idx[idx]
                if cur_idx < 0: continue
                xs = x_s[cur_idx]
                ys = y_s[cur_idx]
                if ys - (self.args.num_classes // self.args.tasks * task) == yt:
                    injection_pairs.append((xs, xt, ys))

            for idx, (xs, ys) in enumerate(zip(x_s, y_s)):
                cur_idx = knn_idx[idx]
                if cur_idx < 0: continue
                xt = x_t[cur_idx]
                yt = iy_t[cur_idx]
                if ys - (self.args.num_classes // self.args.tasks * task) == yt:
                    injection_pairs.append((xs, xt, ys))

            acc_s, acc_t, acc_y = [None] * len(accumulator_pairs), [None] * len(accumulator_pairs), [None] * len(accumulator_pairs)
            for i, (xs, xt, ys) in enumerate(accumulator_pairs):
                acc_s[i], acc_t[i], acc_y[i]  = xs, xt, ys

            inj_s, inj_t, inj_y = [None] * len(injection_pairs), [None] * len(injection_pairs), [None] * len(injection_pairs)
            for i, (xs, xt, ys) in enumerate(injection_pairs):
                inj_s[i], inj_t[i], inj_y[i]  = xs, xt, ys

            if len(acc_s) == 0:
                acc_s, acc_t, acc_y = None, None, None
            else:
                acc_s, acc_t, acc_y = torch.cat(acc_s).view(-1, C, H, W), torch.cat(acc_t).view(-1, C, H, W), torch.stack(acc_y)

            if len(inj_s) == 0:
                inj_s, inj_t, inj_y = None, None, None
            else:
                inj_s, inj_t, inj_y = torch.cat(inj_s).view(-1, C, H, W), torch.cat(inj_t).view(-1, C, H, W), torch.stack(inj_y)

            return (inj_s, inj_t, inj_y), (acc_s, acc_t, acc_y)