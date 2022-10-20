from itertools import cycle
import torch
import numpy as np
import tqdm

class CenterAwarePseudoModule(torch.nn.Module):

    def __init__(self, model, loader_target, loader=None, distance='cosine', threshold=0.):
        super(CenterAwarePseudoModule, self).__init__()

        model.eval()
        
        counter = 0
        with torch.no_grad():
            if loader is None:
                for n_iter, (input, _) in enumerate(tqdm.tqdm(loader_target)):
                    input = input.cuda()
                    outputs, feas = model(input, return_features=True)

                    if n_iter == 0:
                        all_fea = torch.zeros(len(loader_target.dataset), feas.flatten(1).size(1))
                        all_output = torch.zeros(len(loader_target.dataset), outputs.size(1))
                    
                    for _, (output, fea) in enumerate(zip(outputs, feas)):
                        all_fea[counter] = fea.flatten().clone().float().cpu()
                        all_output[counter] = output.clone().float().cpu()
                        counter += 1
            else:
                for n_iter, (input, _) in enumerate(tqdm.tqdm(loader)):
                    input = input.cuda()
                    outputs, feas = model(input, return_features=True)

                    if n_iter == 0:
                        all_fea = torch.zeros(len(loader_target.dataset) + len(loader.dataset), feas.flatten(1).size(1))
                        all_output = torch.zeros(len(loader_target.dataset) + len(loader.dataset), outputs.size(1))

                    for _, (output, fea) in enumerate(zip(outputs, feas)):
                        all_fea[counter] = fea.flatten().clone().float().cpu()
                        all_output[counter] = output.clone().float().cpu()
                        counter += 1

                for _, (input, _) in enumerate(tqdm.tqdm(loader_target)):
                    input = input.cuda()
                    outputs, feas = model(input, return_features=True)

                    for _, (output, fea) in enumerate(zip(outputs, feas)):
                        all_fea[counter] = fea.flatten().clone().float().cpu()
                        all_output[counter] = output.clone().float().cpu()
                        counter += 1

            all_output = torch.nn.Softmax(dim=1)(all_output)
            _, predict = all_output.max(1)

            if distance == 'cosine':
                all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
                all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

            all_fea = all_fea.float().cpu()
            K = all_output.size(1)
            aff = all_output.float().cpu()

            initc = aff.t().matmul(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            cls_count = torch.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count > threshold)
            labelset = labelset[0]
            labelset = torch.LongTensor(labelset)

            # dd = torch.cdist(all_fea, initc[labelset])
            # pred_label = dd.argmin(1)
            # pred_label = labelset[pred_label]

            # for _ in range(1):
            #     aff = torch.eye(K)[pred_label]
            #     initc = aff.t().matmul(all_fea)
            #     initc = initc / (1e-8 + aff.sum(0)[:, None])
            #     dd = torch.cdist(all_fea, initc[labelset])
            #     pred_label = dd.argmin(axis=1)
            #     pred_label = labelset[pred_label]

            self.labelset = labelset
            self.initc = initc.cuda()

    def forward(self, model, x, distance='cosine'):
        model.eval()
        with torch.no_grad():
            all_fea = model.tokenizer(x)
            all_fea = all_fea.flatten(1)
            if distance == 'cosine':
                all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1).cuda()), 1)
                all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

            dd = torch.cdist(all_fea, self.initc[self.labelset])
            pred_label = dd.argmin(axis=1)
            pred_label = self.labelset[pred_label]

            return pred_label

    def reorder_datasets(self, model, x_s, y_s, x_t):
        model.eval()
        with torch.no_grad():
            _, C, H, W = x_s.size()

            y_t = self(model, x_t)
            if len(y_s.size()) > 1:
                y_s = y_s.argmax()

            cx_s, cy_s, cx_t, cy_t = x_s, y_s, x_t, y_t
            if x_s.size(0) > x_t.size(0):
                cx_s, cy_s = cycle(x_s), cycle(y_s)
            elif x_s.size(0) < x_t.size(0):
                cx_t, cy_t = cycle(x_t), cycle(y_t)

            distmat = torch.cdist(model.tokenizer(x_s).flatten(1), model.tokenizer(x_t).flatten(1))

            pairs = []
            for idx, (xs, xt, ys, yt) in enumerate(zip(cx_s, cx_t, cy_s, cy_t)):
                if ys == y_t[distmat[idx % distmat.size(0)].argmin()]:
                    pairs.append((xs, x_t[distmat[idx % distmat.size(0)].argmin()], ys))
                if y_s[distmat.t()[idx % distmat.t().size(0)].argmin()] == yt:
                    pairs.append((x_s[distmat.t()[idx % distmat.t().size(0)].argmin()], xt, y_s[distmat.t()[idx % distmat.t().size(0)].argmin()]))

            s, t, y = [None] * len(pairs), [None] * len(pairs), [None] * len(pairs)
            for i, (xs, xt, ys) in enumerate(pairs):
                s[i] = xs
                t[i] = xt
                y[i] = ys

            if len(s) == 0:
                return None, None, None

            s, t, y = torch.cat(s).view(-1, C, H, W), torch.cat(t).view(-1, C, H, W), torch.stack(y)

            return s, t, y

    def reorder_datasets2(self, model, x_s, y_s, x_t):
        model.eval()
        with torch.no_grad():
            _, C, H, W = x_s.size()

            sim_mat = torch.matmul(x_s, x_t.T)
            _, knn_idx = torch.max(sim_mat, 1)
            target_knn_sim, target_knn_idx = torch.max(sim_mat, 0)
            _, knn_idx_topk = torch.topk(sim_mat, k=2, dim=1)
            del sim_mat
            y_t = self.ob

            y_t = model(x_t).argmin(1)
            if len(y_s.size()) > 1:
                y_s = y_s.argmax()

            cx_s, cy_s, cx_t, cy_t = x_s, y_s, x_t, y_t
            if x_s.size(0) > x_t.size(0):
                cx_s, cy_s = cycle(x_s), cycle(y_s)
            elif x_s.size(0) < x_t.size(0):
                cx_t, cy_t = cycle(x_t), cycle(y_t)

            distmat = torch.cdist(model.tokenizer(x_s).flatten(1), model.tokenizer(x_t).flatten(1))

            pairs = []
            for idx, (xs, xt, ys, yt) in enumerate(zip(cx_s, cx_t, cy_s, cy_t)):
                if ys == y_t[distmat[idx % distmat.size(0)].argmin()]:
                    pairs.append((xs, x_t[distmat[idx % distmat.size(0)].argmin()], ys))
                if y_s[distmat.t()[idx % distmat.t().size(0)].argmin()] == yt:
                    pairs.append((x_s[distmat.t()[idx % distmat.t().size(0)].argmin()], xt, y_s[distmat.t()[idx % distmat.t().size(0)].argmin()]))

            s, t, y = [None] * len(pairs), [None] * len(pairs), [None] * len(pairs)
            for i, (xs, xt, ys) in enumerate(pairs):
                s[i] = xs
                t[i] = xt
                y[i] = ys

            if len(s) == 0:
                return None, None, None

            s, t, y = torch.cat(s).view(-1, C, H, W), torch.cat(t).view(-1, C, H, W), torch.stack(y)

            return s, t, y

    def reorder_loaders(self, model, loader, loader_target):
        model.eval()

        with torch.no_grad():
            counter = 0
            for n_iter, (inputs, labels) in enumerate(tqdm.tqdm(loader)):
                inputs = model.tokenizer(inputs).flatten(1)
                if n_iter == 0:
                    _, C, H, W = inputs.size()
                    x_s = torch.zeros(tuple([len(loader.dataset)] + list(inputs[0].size())))
                    y_s = torch.zeros(tuple([len(loader.dataset)] + list(labels[0].size())))

                for _, (input, label) in enumerate(zip(inputs, labels)):
                    x_s[counter] = input.clone().float().cpu()
                    y_s[counter] = label.clone().float().cpu()
                    counter += 0
            
            counter = 0
            for n_iter, (inputs, _) in enumerate(tqdm.tqdm(loader_target)):
                labels, inputs = model(inputs, return_features=True)
                inputs = inputs.flatten(1)

                if n_iter == 0:
                    x_t = torch.zeros(tuple([len(loader_target.dataset)] + list(inputs[0].size())))
                    y_t = torch.zeros(tuple([len(loader_target.dataset)] + list(labels[0].size())))

                for _, (input, label) in enumerate(zip(inputs, labels)):
                    x_t[counter] = input.clone().float().cpu()
                    y_t[counter] = label.clone().float().cpu()
                    counter += 0

        cx_s, cy_s, cx_t, cy_t = x_s, y_s, x_t, y_t
        if x_s.size(0) > x_t.size(0):
            cx_s, cy_s = cycle(x_s), cycle(y_s)
        elif x_s.size(0) < x_t.size(0):
            cx_t, cy_t = cycle(x_t), cycle(y_t)

        distmat = torch.cdist(model.tokenizer(x_s), model.tokenizer(x_t))

        pairs = []
        for idx, (xs, xt, ys, yt) in enumerate(zip(cx_s, cx_t, cy_s, cy_t)):
            if ys == y_t[distmat[idx % distmat.size(0)].argmin()]:
                pairs.append((xs, x_t[distmat[idx % distmat.size(0)].argmin()], ys))
            if y_s[distmat.t()[idx % distmat.t().size(0)].argmin()] == yt:
                pairs.append((x_s[distmat.t()[idx % distmat.t().size(0)].argmin()], xt, y_s[distmat.t()[idx % distmat.t().size(0)].argmin()]))

        s, t, y = [None] * len(pairs), [None] * len(pairs), [None] * len(pairs)
        for i, (xs, xt, ys) in enumerate(pairs):
            s[i] = xs
            t[i] = xt
            y[i] = ys

        if len(s) == 0:
            return None, None, None

        s, t, y = torch.cat(s).view(-1, C, H, W), torch.cat(t).view(-1, C, H, W), torch.stack(y)

        return s, t, y
