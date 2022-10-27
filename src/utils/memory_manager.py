import torch

class RehearsalMemoryManager(torch.nn.Module):
    def __init__(self, args):
        super(RehearsalMemoryManager, self).__init__()

        self.source_memory = torch.zeros(args.memory_size, args.chans, args.img_size, args.img_size, requires_grad=False)
        self.target_memory = torch.zeros(args.memory_size, args.chans, args.img_size, args.img_size, requires_grad=False)
        self.label_memory = torch.zeros(args.memory_size, dtype=torch.int64, requires_grad=False)
        self.injection_source_logits = torch.zeros(args.memory_size, args.num_classes // args.tasks, requires_grad=False)
        self.injection_target_logits = torch.zeros(args.memory_size, args.num_classes // args.tasks, requires_grad=False)
        self.accumulator_source_logits = torch.zeros(args.memory_size, args.num_classes, requires_grad=False)
        self.accumulator_target_logits = torch.zeros(args.memory_size, args.num_classes, requires_grad=False)
        self.memory_size = args.memory_size
        self.n_tasks = 1
        self.memory_size_per_task = self.memory_size // self.n_tasks
        self.counter = [0]
        self.args = args
        

    def increment_task(self):
        s = torch.zeros(self.args.memory_size, self.args.chans, self.args.img_size, self.args.img_size, requires_grad=False)
        t = torch.zeros(self.args.memory_size, self.args.chans, self.args.img_size, self.args.img_size, requires_grad=False)
        y = torch.zeros(self.args.memory_size, requires_grad=False)
        ils = torch.zeros(self.args.memory_size, self.args.num_classes // self.args.tasks, requires_grad=False)
        ilt = torch.zeros(self.args.memory_size, self.args.num_classes // self.args.tasks, requires_grad=False)
        als = torch.zeros(self.args.memory_size, self.args.num_classes, requires_grad=False)
        alt = torch.zeros(self.args.memory_size, self.args.num_classes, requires_grad=False)

        for task in range(self.n_tasks):
            s[0:self.memory_size // (self.n_tasks + 1)]  = self.source_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            t[0:self.memory_size // (self.n_tasks + 1)]  = self.target_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            y[0:self.memory_size // (self.n_tasks + 1)]  = self.label_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            ils[0:self.memory_size // (self.n_tasks + 1)] = self.injection_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            ilt[0:self.memory_size // (self.n_tasks + 1)] = self.injection_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            als[0:self.memory_size // (self.n_tasks + 1)] = self.accumulator_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            alt[0:self.memory_size // (self.n_tasks + 1)] = self.accumulator_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][0:self.memory_size // (self.n_tasks + 1)]
            self.counter[task] = self.memory_size // (self.n_tasks + 1)

        self.n_tasks = self.n_tasks + 1
        self.memory_size_per_task = self.memory_size // self.n_tasks
        self.counter = self.counter + [0]

    def add_sample(self, source, target, label, injection_source_logit, injection_target_logit, accumulator_source_logit, accumulator_target_logit, task=0):
        if self.counter[task] >= self.memory_size_per_task:
            sort_indeces = self.injection_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task].max(1)[0].sort()[1]

            s  = self.source_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]
            t  = self.target_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]
            y  = self.label_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]
            ils = self.injection_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]
            ilt = self.injection_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]
            als = self.accumulator_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]
            alt = self.accumulator_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task][sort_indeces]

            if injection_source_logit.max(0)[0] >= ils[sort_indeces][-1].max(0)[0] and (injection_source_logit.max(0)[0] > ils[sort_indeces][-1].max(0)[0] or injection_target_logit.max(0)[0] > ilt[sort_indeces][-1].max(0)[0]):
                s[-1]  = source.clone().detach().cpu()
                t[-1]  = target.clone().detach().cpu()
                y[-1]  = label.clone().detach().cpu()
                ils[-1] = injection_source_logit.clone().detach().cpu()
                ilt[-1] = injection_target_logit.clone().detach().cpu()
                als[-1] = accumulator_source_logit.clone().detach().cpu()
                alt[-1] = accumulator_target_logit.clone().detach().cpu()

            self.source_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task] = s
            self.target_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task] = t
            self.label_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task]  = y
            self.injection_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task] = ils
            self.injection_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task] = ilt
            self.accumulator_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task] = als
            self.accumulator_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task] = alt
        else:
            self.source_memory[self.memory_size_per_task * task + self.counter[task]] = source.clone().detach().cpu()
            self.target_memory[self.memory_size_per_task * task + self.counter[task]] = target.clone().detach().cpu()
            self.label_memory[self.memory_size_per_task * task + self.counter[task]]  = label.clone().detach().cpu()
            self.injection_source_logits[self.memory_size_per_task * task + self.counter[task]] = injection_source_logit.clone().detach().cpu()
            self.injection_target_logits[self.memory_size_per_task * task + self.counter[task]] = injection_target_logit.clone().detach().cpu()
            self.accumulator_source_logits[self.memory_size_per_task * task + self.counter[task]] = accumulator_source_logit.clone().detach().cpu()
            self.accumulator_target_logits[self.memory_size_per_task * task + self.counter[task]] = accumulator_target_logit.clone().detach().cpu()
            self.counter[task] += 1

    def dataset_loader(self, task=0):
        loader = torch.utils.data.DataLoader(CustomRehearsalDataset(self.source_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task],
                                                                    self.target_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task],
                                                                    self.label_memory[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task],
                                                                    self.accumulator_source_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task],
                                                                    self.accumulator_target_logits[task*self.memory_size_per_task:(task+1)*self.memory_size_per_task]),
                                             batch_size=self.args.source_batch_size,
                                             shuffle=True)

        return loader
            
class CustomRehearsalDataset(torch.utils.data.Dataset):
    def __init__(self, source, target, label, source_logit, target_logit):
        super(CustomRehearsalDataset, self).__init__()
        self.source = source
        self.target = target
        self.label = label
        self.source_logit = source_logit
        self.target_logit = target_logit

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        return (self.source[idx],
                self.target[idx],
                self.label[idx],
                self.source_logit[idx],
                self.target_logit[idx])