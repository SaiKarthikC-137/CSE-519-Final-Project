import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
import math
from collections import defaultdict
from typing import List

def superadam_helper(p_list: List[Tensor], g_list: List[Tensor], ea_list: List[Tensor],
                     easq_list: List[Tensor], gn_sums: List[Tensor], steps: List[int],
                     glob_H: bool, coord_glob_H: bool, tau_flag: bool, beta: float,
                     k: float, c: float, m: float, gamma: float, eps: float):
    for i, p in enumerate(p_list):
        g = g_list[i]
        sq_avg = easq_list[i]
        sq_avg.mul_(beta).addcmul_(g, g, value=1 - beta)

    if coord_glob_H:
        stacked_vals = [x.sqrt().view(-1) for x in easq_list]
        combined_vals = torch.cat(stacked_vals)
        norm_val = torch.norm(combined_vals)

    if glob_H:
        all_p = [xx.view(-1) for xx in p_list]
        big_cat = torch.cat(all_p)
        big_norm = torch.norm(big_cat)

    for i, p in enumerate(p_list):
        g = g_list[i]
        ea = ea_list[i]
        sq_avg = easq_list[i]
        curr_step = steps[i]

        if not tau_flag:
            ea.add_(g)
        else:
            tmp_eta = k / (m + curr_step)**0.5
            a_val = c * tmp_eta
            ea.mul_(1 - a_val).add_(g, alpha=a_val)

        bias_corr = 1 - (beta ** curr_step)
        if glob_H:
            gsum = gn_sums[i]
            gsum.mul_(beta).add_(big_norm**2, value=(1 - beta))
            denom = gsum.sqrt().add_(eps)
        elif coord_glob_H:
            denom = norm_val.add_(eps)
        else:
            denom = (sq_avg.sqrt() / math.sqrt(bias_corr)).add_(eps)

        if not tau_flag:
            chosen_eta = gamma * k / (m + curr_step)**(1/3)
        else:
            chosen_eta = gamma * k / (m + curr_step)**0.5

        p.addcdiv_(ea, denom, value=-chosen_eta)


class SuperAdam(Optimizer):
    def __init__(self, params, tau=False, gamma=0.01, k=1e-3, m=100, c=5, beta=0.999,
                 eps=1e-3, glob_H=False, coord_glob_H=False):
        d = dict(tau=tau, k=k, beta=beta, eps=eps, c=c, m=m, gamma=gamma,
                 glob_H=glob_H, coord_glob_H=coord_glob_H)
        super(SuperAdam, self).__init__(params, d)

    def __setstate__(self, state):
        super(SuperAdam, self).__setstate__(state)
        for g in self.param_groups:
            g.setdefault('amsgrad', False)

    def step(self):
        for g in self.param_groups:
            p_with_g = []
            grad_list = []
            ea_list = []
            easq_list = []
            gn_sum_list = []
            step_list = []

            for p in g['params']:
                if p.grad is not None:
                    p_with_g.append(p)
                    grad_list.append(p.grad)
                    st = self.state[p]
                    # Instead of being cautious, just set stuff if missing
                    if 'step' not in st:
                        st['step'] = 0
                        st['exp_avg'] = torch.zeros_like(p)
                        st['exp_avg_sq'] = torch.zeros_like(p)
                        if g['glob_H']:
                            st['grad_norm_sum'] = torch.zeros(1)

                    ea_list.append(st['exp_avg'])
                    easq_list.append(st['exp_avg_sq'])
                    if g['glob_H']:
                        gn_sum_list.append(st['grad_norm_sum'])

                    st['step'] += 1
                    step_list.append(st['step'])

            superadam_helper(p_with_g, grad_list, ea_list, easq_list, gn_sum_list,
                             step_list, g['glob_H'], g['coord_glob_H'], g['tau'],
                             g['beta'], g['k'], g['c'], g['m'], g['gamma'], g['eps'])
        return

    def update_momentum(self, closure=None):
        for g in self.param_groups:
            p_new = []
            g_new = []
            ea_new = []

            for p in g['params']:
                if p.grad is not None:
                    p_new.append(p)
                    g_new.append(p.grad)
                    stt = self.state[p]
                    ea_new.append(stt['exp_avg'])

            eta_val = g['k'] / (g['m'] + stt['step']) ** (1/3)
            a_val = min(g['c'] * (eta_val ** 2), 0.99)
            for i, _ in enumerate(p_new):
                ea_new[i].add_(-g_new[i]).mul_(1 - a_val)
        return
