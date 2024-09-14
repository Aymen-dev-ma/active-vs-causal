# plotting.py

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torch.nn.functional import sigmoid
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns

def generate_traversals(model, s_dim, s_sample, S_real, filenames=[], naive=False, colour=False):
    elements = 10

    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(s_dim, 3, width_ratios=[5,1,1])
    arg_max_hist_value = torch.zeros(s_dim)
    start_val = torch.zeros(s_dim)
    end_val = torch.zeros(s_dim)
    for s_indx in range(s_dim):
        plt.subplot(gs[s_indx*3+1])
        hh = plt.hist(s_sample[:,s_indx].cpu().numpy())

        if naive:
            arg_max_hist_value[s_indx] = 0.0
            start_val[s_indx] = -3.0
            end_val[s_indx] = 3.0
        else:
            index_of_highest = torch.argmax(torch.tensor(hh[0]))
            arg_max_hist_value[s_indx] = (hh[1][index_of_highest]+hh[1][index_of_highest+1])/2.0
            start_val[s_indx] = (hh[1][0]+hh[1][1])/2.0
            end_val[s_indx] = (hh[1][-2]+hh[1][-1])/2.0

    # Optional custom start and end values
    # start_val = torch.tensor([...])
    # arg_max_hist_value = torch.tensor([...])
    # end_val = torch.tensor([...])

    if len(S_real) > 0:
        correlations = torch.zeros((s_dim, S_real.shape[1]))
        correlations_cat = torch.zeros((s_dim, S_real.shape[1]))
        correlations_p = torch.zeros((s_dim, S_real.shape[1]))
        labels = ['shape', 'scale', 'orientation', 'posX', 'posY', 'reward']
        for real_s_indx in range(S_real.shape[1]):
            for s_indx in range(s_dim):
                corr, p_value = spearmanr(s_sample[:,s_indx].cpu().numpy(), S_real[:,real_s_indx].cpu().numpy())
                correlations[s_indx,real_s_indx] = abs(corr)
                correlations_p[s_indx,real_s_indx] = p_value
                correlations_cat[s_indx,real_s_indx] = mutual_info_regression(s_sample[:,s_indx].cpu().numpy().reshape(-1,1), S_real[:,real_s_indx].cpu().numpy())

        for s_indx in range(s_dim):
            plt.subplot(gs[s_indx*3+2])
            sns.lineplot(data=correlations[s_indx].cpu().numpy())
            if torch.max(correlations[s_indx]) < 0.5:
                plt.ylim(0.0,0.5)
            sns.lineplot(data=correlations_cat[s_indx].cpu().numpy())
            plt.ylabel('Correlation')
            plt.xticks(range(len(labels)), labels, rotation='vertical')

    for s_indx in range(s_dim):
        plt.subplot(gs[s_indx*3])
        plt.ylabel(r'$s_{'+str(s_indx)+'}$')
        s = torch.zeros((elements,s_dim))
        for x in range(elements):
            for y in range(s_dim):
                s[x,y] = arg_max_hist_value[y]

        for x,s_x in enumerate(torch.linspace(start_val[s_indx],end_val[s_indx],elements)):
            s[x,s_indx] = s_x
        with torch.no_grad():
            new_img = model.decode(s)
        if colour:
            plt.imshow(torch.hstack(new_img).cpu(), vmin=0, vmax=1)
        else:
            plt.imshow(torch.hstack(new_img).cpu(), cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(f"{start_val[s_indx]:.4f} <-- {arg_max_hist_value[s_indx]:.4f} --> {end_val[s_indx]:.4f}")

    fig.tight_layout()
    for filename in filenames:
        plt.savefig(filename)
    plt.close()

def reconstructions_plot(o0, o1, po1, filename, colour=False):
    if colour:
        o0 = o0[:7,:,:]
        o1 = o1[:7,:,:]
        po1 = po1[:7,:,:]
    else:
        o0 = o0[:7,:,:,0]
        o1 = o1[:7,:,:,0]
        po1 = po1[:7,:,:,0]
    fig = plt.figure(figsize=(10,5))
    plt.subplot(3,1,1)
    if colour: plt.imshow(np.hstack(o0), vmin=0, vmax=1)
    else: plt.imshow(np.hstack(o0), cmap='gray', vmin=0, vmax=1)
    plt.ylabel('o0')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,2)
    if colour: plt.imshow(np.hstack(o1))
    else: plt.imshow(np.hstack(o1), cmap='gray', vmin=0, vmax=1)
    plt.ylabel('o1')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,3)
    if colour: plt.imshow(np.hstack(po1), vmin=0, vmax=1)
    else: plt.imshow(np.hstack(po1), cmap='gray', vmin=0, vmax=1)
    plt.ylabel('o1 reconstr')
    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
