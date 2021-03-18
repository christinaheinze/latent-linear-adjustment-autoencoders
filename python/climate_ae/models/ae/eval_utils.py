import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_r2(target, pred):
    ''' computes R2 given target time series and predicted time series
         in other words, computes R2 at the grid point level'''
    residual_ss = np.sum((target-pred)**2)
    colmeans_target = np.mean(target, axis=0)
    total_ss = np.sum((target-colmeans_target)**2)
    r2 = 1-residual_ss/total_ss  
    return r2


def get_r2_maps(x, xhat, xhatexp):
    map_xxhat = compute_r2_map(x, xhat)
    map_xxhatexp = compute_r2_map(x, xhatexp)
    return map_xxhat, map_xxhatexp


def get_mse_maps(x, xhat, xhatexp):
    map_xxhat = compute_mse_map(x, xhat)
    map_xxhatexp = compute_mse_map(x, xhatexp)
    return map_xxhat, map_xxhatexp


def plot_mse_map(x, xhat, xhatexp, out_dir, split, pdf=True):
    map_xxhat, map_xxhatexp = get_mse_maps(x, xhat, xhatexp)
    _plot_mse_map(map_xxhat, map_xxhatexp, out_dir, split, pdf)
    _plot_mse_hist(map_xxhat, map_xxhatexp, out_dir, split)
    return map_xxhat, map_xxhatexp


def plot_r2_map(x, xhat, xhatexp, out_dir, split, pdf=True):
    map_xxhat, map_xxhatexp = get_r2_maps(x, xhat, xhatexp)    
    _plot_r2_map(map_xxhat, map_xxhatexp, out_dir, split, pdf)
    _plot_r2_hist(map_xxhat, map_xxhatexp, out_dir, split)
    return map_xxhat, map_xxhatexp


def _plot_r2_map(map_xxhat, map_xxhatexp, out_dir, split, pdf):    
    # plot
    plot_format = 'pdf' if pdf else 'png'
    fig, axes = plt.subplots(1, 2, figsize=(2*5,4))
    min_val = 0 
    max_val = 1
    current_cmap = plt.get_cmap('RdBu_r')
    current_cmap.set_bad(color='yellow')
    axes[0].imshow(np.squeeze(map_xxhat), cmap=current_cmap, vmin = min_val, 
        vmax = max_val)
    axes[0].axis('off')
    axes[0].set_title('R^2(x, xhat)')
    pcm = axes[1].imshow(np.squeeze(map_xxhatexp), cmap=current_cmap, 
        vmin = min_val, vmax = max_val)
    axes[1].axis('off')
    axes[1].set_title('R^2(x, xhatexp)')
    fig.colorbar(pcm, ax=axes, extend='both', shrink=.7)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(out_dir,'{}_R2_map.{}'.format(split, plot_format)), 
        bbox_inches='tight')


def _plot_r2_hist(map_xxhat, map_xxhatexp, out_dir, split):    
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(2*5,4))
    axes[0].hist(map_xxhat.flatten(), bins='auto')
    mean_xxhat, med_xxhat = np.mean(map_xxhat), np.median(map_xxhat)
    std_xxhat = np.std(map_xxhat)
    axes[0].set_title('R^2(x, xhat). Mean: {:.2f}, Std: {:.2f}, Med: {:.2f}'.format(
        mean_xxhat, std_xxhat, med_xxhat))

    axes[1].hist(map_xxhatexp.flatten(), bins='auto')
    mean_xxhatexp, med_xxhatexp = np.mean(map_xxhatexp), np.median(map_xxhatexp)
    std_xxhatexp = np.std(map_xxhatexp)
    axes[1].set_title('R^2(x, xhatexp). Mean: {:.2f}, Std: {:.2f}, Med: {:.2f}'.format(
        mean_xxhatexp, std_xxhatexp, med_xxhatexp))

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_R2_hist.pdf'.format(split)), 
        bbox_inches='tight')


def _plot_mse_hist(map_xxhat, map_xxhatexp, out_dir, split):    
    # plot
    fig, axes = plt.subplots(1, 2, figsize=(2*5,4))
    axes[0].hist(map_xxhat.flatten(), bins='auto')
    mean_xxhat, med_xxhat = np.mean(map_xxhat), np.median(map_xxhat)
    std_xxhat = np.std(map_xxhat)
    axes[0].set_title('MSE(x, xhat). Mean: {:.2f}, Std: {:.2f}, Med: {:.2f}'.format(
        mean_xxhat, std_xxhat, med_xxhat))

    axes[1].hist(map_xxhatexp.flatten(), bins='auto')
    mean_xxhatexp, med_xxhatexp = np.mean(map_xxhatexp), np.median(map_xxhatexp)
    std_xxhatexp = np.std(map_xxhatexp)
    axes[1].set_title('MSE(x, xhatexp). Mean: {:.2f}, Std: {:.2f}, Med: {:.2f}'.format(
        mean_xxhatexp, std_xxhatexp, med_xxhatexp))

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_mse_hist.pdf'.format(split)), 
        bbox_inches='tight')


def _plot_mse_map(map_xxhat, map_xxhatexp, out_dir, split, pdf):    
    # plot
    plot_format = 'pdf' if pdf else 'png'
    fig, axes = plt.subplots(1, 2, figsize=(2*5,4))
    concat = np.concatenate((map_xxhat, map_xxhatexp))
    min_val = np.min(concat)
    max_val = np.max(concat)
    current_cmap = plt.get_cmap('RdBu_r')
    current_cmap.set_bad(color='yellow')
    axes[0].imshow(np.squeeze(map_xxhat), cmap=current_cmap, vmin = min_val, 
        vmax = max_val)
    axes[0].axis('off')
    axes[0].set_title('MSE(x, xhat)')
    pcm = axes[1].imshow(np.squeeze(map_xxhatexp), cmap=current_cmap, 
        vmin = min_val, vmax = max_val)
    axes[1].axis('off')
    axes[1].set_title('MSE(x, xhatexp)')
    fig.colorbar(pcm, ax=axes, extend='both', shrink=.7)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(out_dir,'{}_mse_map.{}'.format(split, plot_format)), 
        bbox_inches='tight')


def compute_r2_map(x, xhat):
    '''computes the R2 map, i.e. computes R2 for each grid point 
       and returns the resulting map'''
    len_x = x.shape[1]
    len_y = x.shape[2]
    output_map = np.zeros([len_x, len_y])
    for i in range(len_x):
        for j in range(len_y):
            grid_point_x = x[:,i,j,:] 
            grid_point_xhat = xhat[:,i,j,:] 
            output_map[i,j] = compute_r2(grid_point_x, grid_point_xhat)
    return output_map


def compute_mse_map(x, xhat):
    len_x = x.shape[1]
    len_y = x.shape[2]
    output_map = np.zeros([len_x, len_y])
    for i in range(len_x):
        for j in range(len_y):
            grid_point_x = x[:,i,j,:] 
            grid_point_xhat = xhat[:,i,j,:] 
            output_map[i,j] = np.mean((grid_point_x-grid_point_xhat)**2)
    return output_map


def visualize(te_inputs, te_annos, model, reg, out_dir, split, 
    num_imgs = 15, seed=2, transform_back=False, pdf=True, random=True):

    plot_format = 'pdf' if pdf else 'png'

    # number of interventions first viz 
    num_int = 3
    # get a subsample to plot
    if random:
        np.random.seed(seed)
        indices = np.random.choice(np.arange(te_inputs.shape[0]), size=num_imgs)
    else:
        indices = range(num_imgs)

    te_input_sub = te_inputs[indices, ...]
    te_anno_sub = te_annos[indices, ...]

    # reconstructions
    model_output = model.autoencode(te_input_sub, training=False)
    te_reconstructions_sub = model_output["output"].numpy()
    te_fitted_sub = reg.predict(te_anno_sub)
    
    te_exp_reconstructions_sub = model.decode(te_fitted_sub, 
        training=False)["output"].numpy()

    if transform_back:
        te_input_sub = te_input_sub ** 2
        te_reconstructions_sub = te_reconstructions_sub ** 2
        te_exp_reconstructions_sub = te_exp_reconstructions_sub ** 2

    # plot input and reconstructions
    fig, axes = plt.subplots(num_imgs, 2, figsize=(2*2,2*num_imgs))
    concat = np.concatenate((te_input_sub, te_reconstructions_sub))
    min_val = np.min(concat)
    max_val = np.max(concat)
    for i in range(num_imgs):
        axes[i, 0].imshow(np.squeeze(te_input_sub[i, ...]), cmap='Blues', 
            vmin = min_val, vmax = max_val)
        axes[i, 0].axis('off')
        pcm = axes[i, 1].imshow(np.squeeze(te_reconstructions_sub[i, ...]), 
            cmap='Blues', vmin = min_val, vmax = max_val)
        axes[i, 1].axis('off')
        fig.colorbar(pcm, ax=axes[i, :], extend='both', shrink=.7)

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(out_dir,'{}_input_recon.{}'.format(split, plot_format)), 
        bbox_inches='tight')

    # plot interventions
    concat2 = np.concatenate((te_input_sub, te_reconstructions_sub, 
        te_exp_reconstructions_sub))
    min_val2 = np.min(concat2)
    max_val2 = np.max(concat2)
    fig2, axes2 = plt.subplots(num_imgs, num_int, figsize=(2.2*num_int,1.8*num_imgs))
    for i in range(num_imgs):
        # original
        axes2[i, 0].imshow(np.squeeze(te_input_sub[i, ...]), cmap='Blues', 
            vmin = min_val2, vmax = max_val2)
        axes2[i, 0].axis('off')
        if i == 0:
            axes2[i, 0].set_title('Original')
        # reconstruction
        axes2[i, 1].imshow(np.squeeze(te_reconstructions_sub[i, ...]), 
            cmap='Blues', vmin = min_val2, vmax = max_val2)
        axes2[i, 1].axis('off')
        if i == 0:
            axes2[i, 1].set_title('Reconstruction')
        # expected reconstruction
        pcm = axes2[i, 2].imshow(np.squeeze(te_exp_reconstructions_sub[i, ...]), 
            cmap='Blues', vmin = min_val2, vmax = max_val2)
        axes2[i, 2].axis('off')
        if i == 0:
            axes2[i, 2].set_title('Prediction')
        fig.colorbar(pcm, ax=axes2[i, :], extend='both', shrink=.7)

    fig2.savefig(
        os.path.join(out_dir,'{}_interventions.{}'.format(split, plot_format)), 
        bbox_inches='tight')
    
    return te_input_sub, te_reconstructions_sub, te_exp_reconstructions_sub


def plot_timeseries_boots(xhatexp, xhatexp_boots, out_dir, split, loc=None, 
    num_imgs=16, seed=1):
    plot_range_start = 0
    plot_range_stop = 500

    # get a subsample to plot
    np.random.seed(seed)
    if loc is None:
        indices_x = np.random.choice(xhatexp.shape[1], size=num_imgs, 
            replace=False)
        indices_y = np.random.choice(xhatexp.shape[2], size=num_imgs, 
            replace=False)
    else:
        indices_x = []
        indices_y = []
        r2 = []
        _, map_xxhatexp, loc_xxhat, loc_xxhatexp = loc
        for entry in sorted(loc_xxhatexp.keys()):
            indices_x.append(entry[0])
            indices_y.append(entry[1])
            r2.append(loc_xxhatexp[entry])
        for entry in sorted(loc_xxhat.keys()):
            indices_x.append(entry[0])
            indices_y.append(entry[1])
            r2.append(map_xxhatexp[entry])

    n = xhatexp.shape[0]
    xhatexp_sub = np.zeros([num_imgs, n])
    xhatexp_boots_sub = np.zeros([num_imgs, xhatexp_boots.shape[0]])

    for i, xy in enumerate(zip(indices_x, indices_y)):
        x, y = xy
        xhatexp_sub[i, :] = xhatexp[:, x, y, 0]
        xhatexp_boots_sub[i, :] = xhatexp_boots[:, x, y, 0]

    # plot input and reconstructions
    fig, axes = plt.subplots(num_imgs, 2, figsize=(20*2,2*num_imgs))
    fig.tight_layout()
    for i in range(num_imgs):
        axes[i, 0].plot(xhatexp_sub[i, plot_range_start:plot_range_stop], color='blue')
        axes[i, 1].plot(xhatexp_boots_sub[i, plot_range_start:plot_range_stop], color='blue')
        if i == 0:
            if loc is not None:
                axes[i, 0].set_title('Prediction. Position: ({}, {}), '
                    'R2: {:.2f}'.format(indices_x[i], indices_y[i], r2[i]))
                axes[i, 1].set_title('Bootstrapped prediction. '
                    'Position: ({}, {}) '.format(indices_x[i], indices_y[i]))
            else:
                axes[i, 0].set_title('Prediction')
                axes[i, 1].set_title('Bootstrapped prediction')
        else:
            if loc is not None:
                axes[i, 0].set_title('Position: ({}, {}), R2: {:.2f}'.format(
                    indices_x[i], indices_y[i], r2[i]))
                axes[i, 1].set_title('Position: ({}, {})'.format(
                    indices_x[i], indices_y[i]))
            else:
                axes[i, 0].set_title('Position: ({}, {})'.format(
                    indices_x[i], indices_y[i]))
                axes[i, 1].set_title('Position: ({}, {})'.format(
                    indices_x[i], indices_y[i]))

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_boots_ts.pdf'.format(split)), 
        bbox_inches='tight')


def plot_timeseries_boots_list(xhatexp, xhatexp_boots_list, out_dir, split, 
    loc=None, num_imgs=16, seed=1):
    plot_range_start = 0
    plot_range_stop = 500

    # get a subsample to plot
    np.random.seed(seed)
    if loc is None:
        indices_x = np.random.choice(xhatexp.shape[1], size=num_imgs, 
            replace=False)
        indices_y = np.random.choice(xhatexp.shape[2], size=num_imgs, 
            replace=False)
    else:
        indices_x = []
        indices_y = []
        r2 = []
        _, map_xxhatexp, loc_xxhat, loc_xxhatexp = loc
        for entry in sorted(loc_xxhatexp.keys()):
            indices_x.append(entry[0])
            indices_y.append(entry[1])
            r2.append(loc_xxhatexp[entry])
        for entry in sorted(loc_xxhat.keys()):
            indices_x.append(entry[0])
            indices_y.append(entry[1])
            r2.append(map_xxhatexp[entry])

    n = xhatexp.shape[0]
    xhatexp_sub = np.zeros([num_imgs, n])
    xhatexp_boots_sub_list = list()
    for i in range(len(xhatexp_boots_list)):
        xhatexp_boots_sub_list.append(
            np.zeros([num_imgs, xhatexp_boots_list[i].shape[0]]))

    for i, xy in enumerate(zip(indices_x, indices_y)):
        x, y = xy
        xhatexp_sub[i, :] = xhatexp[:, x, y, 0]
        for j in range(len(xhatexp_boots_sub_list)):
            xhatexp_boots_sub_list[j][i, :] = xhatexp_boots_list[j][:, x, y, 0]

    # plot input and reconstructions
    fig, axes = plt.subplots(num_imgs, 2, figsize=(20*2,2*num_imgs))
    fig.tight_layout()
    for i in range(num_imgs):
        axes[i, 0].plot(xhatexp_sub[i, plot_range_start:plot_range_stop], color='blue')
        for j in range(len(xhatexp_boots_sub_list)):
            axes[i, 1].plot(xhatexp_boots_sub_list[j][i, plot_range_start:plot_range_stop], color='blue')
        if i == 0:
            if loc is not None:
                axes[i, 0].set_title('Prediction. Position: ({}, {}), '
                    'R2: {:.2f}'.format(indices_x[i], indices_y[i], r2[i]))
                axes[i, 1].set_title('Bootstrapped prediction. '
                    'Position: ({}, {}) '.format(indices_x[i], indices_y[i]))
            else:
                axes[i, 0].set_title('Prediction')
                axes[i, 1].set_title('Bootstrapped prediction')
        else:
            if loc is not None:
                axes[i, 0].set_title('Position: ({}, {}), R2: {:.2f}'.format(
                    indices_x[i], indices_y[i], r2[i]))
                axes[i, 1].set_title('Position: ({}, {})'.format(
                    indices_x[i], indices_y[i]))
            else:
                axes[i, 0].set_title('Position: ({}, {})'.format(
                    indices_x[i], indices_y[i]))
                axes[i, 1].set_title('Position: ({}, {})'.format(
                    indices_x[i], indices_y[i]))

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_boots_list_ts.pdf'.format(split)), 
        bbox_inches='tight')


def visualize_boots(xhatexp, xhatexp_boots, out_dir, split, 
    num_imgs = 30, seed=1):
    
    # get a subsample to plot
    np.random.seed(seed)
    indices = range(num_imgs)
    xhatexp_sub = xhatexp[indices, ...]
    xhatexp_boots_sub = xhatexp_boots[indices, ...]

    # plot predictions and bootstrapped predictions
    fig, axes = plt.subplots(num_imgs, 2, figsize=(3*2,2*num_imgs))
    concat = np.concatenate((xhatexp_sub, xhatexp_boots_sub))
    min_val = np.min(concat)
    max_val = np.max(concat)
    for i in range(num_imgs):
        axes[i, 0].imshow(np.squeeze(xhatexp_sub[i, ...]), cmap='Blues', 
            vmin = min_val, vmax = max_val)
        axes[i, 0].axis('off')
        pcm = axes[i, 1].imshow(np.squeeze(xhatexp_boots_sub[i, ...]), 
            cmap='Blues', vmin = min_val, vmax = max_val)
        axes[i, 1].axis('off')
        fig.colorbar(pcm, ax=axes[i, :], extend='both', shrink=.7)
        if i == 0:
            axes[i, 0].set_title('Prediction')
            axes[i, 1].set_title('Bootstrapped prediction')

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_predictions_boots.pdf'.format(split)), 
        bbox_inches='tight')


def visualize_boots_list(xhatexp, xhatexp_boots_list, out_dir, split, 
    num_imgs = 30, seed=1):
    # number of bts samples
    n_bts_samples = len(xhatexp_boots_list)

    # get a subsample to plot
    np.random.seed(seed)
    # indices = range(num_imgs)
    choose_from = min(xhatexp.shape[0], xhatexp_boots_list[0].shape[0])
    indices = np.random.choice(choose_from, size=num_imgs, 
            replace=False)
    xhatexp_sub = xhatexp[indices, ...]
    xhatexp_boots_sub_list = list()
    for i in range(len(xhatexp_boots_list)):
        xhatexp_boots_sub_list.append(xhatexp_boots_list[i][indices, ...])

    # plot predictions and bootstrapped predictions
    fig, axes = plt.subplots(num_imgs, 1+n_bts_samples, 
        figsize=(3*(1+n_bts_samples), 2*num_imgs))
    concat = np.concatenate((xhatexp_sub, np.concatenate(xhatexp_boots_sub_list, axis = 0)))
    min_val = np.min(concat)
    max_val = np.max(concat)
    for i in range(num_imgs):
        axes[i, 0].imshow(np.squeeze(xhatexp_sub[i, ...]), cmap='Blues', 
            vmin = min_val, vmax = max_val)
        axes[i, 0].axis('off')
        for j in range(n_bts_samples):
            pcm = axes[i, j+1].imshow(np.squeeze(xhatexp_boots_sub_list[j][i, ...]), 
                cmap='Blues', vmin = min_val, vmax = max_val)
            axes[i, j+1].axis('off')
        fig.colorbar(pcm, ax=axes[i, :], extend='both', shrink=.7)
        if i == 0:
            axes[i, 0].set_title('Prediction')
            for j in range(1, n_bts_samples+1):
                axes[i, j].set_title('Emulated')

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_predictions_boots_list.pdf'.format(split)), 
        bbox_inches='tight')
    return xhatexp_sub, xhatexp_boots_sub_list


def visualize_quiz(xhatexp, xhatexp_boots, out_dir, split, 
    seed=1):
    num_imgs = 5
    # get a subsample to plot
    np.random.seed(seed)
    # indices = range(num_imgs)
    choose_from = min(xhatexp.shape[0], xhatexp_boots.shape[0])
    indices = np.random.choice(choose_from, size=num_imgs, 
            replace=False)

    xhatexp_sub = xhatexp[indices, ...]
    xhatexp_boots_sub = xhatexp_boots[indices, ...]
    all_xhat = np.concatenate((xhatexp_sub, xhatexp_boots_sub), axis = 0)
    # plot order
    # 0-4: orig
    # 5-9: boots
    order_ = [7, 2, 3, 9, 0, 8, 6, 5, 4, 1]

    # plot predictions and bootstrapped predictions
    fig, axes = plt.subplots(2*num_imgs, 1, figsize=(6,5*2*num_imgs))
    concat = np.concatenate((xhatexp_sub, xhatexp_boots_sub))
    min_val = np.min(concat)
    max_val = np.max(concat)
    for i, j in enumerate(order_):
        pcm = axes[i].imshow(np.squeeze(all_xhat[j, ...]), cmap='Blues', 
            vmin = min_val, vmax = max_val)
        axes[i].axis('off')
        if j > 4:
            axes[i].set_title('Boots prediction')
        else:
            axes[i].set_title('Orig prediction')
        
        fig.colorbar(pcm, ax=axes[i], extend='both', shrink=.7)
        
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_predictions_boots_quiz_sol.pdf'.format(split)), 
        bbox_inches='tight')

    # plot predictions and bootstrapped predictions
    fig, axes = plt.subplots(2*num_imgs, 1, figsize=(6,5*2*num_imgs))
    concat = np.concatenate((xhatexp_sub, xhatexp_boots_sub))
    min_val = np.min(concat)
    max_val = np.max(concat)
    for i, j in enumerate(order_):
        pcm = axes[i].imshow(np.squeeze(all_xhat[j, ...]), cmap='Blues', 
            vmin = min_val, vmax = max_val)
        axes[i].axis('off')
        fig.colorbar(pcm, ax=axes[i], extend='both', shrink=.7)
        
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_predictions_boots_quiz.pdf'.format(split)), 
        bbox_inches='tight')

    
def plot_timeseries_boots_quiz(xhatexp, xhatexp_boots, out_dir, split, loc=None, 
    seed=1):
    num_imgs=5
    plot_range_start = 0
    plot_range_stop = 500

    # get a subsample to plot
    np.random.seed(seed)
    if loc is None:
        indices_x = np.random.choice(xhatexp.shape[1], size=num_imgs, 
            replace=False)
        indices_y = np.random.choice(xhatexp.shape[2], size=num_imgs, 
            replace=False)
    else:
        indices_x = []
        indices_y = []
        r2 = []
        _, map_xxhatexp, loc_xxhat, loc_xxhatexp = loc
        for entry in sorted(loc_xxhatexp.keys()):
            indices_x.append(entry[0])
            indices_y.append(entry[1])
            r2.append(loc_xxhatexp[entry])
        for entry in sorted(loc_xxhat.keys()):
            indices_x.append(entry[0])
            indices_y.append(entry[1])
            r2.append(map_xxhatexp[entry])

    n = xhatexp.shape[0]
    xhatexp_sub = np.zeros([num_imgs, n])
    xhatexp_boots_sub = np.zeros([num_imgs, xhatexp_boots.shape[0]])

    for i, xy in enumerate(zip(indices_x, indices_y)):
        x, y = xy
        xhatexp_sub[i, :] = xhatexp[:, x, y, 0]
        xhatexp_boots_sub[i, :] = xhatexp_boots[:, x, y, 0]

    xhatexp_all = np.concatenate(
        (xhatexp_sub[:,plot_range_start:plot_range_stop], 
        xhatexp_boots_sub[:,plot_range_start:plot_range_stop]), axis=0)
    order_ = [2, 6, 8, 0, 4, 5, 3, 9, 7, 1]

    # plot input and reconstructions
    fig, axes = plt.subplots(2*num_imgs, 1, figsize=(20,3*num_imgs))
    fig.tight_layout()
    for i, j in enumerate(order_):
        axes[i].plot(xhatexp_all[j, :], color='blue')

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_boots_ts_quiz.pdf'.format(split)), 
        bbox_inches='tight')


    # plot input and reconstructions
    fig, axes = plt.subplots(2*num_imgs, 1, figsize=(20,3*num_imgs))
    fig.tight_layout()
    for i, j in enumerate(order_):
        axes[i].plot(xhatexp_all[j, plot_range_start:plot_range_stop], color='blue')
        if j > 4:
            axes[i].set_title('Boots prediction')
        else:
            axes[i].set_title('Orig prediction')

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir,'{}_boots_ts_quiz_sol.pdf'.format(split)), 
        bbox_inches='tight')

