import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS, TSNE
import itertools

def plot_metrics(metrics, labels, x_axis=None, basepath=None):
    for metric, label in zip(metrics, labels):
        if x_axis is not None:
            plt.plot(x_axis, metric, label=label)
        else:
            plt.plot(metric, label=label)
    plt.legend()
    savepath = 'metrics.png'
    plt.savefig(savepath)
    if basepath is not None:
        savepath = basepath + savepath
    plt.savefig(savepath)
    plt.close()


def plot_scatter(metrics, labels, savepath=None):
    assert len(metrics) == 2
    fig, ax = plt.subplots()
    c = [i for i in range(len(metrics[0]))]
    pcm = ax.scatter(metrics[0], metrics[1], c=c, s=20, cmap='viridis')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.savefig('info_plane.png')
    else:
        plt.show()
    plt.close()


def plot_multi_trials(multi_metrics, series_labels, sizes, ylabel=None, xlabel=None, colors=None, filename=None):
    font = {'family': 'normal',
            'size': 20}

    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = 0

    annotations = []
 
    reset_period = 8
    color_cycle = itertools.cycle(plt.cm.magma(np.linspace(0, 1, 4)))
    shape_cycle = itertools.cycle(['o', 's', 'x'])
    periodx = []
    periody = []
    periodx_std = []
    periody_std = []
    plot_eng_comp = False
    for metric_x, metric_y, label, s in zip(multi_metrics[0], multi_metrics[1], series_labels, sizes):
        if idx % reset_period == 0:
            c = next(color_cycle)
            m = next(shape_cycle)
            periodx = []
            periody = []
            periodx_std = []
            periody_std = []
        if colors is not None:
            c = colors[idx]

        yerr = multi_metrics[2][idx] if len(multi_metrics) == 3 else None

        # Error-bar version
        xstd = np.std(metric_x)
        ystd = np.std(metric_y)
        xmean = np.mean(metric_x)
        ymean = np.mean(metric_y)
        periodx.append(xmean)
        periody.append(ymean)
        periody_std.append(xstd)
        periodx_std.append(xstd)
        pcm = ax.scatter(xmean, ymean, s=s, label=label, color=c, marker=m)
        plt.errorbar(xmean, ymean, xerr=xstd, yerr=ystd, color=c)
        if colors is None:
            # And add a dashed line between the series
            plt.plot(periodx, periody, 'k--', alpha=1.0)
            print()
            for i in range(len(periodx)):
                print(str(periodx[i]) + " (" + str(periodx_std[i]) + ")")
                # print(str(periody[i]) + " (" + str(periody_std[i]) + ")")
        if yerr is not None:
            plt.errorbar(metric_x, metric_y, yerr=yerr, fmt='o')
        if idx < len(annotations):
            xdiff = 0.1 if idx != 1 else 0
            ax.annotate(annotations[idx], (xmean - xdiff, ymean - 0.2))
        idx += 1
    xlabel = xlabel if xlabel is not None else 'Complexity (nats)'
    # xlabel = xlabel if xlabel is not None else '$\lambda_I$'
    if plot_eng_comp:
        plt.axvline(x=2.08, color='g')
        ax.annotate('Eng. responses', (2.08, 0.05), color='g')
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
   
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    print("Saving to", filename)
    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_multi_metrics(multi_metrics, labels=None, file_root=''): 
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    fig, ax = plt.subplots()
    comm_to_color = {0: 'xkcd:blue', 1: 'xkcd:violet', 2: 'xkcd:red'}
    for comm_type, metrics in multi_metrics.items():
        num_metrics = len(metrics) - 1  # Last one is just epoch
        epochs = metrics[-1]
        overalls = [[] for _ in range(num_metrics)]
        # Iterate over evaluation sets
        for eval_idx, eval_type in enumerate(metrics[:-1]):
            # Now iterate over trials.
            color = comm_to_color[eval_idx]
            # linestyle = 'solid' if eval_idx < num_metrics / 2 else 'dashed'
            linestyle = 'solid'
            for trial_idx in range(len(eval_type)):
                accs = metrics[eval_idx][trial_idx]
                plt.plot(epochs, accs, color, linestyle='dashed', alpha=0.2)
                overalls[eval_idx].append(accs)
            mean_overall = np.median(np.vstack(overalls[eval_idx]), axis=0)
            std = np.std(np.vstack(overalls[eval_idx]), axis=0)
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            # print("Epoch", epochs)
            # print("Median overall", ", ".join([str(elt)[:5] for elt in mean_overall]))
            # print("Std overall", ", ".join([str(elt / np.sqrt(5))[:5] for elt in std]))
            plt.plot(epochs, mean_overall, color, linestyle=linestyle, label=labels[eval_idx])
            
    plt.legend(loc='lower right')
    plt.xlabel('Training epoch')
    plt.ylabel('Success rate')
    plt.ylim(0.0, 1.02)
    
    plt.tight_layout()
    plt.savefig(file_root + 'trials.png')
    plt.close()


def invert_permutation(p):
    p = np.asanyarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


# Helper function from stackoverflow to adjust a color's lightness.
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_naming(all_data, true_names, viz_method, labels=None, savepath=None, plot_all_colors=False):
    # The only difference between different plotting methods is the embedding version. Coloring, labeling, etc.
    # are all the same.
    assert viz_method in ['mds', 'tsne'], "Only support mds or tsne visualization"
    is_mds = viz_method == 'mds'
    embedder = MDS(n_components=2, random_state=0) if is_mds else TSNE(n_components=2, learning_rate='auto', random_state=0)
    catted = np.vstack(all_data)
    max_entries = 1000
    if catted.shape[0] > max_entries:
        print("Warning, data very long. Truncating")
        catted = catted[:max_entries]
    # Sort the data for reproducibility.
    sort_permutation = catted[:, 0].argsort()
    undo_permutation = invert_permutation(sort_permutation)
    catted = catted[sort_permutation]
    similarities = euclidean_distances(catted.astype(np.float64))
    transformed = embedder.fit_transform(similarities)
    transformed = transformed[undo_permutation]  # Undo the permutation for plotting, so it lines up with labels.
    x = transformed[:, 0]
    y = transformed[:, 1]
    # Rescale to be within a smaller range
    x = x / (max(x) - min(x))
    x = x - min(x)
    y = y / (max(y) - min(y))
    y = y - min(y)
    cmap = plt.get_cmap('hsv')
    colors = cmap(x)
    # Transform color by the y coordinate as well to make lower values darker
    darkness = y / 2 + 0.5
    for i, dark in enumerate(darkness):
        colors[i, :3] = adjust_lightness(colors[i, :3], dark)
    if plot_all_colors:
        # No labels, just color things.
        fig, ax = plt.subplots()
        pcm = ax.scatter(x, y, s=20, color=colors, edgecolors='black')
        plt.savefig('all_colors_' + viz_method + '.png')
        plt.close()
    fig, ax = plt.subplots()
    last_idx = 0
    print(len(all_data))
    for data_idx, data in enumerate(all_data):
        label = None if labels is None else labels[data_idx]
        sub_x = x[last_idx: last_idx + len(data)]
        sub_y = y[last_idx: last_idx + len(data)]
        sub_colors = colors[last_idx: last_idx + len(data)]
        mean_color = np.mean(sub_colors, axis=0)
        #if label == -1 or label == '-1':
        #    pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', facecolors='none', edgecolors='black')
        #else:
        pcm = ax.scatter(sub_x, sub_y, s=20, marker='o', color=mean_color)
        for i in range(len(sub_x)):
            ax.text(sub_x[i], sub_y[i], true_names[i])
            
        # And plot the centroids (in the transformed coordinate frames) for each label
        if len(sub_x) > 2:
            center_x = np.mean(sub_x)
            center_y = np.mean(sub_y)
            pcm = ax.scatter([center_x], [center_y], s=200, marker='*', color=mean_color, edgecolors='black')
        last_idx += len(data)
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
