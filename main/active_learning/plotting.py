import matplotlib.pyplot as plt


def plot_al_results(v_results, a_results, batch_size, lr):

    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.05, wspace=0.05)

    axs = gs.subplots(sharex=True)


    for i, method in enumerate(v_results[batch_size][lr].keys()):
        axs[0, 0].plot(v_results[batch_size][lr][method]['train'], label=method)
        axs[1, 0].plot(v_results[batch_size][lr][method]['test'], label=method)
        axs[0, 0].legend(); axs[1, 0].legend()
        axs[0, 0].grid(); axs[1, 0].grid()

        axs[0,0].set_title('Valence')
        axs[0,0].set_ylabel('Training')
        axs[1,0].set_ylabel('Testing')

    for i, method in enumerate(a_results[batch_size][lr].keys()):
        axs[0, 1].plot(a_results[batch_size][lr][method]['train'], label=method)
        axs[1, 1].plot(a_results[batch_size][lr][method]['test'], label=method)
        axs[0, 1].legend(); axs[1, 1].legend()
        axs[0, 1].yaxis.tick_right(); axs[1, 1].yaxis.tick_right()
        axs[0, 1].grid(); axs[1, 1].grid()

        axs[0, 1].set_title('Arousal')

    fig.text(0.5, 0.07, 'Number of Epochs', ha='center')
    fig.text(0.05, 0.5, 'RMSE', ha='center', va='center', rotation='vertical')

    plt.savefig(f'files/results/active_learning_experiments/NEW_al_exp_batch_size_{batch_size}_learning_rate_{lr}_fullsample.png')
    #plt.savefig(f'files/results/active_learning_experiments/al_exp_batch_size_{batch_size}_learning_rate_{lr}.png')
    #plt.show()