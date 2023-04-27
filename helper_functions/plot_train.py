import os
from matplotlib import pyplot as plt

def plot_train(k_fold, lang_model, hyperparameters, file_path, acc_training_loss_cmb, acc_training_loss_ner, acc_training_loss_rc, acc_val_loss_cmb, acc_val_loss_ner, acc_val_loss_rc):

    plt.figure()
    plt.plot(acc_training_loss_cmb, label='train_loss_cmb', color = 'b')
    plt.plot(acc_training_loss_ner, label='train_loss_ner', color = 'b', linestyle = '--')
    plt.plot(acc_training_loss_rc, label='train_loss_rc', color = 'b', linestyle = ':')
    plt.plot(acc_val_loss_cmb,label='val_loss_cmb', color = 'r')
    plt.plot(acc_val_loss_ner,label='val_loss_ner', color = 'r', linestyle = '--')
    plt.plot(acc_val_loss_rc,label='val_loss_rc', color = 'r', linestyle = ':')
    plt.xticks(range(0,hyperparameters['epochs']))
    plt.legend()
    plt.title(lang_model + ' - Fold ' + str(k_fold+1))
    plt.grid()
    plt.show

    # Save plot and hyperparameters
    plot_path = file_path + '/plots/' + lang_model + '/'

    # Create the folder if it doesn't exist already
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    plt.savefig(plot_path + 'fold_' + str(k_fold+1) + '_loss_curves.png')