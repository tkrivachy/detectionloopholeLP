import numpy as np
import os
# import psutil

import config as cf
from utils_nn import single_run, update_results

np.set_printoptions(linewidth=200)

if __name__ == '__main__':
    # Create directories for saving stuff
    for dir in ['0_saved_models', '0_saved_results', '0_saved_configs', '0_figs_distributions', '0_figs_training_sweeps']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    cf.initialize()

    # Try picking up from where training was left off. If not possible, then don't load anything, just start fresh.
    try:
        cf.pnn = cf.load_config("most_recent_pnn")
        print("\nPicking up from where we left off!\n")
        starting_sweep_id = cf.pnn.sweep_id # had +1 in old times when i only saved the sweep id
        starting_i = cf.pnn.current_i
        starting_j = cf.pnn.current_j

        if starting_j==0:
            starting_i -= 1
            starting_j = cf.pnn.target_distributions.shape[1]-1
        else:
            starting_j -= 1
    except FileNotFoundError:
        print("\nStarting fresh!\n")
        starting_sweep_id = cf.pnn.sweep_id
        starting_i = cf.pnn.target_distributions.shape[0]-1
        starting_j = cf.pnn.target_distributions.shape[1]-1

    # Each sweep goes through all distributions. We use different optimizer parameters in different sweeps, and load previous models.
    for sweep_id in range(starting_sweep_id, 25):
        cf.pnn.sweep_id = sweep_id

        # Set parameters of this training sweep.

        ## For a few sweeps, reinitialize completely.
        if cf.pnn.sweep_id<=1:
            cf.pnn.set_starting_points(fresh_start=True)

        ## Then for a few sweeps, start from previous best model for that distribution.
        if cf.pnn.sweep_id>1:
            cf.pnn.set_starting_points(check_neighbors=False, fresh_start=False)

        ## Change to SGD.
        if cf.pnn.sweep_id == 2:
            cf.pnn.optimizer = 'sgd'
            cf.pnn.lr = 1.0
            cf.pnn.decay = 0.0
            cf.pnn.momentum = 0.2

        ## Gradually reduce learning rate for SGD for fine-tuning.
        if cf.pnn.sweep_id > 2:
            cf.pnn.optimizer = 'sgd' # good practice to include this to make sure we're still doing SGD.
            cf.pnn.lr = cf.pnn.lr * 0.4

        if cf.pnn.sweep_id == 7:
            cf.pnn.optimizer = 'adadelta'
        if cf.pnn.sweep_id >= 7:
            cf.pnn.loss = 'l2'
        if cf.pnn.sweep_id == 9:
            cf.pnn.optimizer = 'sgd'
            cf.pnn.lr = 1.0

        ## Check neighbors and learn from them!
        # if cf.pnn.sweep_id >= 15:
        #     cf.pnn.set_starting_points(check_neighbors=False, fresh_start=False)
        # if cf.pnn.sweep_id == 15:
        #     cf.pnn.lr = 0.01

        # Run single sweep
        # Loop through parameters. Convention is to start from right, since that is the least noisy distribution if I do a noise scan.
        for i in range(starting_i,-1,-1):
            for j in range(starting_j,-1,-1):
                # I used this to check if I was running out of memory. Sometimes memory overflows, and I don't know why.
                # However, since progress is saved and loaded automatically, it is not a big deal if it happens.

                # print("Availible memory ratio:", psutil.virtual_memory().available / psutil.virtual_memory().total)
                # if psutil.virtual_memory().available / psutil.virtual_memory().total < 0.20:
                #         print("Less than 80% memory available. Exiting.")
                #         exit()

                if cf.pnn.euclidean_distances[i,j] < 0.0001:
                    print("Skipping because distance is less than 0.0001.")
                else:
                    # Set up new distribution
                    cf.pnn.change_p_target(i,j)
                    print('\nCurrent directory: {}'.format(os.getcwd()))
                    print('In sweep {}.\nAt round {},{} of {},{} (decreasing!), with distribution {} of param {}. Target distribution:\n{}'.format(cf.pnn.sweep_id,i,j,cf.pnn.target_distributions.shape[0]-1,cf.pnn.target_distributions.shape[1]-1,cf.pnn.target_distr_name, cf.pnn.target_ids[i,j,:],cf.pnn.p_target))

                    cf.pnn.current_i=i
                    cf.pnn.current_j=j

                    # Run model
                    model = single_run()
                    # If we loaded weights from somewhere, then compare new distance to previous one in order to know whether new model is better than previous one.
                    update_results(model,i,j)
                    # Save config of the most recently finished sweep. We will continue from here if train_multiple_sweeps is run again.
                    cf.pnn.save("most_recent_pnn")
        starting_i = cf.pnn.target_distributions.shape[0]-1
        starting_j = cf.pnn.target_distributions.shape[1]-1
        # I used to also save the config of each sweep so that I would know what was going on in hindsight. However it takes up a lot of space. Probably should implement with JSON. (or history variables in Config)
        #cf.pnn.save("sweep_"+str(cf.pnn.sweep_id)+"_pnn")
