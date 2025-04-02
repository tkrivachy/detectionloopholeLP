import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda

import config as cf


# I used this to configure my GPU memory usage. Uncomment if you want to use it.
# Memory was restricted since like this I could run multiple trainings in parallel.

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

def build_model():
    cf.pnn.inputsize = 3 # Number of hidden variables, e.g. alpha, beta, gamma
    """ Build NN for triangle """
    # Hidden variables as inputs.
    inputTensor = Input((cf.pnn.inputsize,))

    # Group input tensor according to whether alpha, beta or gamma hidden variable.
    group_alpha = Lambda(lambda x: x[:,:1], output_shape=((1,)))(inputTensor)
    group_beta = Lambda(lambda x: x[:,1:2], output_shape=((1,)))(inputTensor)
    group_gamma = Lambda(lambda x: x[:,2:3], output_shape=((1,)))(inputTensor)

    # Neural network at the sources, for pre-processing (e.g. for going from uniform distribution to non-uniform one)
    ## Note that in the example code greek_depth is set to 0, so this part is trivial.
    for _ in range(cf.pnn.greek_depth):
        group_alpha = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_alpha)
        group_beta = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_beta)
        group_gamma = Dense(cf.pnn.greek_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg)(group_gamma)

    # Route hidden variables to visibile parties Alice, Bob and Charlie
    group_a = Concatenate()([group_beta,group_gamma])
    group_b = Concatenate()([group_gamma,group_alpha])
    group_c = Concatenate()([group_alpha,group_beta])

    # Neural network at the parties Alice, Bob and Charlie.
    ## Note: increasing the variance of the initialization seemed to help in some cases, especially when the number if outputs per party is 4 or more.
    kernel_init = tf.keras.initializers.VarianceScaling(scale=cf.pnn.weight_init_scaling, mode='fan_in', distribution='truncated_normal', seed=None)
    for _ in range(cf.pnn.latin_depth):
        group_a = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_a)
        group_b = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_b)
        group_c = Dense(cf.pnn.latin_width,activation=cf.pnn.activ, kernel_regularizer=cf.pnn.kernel_reg, kernel_initializer = kernel_init)(group_c)

    # Apply final softmax layer
    group_a = Dense(cf.pnn.a_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_a)
    group_b = Dense(cf.pnn.b_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_b)
    group_c = Dense(cf.pnn.c_outputsize,activation=cf.pnn.activ2, kernel_regularizer=cf.pnn.kernel_reg)(group_c)

    outputTensor = Concatenate()([group_a,group_b,group_c])

    model = Model(inputTensor,outputTensor)
    return model

def np_euclidean_distance(p,q=0):
    """ Euclidean distance, useful for plotting results."""
    return np.sqrt(np.sum(np.square(p-q),axis=-1))

def np_distance(p,q=0):
    """ Same as the distance used in the loss function, just written for numpy arrays.
    Implemented losses:
        l2: Euclidean distance (~Mean Squared Error)
        l1: L1 distance (~Mean Absolute Error)
        kl: Kullback-Liebler divergence (relative entropy)
        js: Jensen-Shannon divergence (see https://arxiv.org/abs/1803.08823 pg. 94-95). Thanks to Askery A. Canabarro for the recommendation.
    """
    if cf.pnn.loss.lower() == 'l2':
        return np.sum(np.square(p-q),axis=-1)
    elif cf.pnn.loss.lower() == 'l1':
        return 0.5*np.sum(np.abs(p-q),axis=-1)
    elif cf.pnn.loss.lower() == 'kl':
        p = np.clip(p, K.epsilon(), 1)
        q = np.clip(q, K.epsilon(), 1)
        return np.sum(p * np.log(np.divide(p,q)), axis=-1)
    elif cf.pnn.loss.lower() == 'js':
        p = np.clip(p, K.epsilon(), 1)
        q = np.clip(q, K.epsilon(), 1)
        avg = (p+q)/2
        return np.sum(p * np.log(np.divide(p,avg)), axis=-1) + np.sum(q * np.log(np.divide(q,avg)), axis=-1)

def keras_distance(p,q):
    """ Distance used in loss function.
    Implemented losses:
        l2: Euclidean distance (~Mean Squared Error)
        l1: L1 distance (~Mean Absolute Error)
        kl: Kullback-Liebler divergence (relative entropy)
        js: Jensen-Shannon divergence (see https://arxiv.org/abs/1803.08823 pg. 94-95). Thanks to Askery A. Canabarro for the recommendation.
    """
    if cf.pnn.loss.lower() == 'l2':
        return K.sum(K.square(p-q),axis=-1)
    elif cf.pnn.loss.lower() == 'l1':
        return 0.5*K.sum(K.abs(p-q), axis=-1)
    elif cf.pnn.loss.lower() == 'kl':
        p = K.clip(p, K.epsilon(), 1)
        q = K.clip(q, K.epsilon(), 1)
        return K.sum(p * K.log(p / q), axis=-1)
    elif cf.pnn.loss.lower() == 'js':
        p = K.clip(p, K.epsilon(), 1)
        q = K.clip(q, K.epsilon(), 1)
        avg = (p+q)/2
        return K.sum(p * K.log(p / avg), axis=-1) + K.sum(q * K.log(q / avg), axis=-1)


def customLoss_distr(y_pred):
    """ Converts the output of the neural network to a probability vector.
    That is from a shape of (batch_size, a_outputsize + b_outputsize + c_outputsize) to a shape of (a_outputsize * b_outputsize * c_outputsize,)
    """
    a_probs = y_pred[:,0:cf.pnn.a_outputsize]
    b_probs = y_pred[:,cf.pnn.a_outputsize : cf.pnn.a_outputsize + cf.pnn.b_outputsize]
    c_probs = y_pred[:,cf.pnn.a_outputsize + cf.pnn.b_outputsize : cf.pnn.a_outputsize + cf.pnn.b_outputsize + cf.pnn.c_outputsize]

    a_probs = K.reshape(a_probs,(-1,cf.pnn.a_outputsize,1,1))
    b_probs = K.reshape(b_probs,(-1,1,cf.pnn.b_outputsize,1))
    c_probs = K.reshape(c_probs,(-1,1,1,cf.pnn.c_outputsize))

    probs = a_probs*b_probs*c_probs
    probs = K.mean(probs,axis=0)
    probs = K.flatten(probs)
    return probs

def customLoss(y_true,y_pred):
    """ Custom loss function."""
    # Note that y_true is just batch_size copies of the target distributions. So any row could be taken here. We just take 0-th row.
    return keras_distance(y_true[0,:], customLoss_distr(y_pred))

# Set up generator for X and Y data
training_mean = 0.5
training_sigma = 0.28867513459 #= np.sqrt(1/12)

def generate_xy_batch_genfn():
    while True:
        x_batch = np.divide((np.random.random((cf.pnn.batch_size, cf.pnn.inputsize)) - training_mean), training_sigma)
        y_batch = np.tile(cf.pnn.y_true[0, :], (cf.pnn.batch_size, 1))
        yield (x_batch, y_batch)

def generate_x_test_genfn():
    while True:
        temp = np.divide((np.random.random((cf.pnn.batch_size_test, cf.pnn.inputsize)) - training_mean),training_sigma)
        yield temp

def single_evaluation(model):
    """ Evaluates the model and returns the resulting distribution as a numpy array. """
    generate_x_test = tf.data.Dataset.from_generator(generate_x_test_genfn, name='Gen_x_test', output_signature=tf.TensorSpec(shape=(None, cf.pnn.inputsize), dtype=tf.float32))
    test_pred = model.predict(generate_x_test, steps=1, verbose=0)
    result = K.eval(customLoss_distr(test_pred))
    return result

def single_run():
    """ Runs training algorithm for a single target distribution. Returns model."""
    # Model and optimizer related setup.
    K.clear_session()
    model = build_model()
    if cf.pnn.start_from is not None:
        print("LOADING MODEL WEIGHTS FROM", cf.pnn.start_from)
        model = load_model(cf.pnn.start_from,custom_objects={'customLoss': customLoss})

    if cf.pnn.optimizer.lower() == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate = cf.pnn.lr, rho=0.95)
    elif cf.pnn.optimizer.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=cf.pnn.lr, decay=cf.pnn.decay, momentum=cf.pnn.momentum, nesterov=True)
        print("\n\nWARNING!!! Optimizer {} not recognized. Please implement it if you want to use it. Using SGD instead.\n\n".format(cf.pnn.optimizer))
        cf.pnn.optimizer = 'sgd' # set it for consistency.

    model.compile(loss=customLoss, optimizer = optimizer, metrics=[])

    # Set up data generators
    generate_xy_batch = tf.data.Dataset.from_generator(
        generate_xy_batch_genfn,
        output_signature=(
            tf.TensorSpec(shape=(cf.pnn.batch_size, cf.pnn.inputsize), dtype=tf.float32),
            tf.TensorSpec(shape=(cf.pnn.batch_size, cf.pnn.y_true.shape[1]), dtype=tf.float32)
        )
    )#.repeat()  # Ensure the dataset repeats indefinitely

    # Fit model
    model.fit(generate_xy_batch, steps_per_epoch=cf.pnn.no_of_batches, epochs=1, verbose=1, validation_data=generate_xy_batch, validation_steps=cf.pnn.no_of_validation_batches, class_weight=None, shuffle=False, initial_epoch=0)
    return model

def compare_models(model1,model2):
    """ Evaluates two models for p_target distribution and return one which is closer to it. 
    I don't seem to be using this fn at all..."""
    result1 = single_evaluation(model1)
    result2 = single_evaluation(model2)
    dist1 = np_distance(result1, cf.pnn.p_target)
    dist2 = np_distance(result2, cf.pnn.p_target)
    # if one of the models has a distance of nan, then return the other model.
    if np.isnan(dist1):
        return model2, 2
    elif np.isnan(dist2):
        return model1, 1
    elif  dist1 < dist2:
        return model1, 1
    else:
        return model2, 2

def update_results(model_new,i,j):
    """ Updates plots and results if better than the one I loaded the model from in this round.
    If I am in last sample of the sweep I will plot no matter one, so that there is at least one plot per sweep.
    """
    result_new = single_evaluation(model_new)
    distance_new = np_distance(result_new, cf.pnn.p_target)

    # Decide whether to use new or old model.
    if cf.pnn.start_from is not None: # skips this comparison if I was in a fresh_start
        try:
            model_old = load_model('./0_saved_models/best_'+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_ids.shape[0]))))+'_'+str(j).zfill(int(np.ceil(np.log10(cf.pnn.target_ids.shape[1]))))+'.hdf5', custom_objects={'customLoss': customLoss})
            result_old = single_evaluation(model_old)
            distance_old = np_distance(result_old, cf.pnn.p_target)
            # If new model is worse or if new model is nan, then keep the old model.
            if (distance_new > distance_old) or np.isnan(distance_new):
                print("Moving on. With old model distance is at {}.".format(distance_old))
                result = result_old
                model = model_old
                distance = distance_old
                if np.isnan(distance_new):
                    cf.pnn.nan_memory[cf.pnn.sweep_id,i,j] = 1
                    # save to file
                    np.save("./0_saved_results/nan_memory.npy",cf.pnn.nan_memory)
            else:
                print("Distance imporved! Distance with new model:", distance_new)
                result = result_new
                model = model_new
                distance = distance_new
        except FileNotFoundError:
            print("This distance:", distance_new)
            result = result_new
            model = model_new
            distance = distance_new
    else:
        print("This distance:", distance_new)
        result = result_new
        model = model_new
        distance = distance_new

    # Update results
    model.save(cf.pnn.savebestpath)
    cf.pnn.distributions[i,j,:] = result
    cf.pnn.distances[i,j] = distance
    cf.pnn.euclidean_distances[i,j] = np_euclidean_distance(result, cf.pnn.p_target)
    np.save("./0_saved_results/target_distributions.npy",cf.pnn.target_distributions)
    np.save("./0_saved_results/distributions.npy",cf.pnn.distributions)
    np.save("./0_saved_results/distances.npy",cf.pnn.distances)
    np.save("./0_saved_results/euclidean_distances.npy",cf.pnn.euclidean_distances)

    # Plot distances
    plt.clf()
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('D(p_target,p_machine)')
    pos = ax.imshow(cf.pnn.euclidean_distances ,extent=[cf.pnn.param_range2[0],cf.pnn.param_range2[-1], cf.pnn.param_range[-1],cf.pnn.param_range[0]])
    #ax.set_aspect('equal')

    #cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    #cax.get_xaxis().set_visible(False)
    #cax.get_yaxis().set_visible(False)
    #cax.patch.set_alpha(0)
    #cax.set_frame_on(False)
    plt.colorbar(pos, ax=ax) #, orientation='vertical',
    plt.savefig("./0_figs_training_sweeps/sweep"+str(cf.pnn.sweep_id)+".png")
    """
    plt.title("D(p_target,p_machine)")
    plt.plot(cf.pnn.target_ids,cf.pnn.euclidean_distances, 'ro')
    if i!=0 and cf.pnn.sweep_id==0:
        plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.euclidean_distances))[-2]*1.2)
    else:
        plt.ylim(bottom=0,top = np.sort(np.unique(cf.pnn.euclidean_distances))[-1]*1.2)
    plt.savefig("./0_figs_training_sweeps/sweep"+str(cf.pnn.sweep_id)+".png")
    """
    # Plot distributions
    plt.clf()
    plt.plot(cf.pnn.p_target,'ro',markersize=5)
    plt.plot(result,'gs',alpha = 0.85,markersize=5)
    plt.title("Target distr. (in red)\n{} {}".format(cf.pnn.target_distr_name, cf.pnn.target_ids[i,j,:]))
    plt.ylim(bottom=0,top=max(cf.pnn.p_target)*1.2)
    plt.savefig("./0_figs_distributions/target_"+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_ids.shape[0]))))+'_'+str(j).zfill(int(np.ceil(np.log10(cf.pnn.target_ids.shape[1]))))+".png")

    # Plot strategies (only turn on if you're really interested, since it takes quite a bit of time to update in each step!)
    #plot_strategies(i)

def plot_strategies(i):
    sample_size = 3000 #how many hidden variable triples to sample from
    random_sample_size = 5 #for each hidden variable triple, how many times to sample from strategies.
    alpha_value = 0.25# 3/random_sample_size #opacity of dots. 0.1 or 0.25 make for nice paintings.
    markersize = 5000/np.sqrt(sample_size)

    modelpath = './0_saved_models/best_'+str(i).zfill(int(np.ceil(np.log10(cf.pnn.target_distributions.shape[0]))))+'.hdf5'

    input_data = generate_x_test_genfn()
    inputs = next(input_data)
    inputs = inputs[:sample_size,:]

    K.clear_session()
    model = load_model(modelpath,custom_objects={'customLoss': customLoss})
    y = model.predict(inputs)

    y_a = y[:,0:2]
    y_b = y[:,2:4]
    y_c = y[:,4:6]

    y_a = np.array([np.random.choice(np.array([0,1]),p=y_a[i,:], size = random_sample_size) for i in range(y_a.shape[0])]).reshape(random_sample_size*sample_size)
    y_b = np.array([np.random.choice(np.array([0,1]),p=y_b[i,:], size = random_sample_size) for i in range(y_b.shape[0])]).reshape(random_sample_size*sample_size)
    y_c = np.array([np.random.choice(np.array([0,1]),p=y_c[i,:], size = random_sample_size) for i in range(y_c.shape[0])]).reshape(random_sample_size*sample_size)

    training_mean = 0.5
    training_sigma = np.sqrt(1/12)
    inputs = inputs* training_sigma + training_mean
    # Tile and reshape since we sampled random_sample_size times from each input.
    inputs = np.array(np.array([np.tile(inputs[i,:],(random_sample_size,1)) for i in range(inputs.shape[0])])).reshape(random_sample_size*sample_size,3)

    alphas = inputs[:,0]
    betas = inputs[:,1]
    gammas = inputs[:,2]
    inputs_a = np.stack((betas,gammas)).transpose()
    inputs_b = np.stack((alphas,gammas)).transpose()
    inputs_c = np.stack((alphas,betas)).transpose()

    colordict = {0:'red',1:'green',2:'blue',3:'yellow'}
    colors_alice = [colordict[i] for i in y_a]
    colors_bob = [colordict[i] for i in y_b]
    colors_charlie = [colordict[i] for i in y_c]

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='0',
                              markerfacecolor='red', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='1',
                                markerfacecolor='green', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='2',
                                markerfacecolor='blue', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='3',
                                markerfacecolor='yellow', markersize=8)]

    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    plt.subplot(2,2,1)
    plt.scatter(inputs_a[:,0],inputs_a[:,1], color = colors_alice, alpha=alpha_value, s = markersize)
    plt.gca().invert_yaxis()
    plt.title('Response of Alice to her inputs.')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\gamma$')

    plt.subplot(2,2,2)
    plt.scatter(inputs_b[:,0],inputs_b[:,1], color = colors_bob, alpha=alpha_value, s = markersize)
    plt.gca().invert_yaxis()
    plt.title('Response of Bob to his inputs.')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\gamma$')

    plt.subplot(2,2,3)
    plt.scatter(inputs_c[:,1],inputs_c[:,0], color = colors_charlie, alpha=alpha_value, s = markersize)
    plt.gca().invert_yaxis()
    plt.title('Response of Charlie to his inputs.')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\alpha$')

    plt.subplot(2,2,4)
    plt.plot(cf.pnn.target_distributions[i,:],'ro',markersize=5)
    plt.plot(K.eval(customLoss_distr(y)),'gs',alpha = 0.85,markersize=5)
    plt.title('Target (red) and learned (green) distributions')
    plt.xlabel('outcome')
    plt.ylabel('probability of outcome')

    fig.suptitle(cf.pnn.target_distr_name +', distribution no. '+str(i), fontsize = 14)
    #fig.legend(handles=legend_elements, loc='lower right',bbox_to_anchor = (0.75,0.25))
    fig.legend(handles=legend_elements, loc='upper right')
    plt.savefig('./0_figs_strategies/strat_'+str(i))
