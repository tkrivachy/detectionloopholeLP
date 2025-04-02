import numpy as np
import pickle

from targets import target_distribution_gen_all_2D

class Config:
    """ Config is just a collection of all metadata which can be accessed from auxiliary files as well. """
    def __init__(self):
        ## Note that here I generate target distributions with a custom target_distribution_gen_all fn,
        ## however, in general, you can input any np.array as self.target_distributions with shape (number of distributions, size of a single distribution)

        self.description = "High-dim NOON N=2."
        # Define target distributions to sweep through.
        ## Set up custom target distribution generator function
        self.target_distr_name = "noon2-allphotonslost-click-noclick" # check targets.py for possible names
        self.param_range = np.linspace(0.8, 0.8, 1) #y axis on imshow; t
        self.param_range2 = np.linspace(0, 1, 10) #x axis on imshow; noise (eta)

        ## Set target distributions and their ids
        self.target_distributions = target_distribution_gen_all_2D(self.target_distr_name,  self.param_range, self.param_range2)
        self.target_ids = np.indices((self.param_range.shape[0], self.param_range2.shape[0]))#self.param_range
        self.target_ids = np.moveaxis(self.target_ids,0,2)

        base_outputsize = 4
        self.a_outputsize = base_outputsize # Number of output bits for Alice
        self.b_outputsize = base_outputsize # Number of output bits for Bob
        self.c_outputsize = base_outputsize # Number of output bits for Charlie

        # Neural network parameters
        self.latin_depth = 3
        self.latin_width = 30

        # Training procedure parameters
        self.batch_size = 10000
        self.no_of_batches = 5000 # How many batches to go through during training.
        self.weight_init_scaling = 3.#10. # default is 1. Set to larger values to get more variance in initial weights.
        self.optimizer = 'adadelta'
        self.lr = 0.5
        self.decay = 0.001
        self.momentum = 0.25
        self.loss = 'kl'

        # Initialize some variables
        self.euclidean_distances = np.ones((self.target_ids.shape[0],self.target_ids.shape[1]))
        self.distances = np.ones((self.target_ids.shape[0],self.target_ids.shape[1]))
        self.distributions = np.ones_like(self.target_distributions)/(self.target_distributions.shape[-1]) # learned distributions
        self.set_starting_points(fresh_start = True) # Don't change fresh start here. If you want to change it, change it in your main python script (required due to dependencies).
        self.sweep_id = 0 # id of sweep. Only used if multiple sweeps are done.

        # Neural network parameters that I don't change much
        self.no_of_validation_batches = 100 # How many batches to go through in each validation step. If batch size is large, 1 should be enough.
        self.change_batch_size(self.batch_size) #updates test batch size
        self.greek_depth = 0 # set to 0 if trivial neural networks at sources
        self.greek_width = 1
        self.activ = 'relu' # activation for most of NN
        self.activ2 = 'softmax' # activation for last dense layer
        self.kernel_reg = None

        self.nan_memory = np.zeros((60, self.target_ids.shape[0],self.target_ids.shape[1])) # memory of where nans have been found

    def change_p_target(self,id,jd):
        self.p_target = self.target_distributions[id,jd,:] # current target distribution
        self.p_id = self.target_ids[id,jd,:] # current target id
        self.y_true = np.array([self.p_target for _ in range(self.batch_size)]) # keras technicality that in supervised training y_true should have batch_size dimension as well
        self.start_from = self.start_from_array[id,jd]
        self.savebestpath = './0_saved_models/best_'+str(id).zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'_'+str(jd).zfill(int(np.ceil(np.log10(self.target_ids.shape[1]))))+'.hdf5'

    def change_batch_size(self,new_batch_size):
        self.batch_size = new_batch_size
        self.batch_size_test = int(self.no_of_validation_batches*self.batch_size) # in case we update batch_size we should also update the test batch size

    def save(self,name):
        with open('./0_saved_configs/'+name, 'wb') as f:
            pickle.dump(self, f)

    def set_starting_points(self,check_neighbors = False, fresh_start=False):
        """ Sets where to continue learning from by looking at neighbors' models.
        Useful if you want to smooth out distance function. """
        # If fresh_start, then put None for start_from, which results in not loading a model but doing random initialization fo weights.
        if fresh_start:
            self.start_from_array = np.array([[None for _ in range(self.target_distributions.shape[1])] for _ in range(self.target_distributions.shape[0])])
            print("Fresh start!")
        elif not check_neighbors:
            print("Not checking neighbors, just continuing from previous best.")
            self.start_from_array = np.array([['./0_saved_models/best_'+'0'.zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'_'+'0'.zfill(int(np.ceil(np.log10(self.target_ids.shape[1]))))+'.hdf5' for _ in range(self.target_ids.shape[1])] for _ in range(self.target_ids.shape[0])])
            for id in range(self.target_ids.shape[0]):
                for jd in range(self.target_ids.shape[1]):
                    self.start_from_array[id,jd] = './0_saved_models/best_'+str(id).zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'_'+str(jd).zfill(int(np.ceil(np.log10(self.target_ids.shape[1]))))+'.hdf5'
        else:
            print("Checking neighbors...")
            # Set starting ids to be just their own ids by default.
            starting_ids = np.arange(self.target_distributions.shape[0])

            # Change this default if either broadness_left or broadness_right is nonzero
            if check_neighbors:
                from utils_nn import np_distance # have to import here due to dependency issues
                # Initialize to all 0 models.
                self.start_from_array = np.array([['./0_saved_models/best_'+'0'.zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'_'+'0'.zfill(int(np.ceil(np.log10(self.target_ids.shape[1]))))+'.hdf5' for _ in range(self.target_ids.shape[1])] for _ in range(self.target_ids.shape[0])])
                # For each target distribution, check all other learned distributions to see which is closest. Update start_from_array based on that
                for id in range(self.target_ids.shape[0]):
                    for jd in range(self.target_ids.shape[1]):
                        this_target = self.target_distributions[id,jd]
                        cross_distances = np.ones_like(self.distances)
                        for i in range(self.target_ids.shape[0]):
                            for j in range(self.target_ids.shape[1]):
                                cross_distances[i,j] = np_distance(this_target,self.distributions[i,j,:])
                        temp_npwhere = np.where(cross_distances == np.amin(cross_distances))
                        best_i = temp_npwhere[0][0]
                        best_j = temp_npwhere[1][0]
                        self.start_from_array[id,jd] = './0_saved_models/best_'+str(best_i).zfill(int(np.ceil(np.log10(self.target_ids.shape[0]))))+'_'+str(best_j).zfill(int(np.ceil(np.log10(self.target_ids.shape[1]))))+'.hdf5'
                        print("\nStarting from index {} instead of {}.".format([best_i,best_j],[id,jd]))
            else:
                print("Starting points have not been changed.")
            # Generate model names from starting_ids.
            self.start_from_array = np.array(['./0_saved_models/best_'+str(i).zfill(int(np.ceil(np.log10(self.target_distributions.shape[0]))))+'.hdf5' for i in starting_ids])

def load_config(name):
    with open('./0_saved_configs/'+name, 'rb') as f:
        temp = pickle.load(f)
    return temp

def initialize():
    """ Initializes a Config class as a global variable pnn (for Parameters of Neural Network).
    The pnn object should be accessible and modifiable from all auxiliary files.
    """
    global pnn
    pnn = Config()
    pnn.save('initial_pnn')
