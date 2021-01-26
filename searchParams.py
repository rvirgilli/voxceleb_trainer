import pickle, time, datetime, shutil, os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import trainSpeakerNet

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

space = {
    ## Data loader
    'max_frames': 200,
    'eval_frames': 300,
    'batch_size': 200,
    'max_seg_per_spk': 500,
    'nDataLoaderThread': 6,
    'augment': False,

    ## Training details
    'test_interval': 10,
    'max_epoch': 500,
    'trainfunc': '',

    ## Optimizer
    'optimizer': 'adam',
    'scheduler': 'steplr',
    'lr': 0.001,
    'lr_decay': 0.95,
    'weight_decay': 0,

    ## Loss functions
    'hard_prob': 0.5,
    'hard_rank': 10,
    'margin': 0.1,
    'scale': 30,
    'nPerSpeaker': 1,
    'nClasses': 5994,

    ## Load and save
    'initial_model':'',
    'save_path':'exps/exp1',

    ## Training and test data
    'train_list':'data/train_list.txt',
    'test_list' :'data/test_list.txt',
    'train_path':'data/voxceleb2',
    'test_path' :'data/voxceleb1',
    'musan_path':'data/musan_split',
    'rir_path'  :'data/RIRS_NOISES/simulated_rirs',

    ## Model definition
    'n_mels': 40,
    'log_input': False,
    'model': '',
    'encoder_type': '',
    'nOut': 512,

    ## For test only
    'eval': False,

    ## Distributed and mixed precision training
    'port': "8888",
    'distributed': True,
    'mixedprec': False,
}

#custom args
space['model'] = 'ResNetSE34L'
space['log_input'] = True
space['encoder_type'] = 'SAP'
space['trainfunc'] = 'angleproto'
space['save_path'] = 'exps/temp'
space['nPerSpeaker'] = 2
space['batch_size'] = 200
space['train_list'] = '../../datasets/train_list.txt'
space['train_path'] = '/home/rvirgilli/datasets/vox1_dev_wav'
space['test_interval'] = 200
space['test_list'] = '/home/rvirgilli/datasets/veri_test.txt'
space['test_path'] = '/home/rvirgilli/datasets/vox1_test/wav'
space['max_epoch'] = 2

#hyperopt parameters
trials_file = "search-spect.hyperopt"
best_model_folder = './best'
trained_models_folder = './trained_models/'

if not os.path.exists(best_model_folder):
    os.makedirs(best_model_folder)

def search(space):
    #try:

    start_time = time.time()

    print('New trial started:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print('space sample:')
    print(space)

    if os.path.exists(space['save_path']):
        shutil.rmtree(space['save_path'])

    #train model during 100 epochs
    args = Namespace(**space)
    trainSpeakerNet.main(args)

    #evaluate model
    space['eval'] = True
    args = Namespace(**space)
    eer, mindcf, threshold = trainSpeakerNet.main(args)

    result = {
        "loss": eer,
        "mindcf": mindcf,
        "threshold": threshold,
        "status": STATUS_OK,
    }

    print('Model trained - eer:', eer)

    elapsed = time.time() - start_time

    print('Elapsed time:', str(datetime.timedelta(seconds=elapsed)))

    return result

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

def run_trials():
    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(trials_file, "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn=search, space=space, algo=tpe.suggest, max_evals=max_trials, trials=trials)

    print("Best:", best)


    if len(trials) - 1 == trials.best_trial['tid']:
        newest_file = newest(os.path.join(space['save_path'], 'model'))
        shutil.copy(src=newest_file, dst=trained_models_folder)
        print('Best file copied.')
        #bot.send_message(chat_id=chat_id, text='Best: ' + str(trials.best_trial['result']['loss']))


    # save the trials object
    with open(trials_file, "wb") as f:
        pickle.dump(trials, f)


if __name__ == '__main__':
    while True:
        run_trials()
