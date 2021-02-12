import pickle, time, datetime, shutil, os
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import trainSpeakerNet
import numpy as np

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
    'encoder_type': 'SAP',
    'nOut': 512,

    ## For test only
    'eval': False,

    ## Distributed and mixed precision training
    'port': "8888",
    'distributed': False,
    'mixedprec': False,
}

#custom args
space['model'] = 'ResNetSE34L'
space['log_input'] = True
space['encoder_type'] = 'SAP'
space['nClasses'] = 1251
space['trainfunc'] = 'amsoftmax'
space['save_path'] = 'exps/temp'
space['scale'] = 30
space['margin'] = 0.3
space['batch_size'] = 200
space['train_list'] = 'D:/datasets/vox1_dev/train_list_vox1.txt'
space['train_path'] = 'E:/datasets/vox1_dev/wav'
space['test_interval'] = 1
space['test_list'] = 'E:/datasets/vox1_test/veri_test.txt'
space['test_path'] = 'E:/datasets/vox1_test/wav'
space['max_epoch'] = 1
space['nPerSpeaker'] = 1

#hyperopt parameters
trials_file = "search-spect.hyperopt"
best_model_folder = './best'
trained_models_folder = './trained_models/'

if not os.path.exists(best_model_folder):
    os.makedirs(best_model_folder)

def get_best_model(save_path):

    score_lines = open(os.path.join(save_path, 'result', 'scores.txt'), mode='r').readlines()

    array_it = []
    array_eer = []
    array_tacc = []
    array_tloss = []

    for i in range(0, len(score_lines), 2):
        array_it.append(int(score_lines[i].split(',')[0].strip().split(' ')[1]))
        array_eer.append(float(score_lines[i].split(',')[1].strip().split(' ')[1]))
        array_tacc.append(float(score_lines[i + 1].split(',')[1].strip().split(' ')[1]))
        array_tloss.append(float(score_lines[i + 1].split(',')[2].strip().split(' ')[1]))

    arg_min_eer = np.array(array_eer).argmin()

    it = array_it[arg_min_eer]

    model_path = os.path.join(save_path, 'model', 'model%09d.model'%it)

    return array_eer[arg_min_eer], array_tacc[arg_min_eer], array_tloss[arg_min_eer], model_path

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

    eer, tacc, tloss, model_path = get_best_model(space['save_path'])

    result = {
        'loss': eer,
        'eer': eer,
        'tacc': tacc,
        'tloss': tloss,
        'model_path': model_path,
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
        best_model_path = trials.trials[-1]['result']['model_path']
        best_bkp = './best/best_model.model'
        shutil.copy(src=best_model_path, dst=best_bkp)
        print('Best file copied.')
        #bot.send_message(chat_id=chat_id, text='Best: ' + str(trials.best_trial['result']['loss']))


    # save the trials object
    with open(trials_file, "wb") as f:
        pickle.dump(trials, f)


if __name__ == '__main__':
    while True:
        run_trials()
