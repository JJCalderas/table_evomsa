import pandas as pd
from glob import glob
from microtc.utils import load_model
from EvoMSA.base import EvoMSA
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from joblib import Parallel, delayed
import joblib as joblib
import timeit

import warnings
warnings.filterwarnings('ignore')
    
from EvoMSA import __version__ as version
print(version)    
    
pd.set_option('display.max_colwidth', None)
pd.options.display.max_rows = 1200

stacked_method = "sklearn.svm.LinearSVC"

# 9 Models Pre-Trained EvoMSA
pre_models_dict = {
    "davincis22": [load_model('pre_models/davincis22_Es.evomsa'), stacked_method],
    "detoxis21_aggressiveness" : [load_model('pre_models/detoxis21_aggressiveness_Es.evomsa'), stacked_method],
    "exist21" : [load_model('pre_models/exist21_Es.evomsa'), stacked_method],
    "haha21" : [load_model('pre_models/haha21_Es.evomsa'), stacked_method],
    "meoffendes21" : [load_model('pre_models/meoffendes21_Es.evomsa'), stacked_method],
    "mexa3t18_aggress" : [load_model('pre_models/mexa3t18_aggress_Es.evomsa'), stacked_method],
    "misogyny_centrogeo" : [load_model('pre_models/misogyny_centrogeo_Es.evomsa'), stacked_method],
    "metwo22" : [load_model('pre_models/metwo22_Es.evomsa'), stacked_method]
}  

print(len(pre_models_dict.keys()), pre_models_dict.keys(), '\n')

df_samples = pd.read_csv('premodels_combinations_89_first.csv')
df_samples.rename(columns = {df_samples.columns[0]:'Idx'}, inplace = True)
df_samples.drop('Idx', inplace=True, axis=1)
#df_samples.fillna("", inplace=True)
#df_samples

debug_file = 'debug_301.txt'

def process(experiment):
            
    starttime = timeit.default_timer()
    
    ds_n = experiment[0]
    train_df = experiment[1]
    test_df = experiment[2]
    i = experiment[3]
    sample = experiment[4]

    # print('Start >> ', i, ds_n)

    # recortarlos
    #train_df = train_df.loc[0:49, ['text', 'klass']]
    #test_df = test_df.loc[0:12, ['text', 'klass']]

    X_train, y_train = train_df['text'], train_df['klass']
    X_test,  y_test  = test_df['text'],  test_df['klass']

    X_F, y_F = train_df['text'], train_df['klass']  # X, y  Folds
       
    print(i, ds_n, train_df.shape, test_df.shape, '\n')
    
    sample_avoid = sample.dropna().values[sample.dropna().values != ds_n]
    
    if (i>0) & (sample_avoid.size < 1):
        _ = [ds_n, i, train_df.shape, test_df.shape, 0.0, 0.0, sample_avoid.tolist(), i, 0.0, 0.0, 0.0, 0.0, [], 0.0]
        return _
   
    # toma los objetos de los pre-modelos
    pre_models = [pre_models_dict[key] for key in sample_avoid] 
 
    try:
        evo = EvoMSA(TR=False, B4MSA=True, lang='es', Emo=False, HA=False, TH=False, stacked_method=stacked_method, models = pre_models)     
        evo.fit(X_train, y_train)                
        pred = evo.predict(X_test)
        recall_score = metrics.recall_score(y_test, pred, average="macro")
        f1_score = metrics.f1_score(y_test, pred, average="macro")

        scores_kfold = []
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X_F, y_F):
            X_train_F, X_test_F = X_F[train_index], X_F[test_index]
            y_train_F, y_test_F = y_F[train_index], y_F[test_index]    

            evo = EvoMSA(TR=False, B4MSA=True, lang='es', Emo=False, HA=False, TH=False, stacked_method=stacked_method, models = pre_models)  
            evo.fit(X_train_F, y_train_F)                
            pred = evo.predict(X_test_F)
            f1_score_F = metrics.f1_score(y_test_F, pred, average="macro")

            scores_kfold.append(f1_score_F)

        endtime = timeit.default_timer() - starttime     
            
        # print(i, ds_n, train_df.shape, test_df.shape, i, f1_score, recall_score, ' << \n')

        _ = [ds_n, i, train_df.shape, test_df.shape, f1_score, recall_score, 
             sample_avoid.tolist(),
             i, np.mean(scores_kfold, axis=0), np.std(scores_kfold, axis=0), np.min(scores_kfold), np.max(scores_kfold), scores_kfold,
            endtime]

        with open(debug_file, 'a', encoding='utf-8') as the_file:
            the_file.write(ds_n + ', ' + str(i) + ', ' + str(f1_score) + ', ' + str(np.mean(scores_kfold, axis=0)) + ' \n')
    
    except Exception as ex:
        with open(debug_file, 'a', encoding='utf-8') as the_file:
            the_file.write(ds_n + ', ' + i + ', ' + str(ex) + ' \n')
        
    return _


fnames = glob("../dataset_es/*_train.json") ## todos los train datasets
fnames.sort()
experiments = []
for fn in fnames:
    
    train_df = pd.read_json(fn, lines=True)
    test_df = pd.read_json(fn.replace("_train", "_test"), lines=True)
   
    ds_name = fn.replace("../dataset_es\\",'').replace("../dataset_es/",'').replace("..\\dataset_es\\",'').replace('_Es_train.json','')
    
    for i, sample in df_samples.iterrows():
        experiments.append([ds_name, train_df, test_df, i, sample])
        

parallel_pool = Parallel(n_jobs=19)
delayed_funcs = [delayed(process)(experiment) for experiment in experiments]
results = parallel_pool(delayed_funcs)
    
results_df = pd.DataFrame(data=results, columns=['dataset', 'combina_1', 'train.shape', 'test.shape', 'f1', 'recall','combinations',
                                                 'combina_2', 'media', 'desviacion', 'minimimo', 'maximo', 'folds', 'elapsed'])
results_df.to_csv('results_301.csv')    
                       

