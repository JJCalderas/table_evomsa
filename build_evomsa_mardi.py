import pandas as pd
from glob import glob
from microtc.utils import load_model
from EvoMSA.base import EvoMSA
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
from joblib import Parallel, delayed
import joblib as joblib
from timeit import default_timer as timer

import warnings
warnings.filterwarnings("ignore")

import winsound
winsound.Beep(3000, 900)


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

debug_file = 'debug_101.txt'

def process(fn):

    train_df = pd.read_json(fn, lines=True)
    test_df = pd.read_json(fn.replace("_train", "_test"), lines=True)

    # recortarlos
    #train_df = train_df.loc[0:89, ['text', 'klass']]
    #test_df = test_df.loc[0:16, ['text', 'klass']]

    X_train, y_train = train_df['text'], train_df['klass']
    X_test,  y_test  = test_df['text'],  test_df['klass']
       
    ds_n = fn.replace("../dataset_es\\",'').replace("..\\dataset_es/",'').replace('_Es_train.json','')

    with open(debug_file, 'a', encoding='utf-8') as the_file:
        the_file.write(ds_n + ' > ' + str(train_df.shape) + ':' + str(test_df.shape) + '\n')
        
    print(ds_n, train_df.shape, test_df.shape, '\n')

    X_F, y_F = train_df['text'], train_df['klass']  # X, y  Folds
   
    for i, sample in df_samples.iterrows():   ### [:3] [8:9]
        try:
            
            # remueve el pre-model (de la combinacion) si coincide con el dataset en evaluacion
            sample_avoid = sample.dropna().values[sample.dropna().values != ds_n]

            # toma los objetos de los pre-modelos
            pre_models = [pre_models_dict[key] for key in sample_avoid]  
           
            ### train & test ###
            #evo = EvoMSA(TR=True, B4MSA=True, lang='es', Emo=True, HA=True, stacked_method=stacked_method, models = pre_models)                
            evo = EvoMSA(TR=False, B4MSA=True, lang='es', Emo=False, HA=False, TH=False, stacked_method=stacked_method, models = pre_models)     
            evo.fit(X_train, y_train)                
            pred = evo.predict(X_test)
            recall_score = metrics.recall_score(y_test, pred, average="macro")
            f1_score = metrics.f1_score(y_test, pred, average="macro")

            winsound.Beep(4500, 300)
           
            ### K-Fold / train ###
            scores_kfold = []
            skf = StratifiedKFold(n_splits=5)
            for train_index, test_index in skf.split(X_F, y_F):
                X_train_F, X_test_F = X_F[train_index], X_F[test_index]
                y_train_F, y_test_F = y_F[train_index], y_F[test_index]    

                #evo = EvoMSA(TR=True, B4MSA=True, lang='es', Emo=True, HA=True, stacked_method=stacked_method, models = pre_models)        
                evo = EvoMSA(TR=False, B4MSA=True, lang='es', Emo=False, HA=False, TH=False, stacked_method=stacked_method, models = pre_models)  
                evo.fit(X_train_F, y_train_F)                
                pred = evo.predict(X_test_F)
                recall_score = metrics.recall_score(y_test_F, pred, average="macro")
                f1_score_F = metrics.f1_score(y_test_F, pred, average="macro")

                scores_kfold.append(f1_score_F)

                winsound.Beep(4500, 300)
               
               
            #scores = ', '.join([str(s) for s in scores_kfold])                
            _ = [ds_n, i, train_df.shape, test_df.shape, f1_score, recall_score, 
                 sample_avoid.tolist(),
                 i, train_df.shape, np.mean(scores_kfold, axis=0), np.std(scores_kfold, axis=0), np.min(scores_kfold), np.max(scores_kfold), scores_kfold]
                     
            # print(_)
            with open(debug_file, 'a', encoding='utf-8') as the_file:
                the_file.write(ds_n + ', ' + str(i) + ', ' + str(f1_score) + ', ' + str(np.mean(scores_kfold, axis=0)) + '\n')
           
            results.append(_)

        except Exception as ex:
            winsound.Beep(500, 100)
            winsound.Beep(500, 100)
            winsound.Beep(500, 100)
            winsound.Beep(500, 100)
            print('>> ', ex, '\n')
            with open(debug_file, 'a', encoding='utf-8') as the_file:
                the_file.write('\n')
                the_file.write('=======================================================\n')
                the_file.write('Exception: ' + ds_n + ' > ' + str(i) + ' > ' + '\n')
                the_file.write(ex + '\n')
        finally:
            winsound.Beep(3000, 900)

    return results

fnames = glob("../dataset_es/*_train.json") ## todos los train datasets
fnames.sort()
results = []
parallel_pool = Parallel(n_jobs=7)
delayed_funcs = [delayed(process)(fn) for fn in fnames]

start = timer()
results = parallel_pool(delayed_funcs)


winsound.Beep(500, 900)

clean_results = []
for r in results:
    for a in r:
        clean_results.append(a)

results_df = pd.DataFrame(data=clean_results, columns=['dataset', 'combina_1', 'train.shape', 'test.shape', 'f1', 'recall', 
                                                       'combinations', 'combina_2', 'train.shape', 'mean', 'std', 'min', 'max', 'scores_kfold' ])
results_df.to_csv('results_101.csv')
#results_df

end = timer()
print(end - start, '\n')
print('--- done ---\n')


# haha21, 2, 0.8339409575611665, 0.8361930292269637    35min