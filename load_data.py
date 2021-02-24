import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import os 
import pickle
import torch 
class load_generate_date():
    def __init__(self):
        self.file_name='Data/data.csv'
    
    def load_data(self):
        df=pd.read_csv(os.path.join(os.getcwd(),self.file_name),header=None)
        df=df.iloc[np.random.permutation(len(df))]
        X=df[0].values
        Y=df[1].values
        X_tokenize=Tokenizer(char_level=True,filters='')
        Y_tokenize=Tokenizer(char_level=True,filters='')
        X_tokenize.fit_on_texts(X)
        Y_tokenize.fit_on_texts(Y)
        X=X_tokenize.texts_to_sequences(X)
        Y=Y_tokenize.texts_to_sequences(Y)
        
        with open('Data/tokenize_input.pickle','wb') as handle:
            pickle.dump(X_tokenize,handle,protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('Data/tokenize_output.pickle','wb') as handle:
            pickle.dump(Y_tokenize,handle,protocol=pickle.HIGHEST_PROTOCOL)
        
        # because each element in X hasn't the same number of dimensions
        # so we need to add pad_sequences to satisfy the same dimension 
        X=pad_sequences(X,maxlen=len(X_tokenize.word_index))
        Y=np.asarray(Y)

        return X,Y,X_tokenize.word_index,Y_tokenize.word_index

X,Y,X_wids,Y_wids=load_generate_date().load_data()
print(Y_wids)