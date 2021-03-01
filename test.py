import torch 
import pickle 
import numpy as np 
from model_attention import Encoder,Decoder
from torch.autograd import Variable
from random import randint
from matplotlib import pyplot as plt 
from random import randint
import os 
import argparse


ap=argparse.ArgumentParser()
ap.add_argument('-s','--string',required=False,help="add string for test")
args=ap.parse_args()


with open('Data/tokenize_input.pickle','rb') as handle:
    tokenize_input=pickle.load(handle)

with open('Data/tokenize_output.pickle','rb') as handle:
    tokenize_output=pickle.load(handle)

with open('Data/data_test.npy','rb') as f:
    x_test=np.load(f)
    y_test=np.load(f)
    
x_index_to_word=dict(zip(tokenize_input.word_index.values(),tokenize_input.word_index.keys()))
y_index_to_word=dict(zip(tokenize_output.word_index.values(),tokenize_output.word_index.keys()))
x_test=torch.tensor(x_test,dtype=torch.long).transpose(0,1)
y_test=torch.tensor(y_test,dtype=torch.long).transpose(0,1)

input_size=len(tokenize_input.word_index)+1
output_size=len(tokenize_output.word_index)+2
hidden_size=128
sos_index=len(tokenize_output.word_index)
eos_index=sos_index+1


def load_model():
    encoder=Encoder(input_size,hidden_size)
    decoder=Decoder(output_size,hidden_size,0.1)
    encoder.load_state_dict(torch.load('./Model/encoder.dict'))
    #encoder.eval()
    decoder.load_state_dict(torch.load('./Model/decoder.dict'))
    #decoder.eval()
    return encoder,decoder

def decoder_setence(indxs,vocab):
    text=''.join([vocab[w] for w in indxs if (w>0) and w in vocab])
    return text

def preprocessing_text(text):
    sequence=tokenize_input.texts_to_sequences(text)
    sequence=[i[0] for i in sequence if len(i)!=0]
    #print(sequence)
    max_len=len(tokenize_input.word_index)
    if len(sequence)<max_len:
        sequence=[0]*(max_len-len(sequence))+sequence
    return sequence

def predict(x,encoder,decoder,max_len=10):
    #x=x.unsqueeze(-1)#S*1
    batch=x.size()[1]
    decoder_input=Variable(torch.ones((1,batch),dtype=torch.long)*sos_index)
    encoder_output,hidden=encoder(x)
    preds=[]
    attention_weights=[]

    for _ in range(max_len):
        outputs,attention_weight,hidden=decoder(decoder_input,encoder_output,hidden)
        outputs=outputs.squeeze(dim=0)# B*output_size
        pred_index=torch.argmax(outputs,dim=-1)#B
        attention_weights.append(attention_weight.detach())
        decoder_input=Variable(torch.ones((1,batch),dtype=torch.long)*pred_index)
        preds.append(decoder_input)
    
    attention_weights=torch.cat(attention_weights,dim=0).transpose(0,1)
    preds=torch.cat(preds,dim=0).transpose(0,1) #B*T
    return preds,attention_weights

def show_attention_visualization(attentions,text_ori,text_preds):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    cax=ax.matshow(attentions.numpy(),cmap='bone')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(text_ori)))
    ax.set_xticklabels(list(text_ori))
    ax.set_yticks(np.arange(len(text_preds)))
    ax.set_yticklabels(list(text_preds))
    ax.grid()
    ax.set_xlabel("Input setence")
    ax.set_ylabel("Predict setence")
    ax.set_title("Input:{}\nPredict:{}".format(text_ori,text_preds),pad=20)
    plt.show()

    # save result
    path=os.path.join(os.getcwd(),'Result')
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)

    fig.savefig('result{}.png'.format(randint(1,100000)))


def main():
    encoder,decoder=load_model()
    # s=str("t5 ngày 29, thg 2 1972")
    # print(len(s))
    # data_point=preprocessing_text(s)   
    if args.string is None :
        preds,attention_weights=predict(x_test,encoder,decoder)
        x_rand_index=randint(0,len(x_test))
        text_ori=decoder_setence(x_test[:,x_rand_index].numpy(),x_index_to_word)
        text_preds=decoder_setence(preds[x_rand_index].numpy(),y_index_to_word)
        print("Input:",text_ori)
        print("Output:",text_preds)
        show_attention_visualization(attention_weights[x_rand_index,:,-len(text_ori):],text_ori,text_preds)
    else:
        text_ori=str(args.string)
        x=torch.tensor(preprocessing_text(text_ori)).unsqueeze(1)
        preds,attention_weight=predict(x,encoder,decoder)
        print(text_ori)
        text_preds=decoder_setence(preds[0,:].numpy(),y_index_to_word)
        print(text_preds)
        show_attention_visualization(attention_weight[0,:,-len(text_ori):],text_ori,text_preds)



if __name__=="__main__":
    main()
