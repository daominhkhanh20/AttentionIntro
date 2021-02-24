import torch 
from torch.autograd import Variable 
from torch import optim 
from load_data import load_generate_date
from sklearn.model_selection import train_test_split
import numpy as np 
from matplotlib import pyplot as plt 
from model_attention import Encoder, Decoder
from torch import nn 
import os 
import numpy as np 

hidden_size=128
learning_rate=1e-4
decoder_learning_rato=0.1
X,Y,X_word_index,Y_word_index=load_generate_date().load_data()
X_index_word=dict(zip(X_word_index.values(),X_word_index.keys()))
Y_index_word=dict(zip(Y_word_index.values(),Y_word_index.keys()))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.15)
Y_val=Y_val.astype(np.float32)

os.chdir('Data')
with open('data_test.npy','wb') as f:
    np.save(f,X_test)
    np.save(f,Y_test)
os.chdir('..')

input_size=len(X_word_index)+1 # add padding character
output_size=len(Y_word_index)+2 # plus 2 for start character, end character
sos_index=len(Y_word_index)# start character
eos_index=sos_index+1# end character


def forward_and_compute_loss(inputs,targets,encoder,decoder,criterion):
    batch=inputs.size()[1]
    sos=Variable(torch.ones((1,batch),dtype=torch.long)*sos_index)
    eos=Variable(torch.ones((1,batch),dtype=torch.long)*eos_index)
    decoder_inputs=torch.cat((sos,targets),dim=0)
    decoder_targets=torch.cat((targets,eos),dim=0)

    encoder_output,hidden=encoder(inputs)
    outputs,attention_weights,hidden=decoder(decoder_inputs,encoder_output,hidden)
    
    outputs=torch.transpose(torch.transpose(outputs,0,1),1,2)
    decoder_targets=torch.transpose(decoder_targets,0,1)

    loss=criterion(outputs,decoder_targets)

    return loss,outputs

def evaluate(inputs,targets,encoder,decoder,criterion):
    encoder.eval() #turn on mode evaluate
    decoder.eval()
    eval_loss,outputs=forward_and_compute_loss(inputs,targets,encoder,decoder,criterion)
    #output: T*B*output_size
    outputs=outputs.transpose(1,2)
    """
    tensor.squeeze(dim=?): remove all superfical 1 dimention from tensor
        ex:  a.size()=[3,2,1]   ===> a.shape()=[3,2] after a=a.squeeze(-1)
    """
    preds_indexs=torch.argmax(outputs,dim=-1).squeeze(-1) #T*output_size
    return eval_loss.item(),preds_indexs.data.numpy()

def train(x_train,y_train,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    train_loss,outputs=forward_and_compute_loss(x_train,y_train,encoder,decoder,criterion)
    train_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return train_loss.item()    

def plot_loss(train_loss,val_loss,n_epochs):
    plt.plot(np.arange(n_epochs),train_loss,label='Train loss')
    plt.plot(np.arange(n_epochs),val_loss,label='Val loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

def save_model(encoder,decoder):
    #save model    
    folder="Model"
    if os.path.isdir(folder) is False:
        path=os.path.join(os.getcwd(),folder)
        os.mkdir(path)
    torch.save(encoder.state_dict(),"Model/encoder.dict")
    torch.save(decoder.state_dict(),'Model/decoder.dict')

def start_train(encoder,decoder):
    encoder_optimizer=optim.Adam(encoder.parameters(),lr=learning_rate)
    decoder_optimizer=optim.Adam(decoder.parameters(),lr=learning_rate*decoder_learning_rato)
    train_loss=[]
    val_loss=[]
    batch_size=64
    epochs=40

    x_val=torch.tensor(X_val,dtype=torch.long).transpose(0,1)
    y_val=torch.tensor(Y_val,dtype=torch.long).transpose(0,1)
    criterion=nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for idx in range(len(X_train)//batch_size):
            x_train_batch=torch.tensor(
                X_train[batch_size*idx:min(batch_size*(idx+1),len(X_train))],
                dtype=torch.long
            ).transpose(0,1)

            y_train_batch=torch.tensor(
                Y_train[batch_size*idx:min(batch_size*(idx+1),len(X_train))],
                dtype=torch.long
            ).transpose(0,1)

            loss=train(x_train_batch,y_train_batch,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
        loss1,_=evaluate(x_val,y_val,encoder,decoder,criterion)
        print("Epoch:{} --- Loss:{:5f} --- Val_loss:{:5f}".format(epoch,loss,loss1))
        train_loss.append(loss)
        val_loss.append(loss1)

    plot_loss(train_loss,val_loss,epochs)
    save_model(encoder,decoder)



def main():
    encoder=Encoder(input_size,hidden_size)
    decoder=Decoder(output_size,hidden_size,0.1)
    start_train(encoder,decoder)


if __name__=="__main__":
    main()

# encoder=Encoder(input_size,hidden_size)
# decoder=Decoder(output_size,hidden_size,0.1)

# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_rato)

# input_encoder=torch.randint(1,input_size,(34,6),dtype=torch.long)
# encoder_outputs,hiden_encoder=encoder(input_encoder)

# input_decoder=torch.randint(1,output_size,(10,6),dtype=torch.long)
# output,attention_weights=decoder(input_decoder,encoder_outputs,hiden_encoder)
# print(output.shape)



