import torch 
from torch import nn
import numpy as np 
from torch.nn import functional  

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Encoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)

    def forward(self,inputs):
        """
        inputs: S*B
            S : lengh 1 vector unit
            B : batch size (number of the data points pass throughout model)

        """
        embedded=self.embedding(inputs) #S*B*H
        """
        torch.nn.GRU:
            return output, hidden
                output: tensor containing the output features h_t from the last layer GRU
                hiden : tensor containig the h_t with t= len(sequences)
        """
        output_encoder,hiden=self.gru(embedded)
        return output_encoder,hiden # S*B*H,1*B*H
    
class Attention(nn.Module):
    def __init__(self,hidden_size):
        super(Attention,self).__init__()
        self.hidden_size=hidden_size
    def forward(self,encoder_output,decoder_hidden):
        """
        encoder_output:S*B*H
        decoder_hiden: T*B*H
        """

        # we need transpose for multiple matrix
        encoder_output=encoder_output.transpose(0,1) #B*S*H
        decoder_hidden=torch.transpose(torch.transpose(decoder_hidden,0,1),1,2) #B*H*T
        # luong mechanism attention model 
        energies=torch.bmm(encoder_output,decoder_hidden)
        energies=energies.transpose(1,2)# B*T*S

        attention_weights=functional.softmax(energies,dim=-1) #B*T*S

        context_vector=torch.bmm(attention_weights,encoder_output) #B*T*H

        context_vector=context_vector.transpose(0,1)# T*B*H
        attention_weights=attention_weights.transpose(0,1) # T*B*S
        return context_vector,attention_weights

class Decoder(nn.Module):
    def __init__(self,output_size,hidden_size,dropout=0.1):
        super(Decoder,self).__init__()
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.dropout=dropout

        self.attention=Attention(hidden_size)
        self.embedding=nn.Embedding(output_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size)
        self.dropout=nn.Dropout(dropout)

        #concatenate hidden decoder and contex vector
        self.concat=nn.Linear(2*self.hidden_size,self.hidden_size)
        self.out=nn.Linear(self.hidden_size,self.output_size)

    def forward(self,inputs,encoder_output,hidden_encoder):
        """
        inputs: T*B
        encoder_ouputs: S*B*H
        hiden_encoder: 1*B*H
        """
        embedded=self.embedding(inputs)
        embedded=self.dropout(embedded)
        rnn_output,hidden=self.gru(embedded,hidden_encoder)

        context,attention_weights=self.attention(encoder_output,rnn_output)
        concat_input=torch.cat((rnn_output,context),-1) #T*B*2H
        concat_output=torch.tanh(self.concat(concat_input))

        output=self.out(concat_output)

        return output,attention_weights,hidden#T*B*output_size,B*T*S





