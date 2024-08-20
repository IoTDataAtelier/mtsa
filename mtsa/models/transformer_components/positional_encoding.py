import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #Create a matyrix of shape (seq_len, d_model)
            #Passado tamanho da série como linhas e dimensão do modelo (512) como colunas
        pe = torch.zeros(seq_len, d_model)
        
        #Create a vector of shape (seq_len, 1)
            #Cria um vetor com números de 0 a seq_len - 1
            #Especifica que o tensor gerado deve ter o tipo de dados float
            #Adiciona uma dimensão para transformá-lo em um vetor coluna 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        #Create a divisor term
            #Cria um tensor com os índices pares
            #Espaço logarítmico para estabilidade numérica
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        #Aplicar função de seno para posições pares
        #Aplicar função de cosseno para posições ímpares
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        
        #self.pe como um buffer no __init__, garante-se que as codificações posicionais são armazenadas e reutilizadas de maneira eficiente, sem recalculá-las a cada passo do forward. 
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #Define que as codificações posicionais não têm gradientes, ou seja, elas não serão atualizadas durante o treinamento (as codificações são fixas e não aprendidas).
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        
        
        
        