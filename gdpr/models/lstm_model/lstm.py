import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.max_length = max_length

    def forward(self, input_ids):

        token_embeddings = self.embedding(input_ids)

        position_ids = torch.arange(self.max_length).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = token_embeddings + position_embeddings

        return embeddings
class LSTMCell(nn.Module):

    def __init__(self, input_lenght, hidden_lenght):
        super(LSTMCell, self).__init__()
        self.input_lenght = input_lenght
        self.hidden_lenght = hidden_lenght

        # Forget Gate
        self.forget_w1 = nn.Linear(self.input_lenght, self.hidden_lenght, bias=True)
        self.forget_r1 = nn.Linear(self.hidden_lenght, self.hidden_lenght, bias=False)
        self.forget_sigmoid = nn.Sigmoid()

        # Input Gate
        self.input_w2 = nn.Linear(self.input_lenght, self.hidden_lenght, bias=True)
        self.input_r2 = nn.Linear(self.hidden_lenght, self.hidden_lenght, bias=False)
        self.input_sigmoid = nn.Sigmoid()

        # Cell Memory
        self.memory_w3 = nn.Linear(self.input_lenght, self.hidden_lenght, bias=True)
        self.memory_r3 = nn.Linear(self.hidden_lenght, self.hidden_lenght, bias=False)
        self.memory_activation = nn.Tanh()

        # Output Gate
        self.output_w4 = nn.Linear(self.input_lenght, self.hidden_lenght, bias=True)
        self.output_r4 = nn.Linear(self.hidden_lenght, self.hidden_lenght, bias=False)
        self.output_sigmoid = nn.Sigmoid()
        self.final_activation = nn.Tanh()

    def forget_gate(self, x, h):
        return self.forget_sigmoid(self.forget_w1(x)+self.forget_r1(h))

    def input_gate(self, x, h):
        return self.input_sigmoid(self.input_w2(x)+self.input_r2(h))

    def memory_gate(self, i, f, x, h, c_prev):
        g = self.memory_activation(self.memory_w3(x)+self.memory_r3(h)) * i
        c = f * c_prev
        return g + c


    def output_gate(self, x, h):
        return self.output_sigmoid(self.output_w4(x) + self.output_r4(h))

    def forward(self, x, input):
        (h, c_prev) = input
        # Input
        i = self.input_gate(x, h)
        # Forget
        f = self.forget_gate(x, h)
        # Update memory
        c_next = self.memory_gate(i, f, x, h, c_prev)
        # Output
        o = self.output_gate(x, h)
        h_next = o * self.final_activation(c_next)

        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, vocab_size = 100, seq_size = 128, input_shape=768, hidden_shape=128, num_class=9, cell_number=3):
        super().__init__()
        self.embedding = Embeddings(vocab_size=vocab_size,
                                              embedding_dim=input_shape,
                                              max_length=seq_size)
        
        self.rnn1 = LSTMCell(input_shape, hidden_shape)
        self.rnn2 = LSTMCell(hidden_shape, hidden_shape)
        self.linear = nn.Linear(hidden_shape, num_class)
        self.hidden_shape = hidden_shape

    def forward(self, input):
        outputs = []

        embeds = self.embedding(input)

        h_t = torch.zeros(input.size(0), self.hidden_shape)
        c_t = torch.zeros(input.size(0), self.hidden_shape)
        h_t2 = torch.zeros(input.size(0), self.hidden_shape)
        c_t2 = torch.zeros(input.size(0), self.hidden_shape)


        for i in range(embeds.size(1)):

            # if self.LSTM:
            h_t, c_t = self.rnn1(embeds[:, i, :], (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            # else:
            #     h_t = self.rnn1(input_t, h_t)
            #     h_t2 = self.rnn2(h_t, h_t2)


            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs