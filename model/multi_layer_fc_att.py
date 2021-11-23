import torch as t


class FullyConnectedWithSelfAttention(t.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = t.nn.Dropout(dropout)
        if num_layers == 1:
            self.fc =  t.nn.Linear(in_size, out_size)
            self.att = t.nn.Linear(in_size, out_size)
        elif num_layers == 2:
            self.fc_in = t.nn.Linear(in_size, hidden_size)
            self.fc_out = t.nn.Linear(hidden_size, out_size)
            self.att_in = t.nn.Linear(in_size, hidden_size)
            self.att_out = t.nn.Linear(hidden_size, out_size)
        else:
            self.fc_in = t.nn.Linear(in_size, hidden_size)
            self.fc_hidden = t.nn.ModuleList([t.nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
            self.fc_out = t.nn.Linear(hidden_size, out_size)
            self.att_in = t.nn.Linear(in_size, hidden_size)
            self.att_hidden = t.nn.ModuleList([t.nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
            self.att_out = t.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        if self.num_layers == 1:
            att_out = t.sigmoid(self.dropout(self.att(x)))
            fc_out = att_out * self.dropout(self.fc(x))
        elif self.num_layers == 2:
            att_in = t.relu(self.dropout(self.att_in(x)))
            att_out = t.sigmoid(self.dropout(self.att_out(att_in)))
            fc_in = t.relu(self.dropout(self.fc_in(x)))
            fc_out = att_out * self.dropout(self.fc_out(fc_in))
        else:
            att_hiddenen = t.relu(self.dropout(self.att_in(x)))
            fc_hidden = t.relu(self.dropout(self.fc_in(x)))
            for i in self.num_layers:
                att_hiddenen = att_hiddenen + t.relu(self.dropout(self.att_hidden[i](att_hiddenen)))
                fc_hidden = fc_hidden + t.relu(self.dropout(self.fc_hidden[i](fc_hidden)))
            att_out = t.sigmoid(self.dropout(self.att_out(att_hiddenen)))
            fc_out = att_out * self.dropout(self.fc_out(fc_hidden))
        return fc_out