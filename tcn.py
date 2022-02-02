class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.Tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.batch1 = nn.BatchNorm1d(n_outputs)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.Tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.batch2 = nn.BatchNorm1d(n_outputs)
        self.max_pool = nn.MaxPool1d(2)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.Tanh1, self.dropout1, self.batch1,
                                 self.conv2, self.chomp2, self.Tanh2, self.dropout2, self.batch2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.Tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.Tanh(out + res)
        #return out

class TemporalConvNet(nn.Module):
  def __init__(self, num_features, num_layers, num_filters, kernel_sizes, dropout):
    '''
      -) num_layers: quanti layer conv metto
      -) num_filters: numero di filtri per ciascuna conv
      -) kernel_sizes: array o lista delle dimensioni diverse per ciascun segnale
    '''
    super(TemporalConvNet, self).__init__()
    layers = []
    self.features = nn.ModuleList()
    self.num_features = num_features
    self.num_layers = num_layers
    self.num_filters = num_filters

    for signal in range(self.num_features):
      l = []
      stride = 1
      for i in range(0, self.num_layers):
        dilation_size = 2**i
        in_channel = 1 if i == 0 else self.num_filters
        out_channel = self.num_filters
                                                 
        l += [TemporalBlock(in_channel, out_channel, kernel_sizes[signal], stride=stride, dilation=dilation_size,
                                         padding=(kernel_sizes[signal]-1)*dilation_size, dropout=dropout)]
        stride = 1

      layers.append(l)
    
    for il, layer in enumerate(layers):
      self.features.append(nn.Sequential(*layers[il]))
  
  def forward(self, x):
    dim = x.size(2)
    inputs = []

    for i in range(dim):
      inputs.append(
          x[:,i,:].view(-1,1,dim)
      )
      inputs[i] = self.features[i](inputs[i])
    return inputs


class TCN(nn.Module):
  def __init__(self, num_features,num_layers, num_filters, kernel_sizes, dropout, output_size=2):
      super(TCN, self).__init__()
      self.num_features = num_features
      self.num_layers = num_layers
      self.num_filters = num_filters
      self.kernel_sizes = kernel_sizes
      self.dropout = dropout
      self.output_size = output_size
      
      self.tcn = TemporalConvNet(num_features,num_layers, num_filters, kernel_sizes, dropout=dropout)
      self.fc = nn.Sequential(
          #nn.ReLU(),
          #nn.Dropout(dropout),
          nn.Linear(num_filters*num_features, output_size)
          #nn.Linear(num_filters*num_features, num_features),
          #nn.BatchNorm1d(num_features),
          #nn.Dropout(dropout),
          #nn.Linear(num_features, output_size)
      )


  def forward(self, x):
    #print(x.shape)
    batch_ = x.size(0)
    
    x = self.tcn(x)
    processed_features = []

    for i in range(len(x)):
      processed_features.append(x[i][:,:,-1].view(batch_, -1))
    
    x = torch.stack(processed_features).view(batch_,self.num_filters*self.num_features) #reduce(lambda x,y: torch.cat((x,y)), processed_features)
    #print(x.shape)
    #x = self.fc(x.view(-1,self.num_filters*self.num_features))#[:,0])
    #print(x.shape)
    x = self.fc(x)#[:,0])
    
    #print(x.shape)
    return x
