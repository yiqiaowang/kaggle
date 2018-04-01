class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
        
    def forward(self, x):
        return x.view(x.size(0), -1) 

class ConvNet(nn.Module):
    def __init__(self, params):
        super(ConvNet, self).__init__()
        
        # Set convnet parameters
        self.size_conv_layers = params['size_conv_layers']
        self.size_conv_filters = params['size_conv_filters']
        self.pool_size = params['pool_size']
        self.size_hidden_layers = params['size_hidden_layers']
        self.dropout_p = params['dropout_p']
        self._image_size = params['image_size']
 

        # Assume that the input to the first conv layer is of 1 dim
        self._conv_tuples = zip([1] + self.size_conv_layers[:-1],
                                self.size_conv_layers)
                 
           
        # Define convnet structure
        self.conv_layers = [
            nn.Conv2d(*t, f)
            for t,f in zip(
                self._conv_tuples,
                self.size_conv_filters
            )
        ]
       
        self.pool = nn.MaxPool2d(self.pool_size)    
        
        
        self._infer_hidden_0()
        _hidden_in_dim = self.size_conv_layers[-1] * self._image_size**2
        
        
        self._hidden_tuples = zip([_hidden_in_dim] + self.size_hidden_layers[:-1],
                          self.size_hidden_layers)
        
        self.view = View()
        
        self.hidden_layers = [
            nn.Linear(*t)
            for t in self._hidden_tuples
        ]
               
        self.dropout = nn.Dropout2d(p=self.dropout_p)
    
    @staticmethod  
    def conv_out_size(layer, w_old):
        """
        We assume the kernel_size will always be (nxn)
        and the stride will always by (kxk)
        """
        kern_width = layer.kernel_size[0]
        padding_l = layer.padding[0]
        padding_r = layer.padding[1]
        stride = layer.stride[0]
       
        return (w_old - kern_width + padding_l + padding_r)/stride + 1
  
    @staticmethod  
    def pool_out_size(layer, w_old):

        kern_width = layer.kernel_size
        padding = layer.padding
        stride = layer.stride
       
        # Janky as fk
        _tmp = (w_old - kern_width + padding)/stride + 1
        return int(_tmp)
  
    def _infer_hidden_0(self):
        for layer in self.conv_layers:
            self._image_size = ConvNet.conv_out_size(
                layer,
                self._image_size
            )
            self._image_size = ConvNet.pool_out_size(
                self.pool,
                self._image_size
            )
        
  
    def construct(self):
        
        _seq_mod = []
        
        _seq_mod.append(self.conv_layers[0])
        _seq_mod.append(nn.ReLU())
        _seq_mod.append(self.pool)
        
        for conv in self.conv_layers[1:]:
            _seq_mod.append(conv)
            _seq_mod.append(nn.ReLU())
            _seq_mod.append(self.pool)
            
        _seq_mod.append(self.view)

        if len(self.hidden_layers) > 1:
          _seq_mod.append(self.hidden_layers[0])
          _seq_mod.append(nn.ReLU())
          _seq_mod.append(self.dropout)
        
        for hidden in self.hidden_layers[1:-1]:
          _seq_mod.append(hidden)
          _seq_mod.append(nn.ReLU())
          _seq_mod.append(self.dropout)

        # No need to add a relu to the output layer
        _seq_mod.append(self.hidden_layers[-1])
        
        # Dont need this layer since CrossEntropyLoss already 
        # contains a logsoftmax layer
        #_seq_mod.append(nn.Softmax(dim=1))
        
        #for model in _seq_mod:
        #    print(model)
               
        return _seq_mod
