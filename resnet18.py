import tensorflow as tf

class ResidualBlock:
    '''
    Consists of block for residual learning
    '''
    def __init__(self, filter_size=(3, 3), n_filters=64, n_layers=2):
        self.weights = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        self._initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.filter_size = filter_size
        self.n_layers = n_layers
        
        # Initialize weights for residual block
        i = 0
        while i < n_layers:
            layer_weights = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            j = 0
            while j < n_filters:
                w_ij = self._initializer(shape=filter_size)
                layer_weights.write(j, w_ij)
                j += 1
            
            self.weights.write(i, layer_weights.stack())
            i += 1

    def forward(self, x):
        x_skip = x
        
        # Residual path
        i = 0
        while i < self.weights.size():
            filters = self.weights.read(i)
            x = tf.nn.conv2d(x, self.weights.stack(), strides=1, padding='SAME')
            x = tf.nn.relu(x)
            i += 1
        
        # Adding skip connection with residual mapping
        x = x + x_skip
        return x
        
    def get_weights(self):
        return self.weights.stack()

class ResNet18:
    '''
    ResNet18 Model Class Built from Scratch
    '''
    def __init__(self):
        self._initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.w_conv = self._initializer(shape=(1, 64, 3, 3))
        self.residual_blocks = []  
        n_filters = 64 # Setting our initial number of filters

        for i in range(8):
            if i % 2 == 0 and i != 0:
                n_filters = n_filters * 2
                self.residual_blocks.append(ResidualBlock(n_filters=n_filters))

    def train(self, x, y):
        x = tf.nn.conv2d(x, self.w_conv, strides=1, padding="SAME")
        # Residual blocks training
        for rb in self.residual_blocks:
            output = rb.forward(x)
            print(output)

my_resnet = ResNet18()
my_resnet.train(tf.random.uniform((128, 64, 64, 3)), tf.random.uniform((128, 64, 64, 3)))