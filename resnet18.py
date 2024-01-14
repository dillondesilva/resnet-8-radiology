import tensorflow as tf

class ResidualBlock(tf.Module):
    '''
    Consists of block for residual learning
    '''
    def __init__(self, filter_size=(3, 3), n_filters=64, n_layers=2):
        self._initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.filter_size = filter_size
        self.n_layers = n_layers
        
        # Initialize weights for residual block
        initialized_weights = self._initializer(shape=(1, n_filters, 3, 3), dtype=tf.float32)
        self.w1 = tf.Variable(initialized_weights, dynamic_size=True)
        self.w2 = tf.Variable(initialized_weights, dynamic_size=True)

    def __call__(self, x):
        x_skip = x

        # Residual path
        x = tf.nn.conv2d(x, self.w1, strides=1, padding='SAME')
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.w2, strides=1, padding='SAME')
        x = tf.nn.relu(x)

        # Adding skip connection with residual mapping
        x = x + x_skip
        return x

class ResNet18(tf.Module):
    '''
    ResNet18 Model Class Built from Scratch
    '''
    def __init__(self):
        self._initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        self.w_conv = tf.Variable(self._initializer(shape=(1, 64, 3, 3)))
        self.w_fc = tf.Variable(self._initializer(shape=(12288, 4)))
        self.residual_blocks = []  
        n_filters = 64 # Setting our initial number of filters

        for i in range(8):
            if i % 2 == 0 and i != 0:
                n_filters = n_filters * 2
                self.residual_blocks.append(ResidualBlock(n_filters=n_filters))

    def __call__(self, x):
        x = tf.nn.conv2d(x, self.w_conv, strides=1, padding="SAME")
        # Residual blocks training
        for rb in self.residual_blocks:
            print(x.shape)
            x = rb(x)
        x = tf.nn.avg_pool(x, ksize=(1, 2, 2, 1), strides=1, padding="SAME")
        flattened_x = tf.reshape(x, (1, -1))
        logits = tf.linalg.matmul(flattened_x, self.w_fc)
        outputs = tf.nn.softmax(logits)
        return outputs

# Loss function
def compute_loss(model, inputs, targets):
    predictions = model(inputs)
    loss = tf.reduce_mean(tf.square(predictions - targets))
    return loss

# Function to perform a training step
def train_step(model, inputs, targets, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, inputs, targets)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

my_resnet = ResNet18()
input = tf.random.uniform((1, 64, 64, 3))
target = tf.Variable([0, 0, 1, 0], dtype=tf.float32)

optimizer = tf.optimizers.Adam(learning_rate=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    loss = train_step(my_resnet, input, target, optimizer)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")
