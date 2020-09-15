import tensorflow as tf

class ResidualBlock(tf.keras.Model):
     def __init__ (self, filter_in, filter_out, kernel_size):
          super(ResidualBlock, self).__init__()
          # batch normalization -> ReLU -> conv layer
          # 여기서 Relu 같은 경우는 변수가 없는 layer이므로 여기서 굳이 initialize 할 필요는 없다. )

          self.bn1 = tf.keras.layers.BatchNormalization()
          self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding = 'same')
          self.bn2 = tf.keras.layers.BatchNormalization()
          self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding = 'same')

          if filter_in == filter_out:
               self.identity = lambda x:x
          else:
               self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding = 'same')

     def call(self, x, training = False):
          h = self.bn1(x, training = training)
          h = tf.nn.relu(h)
          h = self.conv1(h)

          h = self.bn2(h, training = training)
          h = tf.nn.relu(h)
          h =self.conv2(h)
          return self.identity(x) + h


class ResnetLayer(tf.keras.Model):

     def __init__(self, filter_in, filters, kernel_size):
          super(ResnetLayer, self).__init__()
          self.sequence = list()
          for f_in, f_out in zip([filter_in]+list(filters), filters):
               self.sequence.append(ResidualBlock(f_in, f_out, kernel_size))

     def call(self, x, training = False):
          for unit in self.sequence:
               x = unit(x, training = training)
          return x

class ResNet(tf.keras.Model):
     def __init__(self):
          super(ResNet, self).__init__()
          self.conv1 = tf.keras.layers.Conv2D(8,(3,3), padding = 'same', activation  = 'relu')

          self.res1 = ResnetLayer(8, (16,16), (3,3))
          self.pool1 = tf.keras.layers.MaxPool2d((2,2))

          self.res2 = ResnetLayer(8, (32,32), (3,3))
          self.pool2 = tf.keras.layers.MaxPool2d((2,2))

          self.res3 = ResnetLayer(8, (64,64), (3,3))

          self.flatten = tf.keras.layers.Flatten()
          self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
          self.dense2 = tf.keras.layers.Dense(10, activation = "softmax")

     def call(self, x, training = False):
          x = self.conv1(x)
          x = self.res1(x, training = training)
          x = self.pool1(x)
          x = self.res2(x, training = training)
          x = self.pool2(x)
          x = self.res3(x, training = training)

          x = self.flatten(x)
          x = self.dense1(x)
          return self.dense2(x)




