from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Concatenate, Lambda, Dot

""" Layer of Spheral Harmonic Units. 
    spherical_ID is the degree (l). Options are l=2,3,4
    Input is the xyz coordinates of the 10 nearest neighbours
    Returns the real spherical harmonics with 2l+1 components 
    
    # Example
    y2 = Layer_Y(spherical_ID=2, input_shape=(30,))(inputTensor)
    
"""

class Layer_Y(Layer):

    def __init__(self, spherical_ID=3, **kwargs):
        self.Y2 = False
        self.Y3 = False
        self.Y4 = False
        if spherical_ID == 2:
            self.Y2 = True
            self.output_dim = 50
        elif spherical_ID == 3:
            self.Y3 = True
            self.output_dim = 70
        elif spherical_ID == 4:
            self.Y4 = True
            self.output_dim = 90

        super(Layer_Y, self).__init__(**kwargs)

    def call(self, inputTensor):
        if self.Y2 == True:
            y2_all = {}

        elif self.Y3 == True:
            y3 = {}
            for i in range(1,8):
                y3[i] = {}
            y3_all = {}

        elif self.Y4 == True:
            y4 = {}
            for i in range(1,10):
                y4[i] = {}
            y4_all = {}

        for i in range(1, 11):
            id_start = (i - 1) * 3
            id_end = id_start + 3

            coord = inputTensor[:, id_start:id_end]

            z = coord[:, 2:3]
            y = coord[:, 1:2]
            x = coord[:, 0:1]

            z_square = K.square(z)
            y_square = K.square(y)
            x_square = K.square(x)

            x2_minus_y2 = K.tf.subtract(x_square, y_square)

            r2 = Dot(axes=-1, normalize=False)([coord, coord])
            r = K.tf.sqrt(r2)

            twos = K.tf.multiply(K.ones_like(x), 2)
            threes = K.tf.multiply(K.ones_like(x), 3)

            if self.Y3:
                r3 = K.tf.multiply(r, r2)
                fours = K.tf.multiply(K.ones_like(x), 4)

            elif self.Y4:
                z4 = K.square(z_square)
                r4 = K.tf.square(r2)
                sevens = Lambda(lambda x: K.tf.multiply(K.ones_like(x), 7))(x)
                thirty = Lambda(lambda x: K.tf.multiply(K.ones_like(x), 30))(x)
                thirty5 = Lambda(lambda x: K.tf.multiply(K.ones_like(x), 35))(x)


            if self.Y2:
                y2 = {}
                y2[1] = K.tf.multiply(x, y)
                y2[2] = K.tf.multiply(y, z)
                y2[3] = K.tf.subtract(K.tf.subtract(K.tf.multiply(z_square, twos), y_square), x_square)
                y2[4] = K.tf.multiply(x, z)
                y2[5] = x2_minus_y2
                y2_all[i] = Concatenate()([y2[1], y2[2], y2[3], y2[4], y2[5]])
                y2_all[i] = K.tf.divide(y2_all[i], r2)

            if self.Y3:

                y3[1][i]  = K.tf.multiply(K.tf.subtract(K.tf.multiply(threes, x_square),y_square), y)
                y3[2][i] =  K.tf.multiply(K.tf.multiply(x, y),z)
                y3[3][i] =  K.tf.multiply(K.tf.subtract(K.tf.multiply(fours, z_square),x2_minus_y2),y)
                y3[4][i] = K.tf.multiply(K.tf.subtract(K.tf.multiply(z_square, twos),K.tf.multiply(x2_minus_y2, threes)), z)
                y3[5][i] = K.tf.multiply(K.tf.subtract(K.tf.multiply(fours, z_square),x2_minus_y2),x)
                y3[6][i] = K.tf.multiply(z, x2_minus_y2)
                y3[7][i] = K.tf.multiply(x, K.tf.subtract(x_square, K.tf.multiply(y_square, threes)))
                y3_all[i] = Concatenate()([y3[1][i], y3[2][i], y3[3][i], y3[4][i], y3[5][i], y3[6][i], y3[7][i]])
                y3_all[i] = K.tf.divide(y3_all[i], r3)

            if self.Y4:
                y4[1][i] = K.tf.multiply(x, K.tf.multiply(y, x2_minus_y2))

                y4[2][i] = K.tf.multiply(z,K.tf.multiply(y,K.tf.subtract(K.tf.multiply(threes, x_square),y_square)))

                y4[3][i] = K.tf.multiply(y,K.tf.multiply(x, K.tf.subtract(K.tf.multiply(sevens, z_square),r2)))

                y4[4][i] = K.tf.multiply(z, K.tf.multiply(y, K.tf.subtract(K.tf.multiply(sevens, z_square),K.tf.multiply(threes, r2))))

                y4[5][i] = K.tf.add(K.tf.subtract(K.tf.multiply(thirty5, z4), K.tf.multiply(r2, K.tf.multiply(thirty, z_square))),K.tf.multiply(threes, r4))

                y4[6][i] = K.tf.multiply(z,K.tf.multiply(x, K.tf.subtract(K.tf.multiply(sevens, z_square),K.tf.multiply(threes, r2))))

                y4[7][i] = K.tf.multiply(x2_minus_y2, K.tf.subtract(K.tf.multiply(sevens, z_square), r2))

                y4[8][i] = K.tf.multiply(z, K.tf.multiply(x, K.tf.subtract(x_square, K.tf.multiply(threes, y_square))))

                y4[9][i] = K.tf.subtract(K.tf.multiply(x_square, K.tf.subtract(x_square, K.tf.multiply(threes, y_square))),
                        K.tf.multiply(y_square, K.tf.subtract(K.tf.multiply(threes, x_square), y_square)))

                y4_all[i] = Concatenate()([y4[1][i], y4[2][i], y4[3][i], y4[4][i], y4[5][i], y4[6][i], y4[7][i], y4[8][i], y4[9][i]])
                y4_all[i] = K.tf.divide(y4_all[i], r4)


        if self.Y2:
            ConcatY2 = Concatenate()([y2_all[1], y2_all[2], y2_all[3], y2_all[4], y2_all[5],
                                  y2_all[6], y2_all[7], y2_all[8], y2_all[9],y2_all[10]])
            return ConcatY2
        elif self.Y3:
            ConcatY3 = Concatenate()([y3_all[1], y3_all[2], y3_all[3], y3_all[4], y3_all[5],
                                  y3_all[6], y3_all[7], y3_all[8], y3_all[9],y3_all[10]])
            return ConcatY3
        elif self.Y4:
            ConcatY4 = Concatenate()([y4_all[1], y4_all[2], y4_all[3], y4_all[4], y4_all[5],
                                  y4_all[6], y4_all[7], y4_all[8], y4_all[9],y4_all[10]])
            return ConcatY4


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
