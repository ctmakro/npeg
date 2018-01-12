import numpy as np
import canton as ct
from canton import *
import tensorflow as tf

# get the fking data
def cifar():
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255
    return X_train

def encoder():
    c=Can()
    def conv(nip,nop,tail=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if tail:
            # c.add(BatchNorm(nop))
            c.add(Act('elu'))
    c.add(Lambda(lambda x:x-0.5))
    conv(3,16)
    conv(16,32)
    conv(32,64)
    conv(64,128,tail=False)
    c.chain()
    return c

def decoder():
    c=Can()
    def conv(nip,nop,tail=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if tail:
            # c.add(BatchNorm(nop))
            c.add(Act('elu'))

    conv(128,64)
    conv(64,32)
    conv(32,16)
    conv(16,3,tail=False)
    c.add(Act('sigmoid'))
    c.chain()
    return c

def get_trainer():
    x = ph([None,None,3])

    # augment the training set by adding random gain and bias pertubation
    sx = tf.shape(x)
    input_gain = tf.random_uniform(
        minval=0.6,
        maxval=1.4,
        shape=[sx[0],1,1,1])
    input_bias = tf.random_uniform(
        minval=-.2,
        maxval=.2,
        shape=[sx[0],1,1,1])
    noisy_x = x * input_gain + input_bias
    noisy_x = tf.clip_by_value(noisy_x,clip_value_max=1.,clip_value_min=0.)

    code_noise = tf.Variable(0.1)
    linear_code = enc(noisy_x)

    # add gaussian before sigmoid to encourage binary code
    noisy_code = linear_code + \
        tf.random_normal(stddev=code_noise,shape=tf.shape(linear_code))
    binary_code = Act('sigmoid')(noisy_code)

    y = dec(binary_code)

    def image_similarity(a,b):
        return tf.reduce_mean((a-b)**2)

    reconstruction_loss = image_similarity(y, noisy_x)

    # encourage sparsity of produced code
    sparsity_penalty = tf.reduce_mean(binary_code**2) * 0.1

    loss =  reconstruction_loss + sparsity_penalty

    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss,
        var_list=enc.get_weights()+dec.get_weights())

    def feed(batch,cnoise):
        sess = ct.get_session()
        res = sess.run([train_step,loss],feed_dict={
            x:batch,
            code_noise:cnoise,
        })
        return res[1]

    set_training_state(False)
    quantization_threshold = tf.Variable(0.5)
    binary_code_test = tf.cast(binary_code>quantization_threshold,tf.float32)
    y_test = dec(binary_code_test)

    def test(batch,quanth):
        sess = ct.get_session()
        res = sess.run([binary_code_test,y_test,binary_code,y,noisy_x],feed_dict={
            x:batch,
            quantization_threshold:quanth,
        })
        return res
    return feed,test

def r(ep=1,cnoise=0.1):
    np.random.shuffle(xt)
    length = len(xt)
    bs = 20
    for i in range(ep):
        print('ep',i)
        for j in range(0,length,bs):
            minibatch = xt[j:j+bs]
            loss = feed(minibatch,cnoise)
            print(j,'loss:',loss)

            if j%1000==0:
                show()

def show(threshold=.5):
    from cv2tools import vis,filt
    bs = 16
    j = np.random.choice(len(xt)-16)
    minibatch = xt[j:j+bs]
    code, rec, code2, rec2, noisy_x = test(minibatch,threshold)

    code = np.transpose(code[0:1],axes=(3,1,2,0))
    code2 = np.transpose(code2[0:1],axes=(3,1,2,0))

    vis.show_batch_autoscaled(code, name='code(quant)', limit=600.)
    vis.show_batch_autoscaled(code2, name='code2(no quant)', limit=600.)

    vis.show_batch_autoscaled(noisy_x,name='input')
    vis.show_batch_autoscaled(rec,name='recon(quant)')
    vis.show_batch_autoscaled(rec2,name='recon(no quant)')

def save():
    enc.save_weights('enc.npy')
    dec.save_weights('dec.npy')

def load():
    enc.load_weights('enc.npy')
    dec.load_weights('dec.npy')

enc,dec = encoder(),decoder()
enc.summary()
dec.summary()
xt = cifar()

if __name__ == '__main__':
    feed,test = get_trainer()
    get_session().run(ct.gvi())
