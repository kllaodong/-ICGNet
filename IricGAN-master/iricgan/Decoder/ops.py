import tensorflow as tf
import numpy as np
from collections import OrderedDict

n_mapping=8
w_dim=512
sn=False
w_ema_decay = 0.995
style_mixing_prob = 0.9
truncation_cutoff = 8
truncation_psi = 0.7

def lerp(a, b, t):
    # t == 1.0: use b
    # t == 0.0: use a
    with tf.name_scope("Lerp"):
        out = a + (b - a) * t
    return out

def resolution_list(img_size) :
    res = 4
    x = []
    while True :
        if res > img_size :
            break
        else :
            x.append(res)
            res = res * 2
    return x

def featuremap_list(img_size) :
    start_feature_map = 512
    feature_map = start_feature_map
    x = []
    fix_num = 0
    while True :
        if img_size < 4 :
            break
        else :
            x.append(feature_map)
            img_size = img_size // 2
            if fix_num > 2 :
                feature_map = feature_map // 2
            fix_num += 1
    return x

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        norm = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        x = x * tf.rsqrt(norm + epsilon)
    return x

def g_mapping( z, n_broadcast, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('g_mapping', reuse=reuse):
        # normalize input first
        x = pixel_norm(z)
        # run through mapping network
        for ii in range(n_mapping):
            with tf.variable_scope('FC_{:d}'.format(ii)):
                x = fully_connected(x, units=w_dim, gain=np.sqrt(2), lrmul=0.01, sn=sn)
                x = apply_bias(x, lrmul=0.01)
                x = lrelu(x, alpha=0.2)
        # broadcast to n_layers
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])
    return x
    
def generator(w_broadcasted, alpha, target_img_size, is_training=True, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("generator", reuse=reuse):
        resolutions = resolution_list(target_img_size)
        featuremaps = featuremap_list(target_img_size)

        w_avg = tf.get_variable('w_avg', shape=[w_dim],
                                dtype=tf.float32, initializer=tf.initializers.zeros(),
                                trainable=False)
        """ mapping layers """
        n_broadcast = len(resolutions) * 2
        #w_broadcasted = g_mapping(z, n_broadcast)
        if is_training:
            """ apply regularization techniques on training """
            # update moving average of w
            w_broadcasted = update_moving_average_of_w(w_broadcasted, w_avg)
            # perform style mixing regularization
            w_broadcasted = style_mixing_regularization(z, w_broadcasted, n_broadcast, resolutions)
        else :
            """ apply truncation trick on evaluation """
            w_broadcasted = truncation_trick(n_broadcast, w_broadcasted, w_avg, truncation_psi)

        """ synthesis layers """
        x_8,x_32,x_256 = g_synthesis(w_broadcasted, alpha, resolutions, featuremaps)
        return x_256

def update_moving_average_of_w( w_broadcasted, w_avg):
    
    with tf.variable_scope('WAvg'):
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
        update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, w_ema_decay))
        with tf.control_dependencies([update_op]):
            w_broadcasted = tf.identity(w_broadcasted)
    return w_broadcasted

def truncation_trick( n_broadcast, w_broadcasted, w_avg, truncation_psi):
    with tf.variable_scope('truncation'):
        layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_indices.shape, dtype=np.float32)
        coefs = tf.where(layer_indices < truncation_cutoff, truncation_psi * ones, ones)
        w_broadcasted = lerp(w_avg, w_broadcasted, coefs)
    return w_broadcasted
    

def style_mixing_regularization( z, w_broadcasted, n_broadcast, resolutions):
    with tf.name_scope('style_mix'):
        z2 = tf.random_normal(tf.shape(z), dtype=tf.float32)
        w_broadcasted2 = g_mapping(z2, n_broadcast)
        layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
        last_layer_index = (len(resolutions)) * 2

        mixing_cutoff = tf.cond(tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
            lambda: tf.random_uniform([], 1, last_layer_index, dtype=tf.int32),
            lambda: tf.constant(last_layer_index, dtype=tf.int32))

        w_broadcasted = tf.where(tf.broadcast_to(layer_indices < mixing_cutoff, tf.shape(w_broadcasted)),
                                    w_broadcasted,
                                    w_broadcasted2)
    return w_broadcasted


def get_style_class(resolutions, featuremaps) :
    coarse_styles = OrderedDict()
    middle_styles = OrderedDict()
    fine_styles = OrderedDict()
    for res, n_f in zip(resolutions, featuremaps) :
        if res >= 4 and res <= 8 :
            coarse_styles[res] = n_f
        elif res >= 16 and res <= 32 :
            middle_styles[res] = n_f
        else :
            fine_styles[res] = n_f
    return coarse_styles, middle_styles, fine_styles



def apply_noise(x):
    with tf.variable_scope('Noise'):
        noise = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1])
        weight = tf.get_variable('weight', shape=[x.get_shape().as_list()[-1]], initializer=tf.initializers.zeros())
        weight = tf.reshape(weight, [1, 1, 1, -1])
        x = x + noise * weight

    return x

def apply_bias(x, lrmul):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros()) * lrmul

    if len(x.shape) == 2:
        x = x + b
    else:
        x = x + tf.reshape(b, [1, 1, 1, -1])

    return x

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def adaptive_instance_norm(x, w):
    x = instance_norm(x)
    x = style_mod(x, w)
    return x

def get_weight(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # equalized learning rate
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    # create variable.
    weight = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32,
                             initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
    return weight


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def fully_connected(x, units, gain=np.sqrt(2), lrmul=1.0, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        weight_shape = [x.get_shape().as_list()[-1], units]
        weight = get_weight(weight_shape, gain, lrmul)

        if sn :
            weight = spectral_norm(weight)

        x = tf.matmul(x, weight)

        return x

def style_mod(x, w):
    with tf.variable_scope('StyleMod'):
        units = x.shape[-1] * 2
        style = fully_connected(w, units=units, gain=1.0, lrmul=1.0)
        style = apply_bias(style, lrmul=1.0)

        style = tf.reshape(style, [-1, 2, 1, 1, x.shape[-1]])
        x = x * (style[:, 0] + 1) + style[:, 1]

    return x

def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('InstanceNorm'):
        x = x - tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)

    return x

def conv(x, channels, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        weight_shape = [kernel, kernel, x.get_shape().as_list()[-1], channels]

        weight = get_weight(weight_shape, gain, lrmul)

        if sn :
            weight = spectral_norm(weight)

        x = tf.nn.conv2d(input=x, filter=weight, strides=[1, stride, stride, 1], padding='SAME')

        return x

def synthesis_const_block(res, w_broadcasted, n_f, sn=False):
    w0 = w_broadcasted[:, 0]
    w1 = w_broadcasted[:, 1]


    batch_size = tf.shape(w0)[0]

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('const_block'):
            x = tf.get_variable('Const', shape=[1, 4, 4, n_f], dtype=tf.float32, initializer=tf.initializers.ones())
            x = tf.tile(x, [batch_size, 1, 1, 1])

            x = apply_noise(x) # B module
            x = apply_bias(x, lrmul=1.0)

            x = lrelu(x, 0.2)
            x = adaptive_instance_norm(x, w0) # A module

        with tf.variable_scope('Conv'):
            x = conv(x, channels=n_f, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=sn)

            x = apply_noise(x) # B module
            x = apply_bias(x, lrmul=1.0)

            x = lrelu(x, 0.2)
            x = adaptive_instance_norm(x, w1) # A module

    return x
def torgb(x, res, sn=False):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('ToRGB'):
            x = conv(x, channels=3, kernel=1, stride=1, gain=1.0, lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
    return x

def g_synthesis( w_broadcasted, alpha, resolutions, featuremaps, reuse=tf.AUTO_REUSE):
    with tf.variable_scope('g_synthesis', reuse=reuse):
        coarse_styles, middle_styles, fine_styles = get_style_class(resolutions, featuremaps)
        layer_index = 2

        """ initial layer """
        res = resolutions[0]
        n_f = featuremaps[0]

        x = synthesis_const_block(res, w_broadcasted, n_f, sn=False)

        """ remaining layers """
        progressive=True
        if progressive :
            images_out = torgb(x, res=res, sn=False)
            coarse_styles.pop(res, None)

            # Coarse style [4 ~ 8]
            # pose, hair, face shape
            for res, n_f in coarse_styles.items():
                x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=False)
                img = torgb(x, res, sn=False)
                images_out = upscale2d(images_out)
                images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)
                layer_index += 2
            images8=images_out
            # Middle style [16 ~ 32]
            # facial features, eye
            for res, n_f in middle_styles.items():
                x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=False)
                img = torgb(x, res, sn=False)
                images_out = upscale2d(images_out)
                images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)     
                layer_index += 2
            images32=images_out
            
            # Fine style [64 ~ 1024]
            # color scheme
            for res, n_f in fine_styles.items():
                x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=False)
                img = torgb(x, res, sn=False)
                images_out = upscale2d(images_out)
                images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                layer_index += 2
                #images256=images_out     
        else :
            for res, n_f in zip(resolutions[1:], featuremaps[1:]) :
                x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=False)

                layer_index += 2
            images_out = torgb(x, resolutions[-1], sn=False)

        
        return images8,images32,images_out


def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor ** 2)
                return dx, lambda ddx: _upscale2d(ddx, factor)

            return y, grad

        return func(x)


def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID')



def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape # [bs, h, w, c]
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[-1]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[-1]])
    return x

def flatten(x) :
    return tf.layers.flatten(x)



def synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=False):
    w0 = w_broadcasted[:, layer_index]
    w1 = w_broadcasted[:, layer_index + 1]

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Conv0_up'):
            x = upscale_conv(x, n_f, kernel=3, gain=np.sqrt(2), lrmul=1.0, sn=sn)
            x = blur2d(x, [1, 2, 1])

            x = apply_noise(x) # B module
            x = apply_bias(x, lrmul=1.0)

            x = lrelu(x, 0.2)
            x = adaptive_instance_norm(x, w0) # A module

        with tf.variable_scope('Conv1'):
            x = conv(x, n_f, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=sn)

            x = apply_noise(x) # B module
            x = apply_bias(x, lrmul=1.0)

            x = lrelu(x, 0.2)
            x = adaptive_instance_norm(x, w1) # A module

    return x


def blur2d(x, f, normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)

            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)

            return y, grad

        return func(x)

def _blur2d(x, f, normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[-1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME')
    x = tf.cast(x, orig_dtype)
    return x


def upscale_conv(x, channels, kernel, gain=np.sqrt(2), lrmul=1.0, sn=False):
    batch_size = tf.shape(x)[0]
    height, width = x.shape[1], x.shape[2]
    fused_scale = (min(height, width) * 2) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = upscale2d(x)
        x = conv(x, channels=channels, kernel=kernel, stride=1, gain=gain, lrmul=lrmul, sn=sn)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    weight_shape = [kernel, kernel, channels, x.get_shape().as_list()[-1]]
    output_shape = [batch_size, height * 2, width * 2, channels]

    weight = get_weight(weight_shape, gain, lrmul)
    weight = tf.pad(weight, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:], weight[:-1, 1:], weight[1:, :-1], weight[:-1, :-1]])

    if sn:
        weight = spectral_norm(weight)

    x = tf.nn.conv2d_transpose(x, filter=weight, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')

    return x


def smooth_transition(prv, cur, res, transition_res, alpha):
    # alpha == 1.0: use only previous resolution output
    # alpha == 0.0: use only current resolution output

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('smooth_transition'):
            # use alpha for current resolution transition
            if transition_res == res:
                out = lerp_clip(cur, prv, alpha)

            # ex) transition_res=32, current_res=16
            # use res=16 block output
            else:   # transition_res > res
                out = lerp_clip(cur, prv, 0.0)
    return out

def lerp_clip(a, b, t):
    # t >= 1.0: use b
    # t <= 0.0: use a
    with tf.name_scope("LerpClip"):
        out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out



def fromrgb(x, res, n_f, sn=False):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('FromRGB'):
            x = conv(x, channels=n_f, kernel=1, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
            x = lrelu(x, 0.2)
    return x


def discriminator_block(x, res, n_f0, n_f1, sn=False):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Conv0'):
            x = conv(x, channels=n_f0, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
            x = lrelu(x, 0.2)

        with tf.variable_scope('Conv1_down'):
            x = blur2d(x, [1, 2, 1])
            x = downscale_conv(x, n_f1, kernel=3, gain=np.sqrt(2), lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
            x = lrelu(x, 0.2)

    return x

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        s = x.shape
        y = tf.reshape(x, [group_size, -1, num_new_features, s[3] // num_new_features, s[1], s[2]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=2)
        y = tf.cast(y, x.dtype)

        y = tf.tile(y, [group_size, s[1], s[2], 1])
        return tf.concat([x, y], axis=-1)

def discriminator_last_block(x, res, n_f0, n_f1, sn=False):

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        x = minibatch_stddev_layer(x, group_size=4, num_new_features=1)

        with tf.variable_scope('Conv0'):
            x = conv(x, channels=n_f0, kernel=3, stride=1, gain=np.sqrt(2), lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
            x = lrelu(x, 0.2)

        with tf.variable_scope('Dense0'):
            x = fully_connected(x, units=n_f1, gain=np.sqrt(2), lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)
            x = lrelu(x, 0.2)

        with tf.variable_scope('Dense1'):
            x = fully_connected(x, units=1, gain=1.0, lrmul=1.0, sn=sn)
            x = apply_bias(x, lrmul=1.0)

    return x

def discriminator( x_init, alpha, target_img_size, reuse=tf.AUTO_REUSE):
    sn=False
    progressive=True

    with tf.variable_scope("discriminator", reuse=reuse):
        resolutions = resolution_list(target_img_size)
        featuremaps = featuremap_list(target_img_size)

        r_resolutions = resolutions[::-1]
        r_featuremaps = featuremaps[::-1]

        """ set inputs """
        x = fromrgb(x_init, r_resolutions[0], r_featuremaps[0], sn)

        """ stack discriminator blocks """
        for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
            res_next = r_resolutions[index + 1]
            n_f_next = r_featuremaps[index + 1]

            x = discriminator_block(x, res, n_f, n_f_next,sn)

            if progressive :
                x_init = downscale2d(x_init)
                y = fromrgb(x_init, res_next, n_f_next, sn)
                x = smooth_transition(y, x, res, r_resolutions[0], alpha)

        """ last block """
        res = r_resolutions[-1]
        n_f = r_featuremaps[-1]

        logit = discriminator_last_block(x, res, n_f, n_f, sn)

        return logit

def compute_loss(real_images, real_logit, fake_logit):
    r1_gamma, r2_gamma = 10.0, 0.0

    # discriminator loss: gradient penalty
    d_loss_gan = tf.nn.softplus(fake_logit) + tf.nn.softplus(-real_logit)
    real_loss = tf.reduce_sum(real_logit)
    real_grads = tf.gradients(real_loss, [real_images])[0]
    r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
    # r1_penalty = tf.reduce_mean(r1_penalty)
    d_loss = d_loss_gan + r1_penalty * (r1_gamma * 0.5)
    d_loss = tf.reduce_mean(d_loss)

    # generator loss: logistic nonsaturating
    g_loss = tf.nn.softplus(-fake_logit)
    g_loss = tf.reduce_mean(g_loss)

    return d_loss, g_loss



def downscale_conv(x, channels, kernel, gain, lrmul, sn=False):
    height, width = x.shape[1], x.shape[2]
    fused_scale = (min(height, width) * 2) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = conv(x, channels=channels, kernel=kernel, stride=1, gain=gain, lrmul=lrmul, sn=sn)
        x = downscale2d(x)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    weight = get_weight([kernel, kernel, x.get_shape().as_list()[-1], channels], gain, lrmul)
    weight = tf.pad(weight, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:], weight[:-1, 1:], weight[1:, :-1], weight[:-1, :-1]]) * 0.25
    if sn:
        weight = spectral_norm(weight)
    x = tf.nn.conv2d(input=x, filter=weight, strides=[1, 2, 2, 1], padding='SAME')
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID')

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1 / factor ** 2)
                return dx, lambda ddx: _downscale2d(ddx, factor)

            return y, grad

        return func(x)


def g_mapping_inverse(input):
    with tf.variable_scope('resnet_inverse',reuse=tf.AUTO_REUSE):
        #normalize
        x=pixel_norm(input)
        w_dim=512
        sn=False
        #mapping network
        for i in range(8):
            with tf.variable_scope('inverse_fC_{:d}'.format(i)):
                x = fully_connected(x, units=w_dim, gain=np.sqrt(2), lrmul=0.01, sn=sn)
                x = apply_bias(x, lrmul=0.01)
                x = lrelu(x, alpha=0.2)
        return x

def compute_feat_loss(feature,feature_hat):

    assert len(feature)==len(feature_hat)
    loss=0.0
    for i in range(len(feature)):
        loss+=tf.reduce_mean(tf.losses.mean_squared_error(feature[i] ,feature_hat[i]))
    return loss
