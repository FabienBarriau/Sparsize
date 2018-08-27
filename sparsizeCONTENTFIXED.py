import vgg

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sys import stderr
import time

RELU_LAYERS = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1',
                   'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2','relu5_3','relu5_4')

def sparsizeCONTENTFIXED(network, layer, img, regularisation_coeff, iterations, learning_rate, beta1,
            beta2, epsilon, pooling, print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """

    shape = (1,) + img.shape
    img_conteneur = np.zeros(shape, dtype='float32')
    img_conteneur[0, :, :, :] = img
    vgg_weights, vgg_mean_pixel = vgg.load_net(network)
    loss_curve = []
    sparsness_curve = []

    # make sparse encoded image using backpropogation
    with tf.Graph().as_default():

        init_image = tf.constant(img_conteneur)
        net_init = vgg.net_preloaded(vgg_weights, init_image, pooling)
        init_encoding = net_init[layer]
        init_sparsness = 1. - tf.to_float(tf.count_nonzero(init_encoding))/ tf.to_float(tf.size(init_encoding))

        lbda = tf.constant(regularisation_coeff)
        image = tf.Variable(tf.random_normal(shape) * 0.256)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        encoding = net[layer]
        reg_term = tf.reduce_sum(encoding)

        loss = tf.nn.l2_loss(net_init["relu5_2"] - net["relu5_2"]) + lbda*reg_term
        sparsness = 1. - tf.to_float(tf.count_nonzero(encoding))/ tf.to_float(tf.size(encoding))

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write(' loss: %g\n' % loss.eval())
            stderr.write(' initial sparsness: %g\n' % init_sparsness.eval())
            stderr.write(' final sparsness: %g\n' % sparsness.eval())
        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            iteration_times = []
            start = time.time()
            for i in range(iterations):
                iteration_start = time.time()
                if i > 0:
                    elapsed = time.time() - start
                    # take average of last couple steps to get time per iteration
                    remaining = np.mean(iteration_times[-10:]) * (iterations - i)
                    stderr.write('Iteration %4d/%4d (%s elapsed, %s remaining)\n' % (
                        i + 1,
                        iterations,
                        hms(elapsed),
                        hms(remaining)
                    ))
                    loss_curve.append(sess.run(loss))
                    sparsness_curve.append(sess.run(sparsness))
                else:
                    stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = best.reshape(shape[1:])

                    yield (
                        (None if last_step else i),
                        img_out
                    )

                iteration_end = time.time()
                iteration_times.append(iteration_end - iteration_start)

    fig1 = plt.figure('loss')
    plt.xlabel('Iterations')
    plt.ylabel('Fonction de perte')
    plt.title('Fonction de perte au fil des itérations')
    plt.step(np.arange(1, iterations), loss_curve, 'r')

    fig2 = plt.figure('sparsness')
    plt.xlabel('Iterations')
    plt.title("Parcimonie de l'encodage de la nouvelle image")
    plt.ylabel("Pourcentage de zéro dans l'encodage de la nouvelle image")
    plt.step(np.arange(1, iterations), sparsness_curve, 'b')

    fig1.savefig('generate_content_fixed/' + layer + '_' + str(regularisation_coeff) + '_loss_plot.png')
    fig2.savefig('generate_content_fixed/' + layer + '_' + str(regularisation_coeff) + '_sparsness_plot.png')

    plt.show()

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds
