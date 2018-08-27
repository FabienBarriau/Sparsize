# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import os

import numpy as np
import scipy.misc

from sparsizeHALF import sparsizeHALF

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
ITERATIONS = 1000
REGCOEFF = 20.
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--img',
            dest='img', help='img',
            metavar='IMG', required=True)
    parser.add_argument('--output',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--regularisation_coeff', type=float,
            dest='regularisation_coeff', help='regularisation_coeff',
            metavar='REGCOEFF', default=REGCOEFF)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            metavar='PRINT_ITERATIONS')
    parser.add_argument('--checkpoint-output',dest='checkpoint_output',
            help='checkpoint output format, e.g. output%%s.jpg',
            metavar='OUTPUT')
    parser.add_argument('--checkpoint-iterations', type=int,
            dest='checkpoint_iterations', help='checkpoint frequency',
            metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--width', type=int,
            dest='width', help='output width',
            metavar='WIDTH')
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--pooling',
            dest='pooling', help='pooling layer configuration: max or avg (default %(default)s)',
            metavar='POOLING', default=POOLING)
    parser.add_argument('--overwrite', action='store_true',
            dest='overwrite', help='write file even if there is already a file with that name')
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    image = imread(options.img)

    if options.checkpoint_output and "%s" not in options.checkpoint_output:
        parser.error("To save intermediate images, the checkpoint output "
                     "parameter must contain `%s` (e.g. `foo%s.jpg`)")

    # try saving a dummy image to the output path to make sure that it's writable
    if os.path.isfile(options.output) and not options.overwrite:
        raise IOError("%s already exists, will not replace it without the '--overwrite' flag" % options.output)
    try:
        imsave(options.output, np.zeros((500, 500, 3)))
    except:
        raise IOError('%s is not writable or does not have a valid file extension for an image file' % options.output)

    for iteration, image in sparsizeHALF(
        network=options.network,
        img=image,
        regularisation_coeff=options.regularisation_coeff,
        iterations=options.iterations,
        learning_rate=options.learning_rate,
        beta1=options.beta1,
        beta2=options.beta2,
        epsilon=options.epsilon,
        pooling=options.pooling,
        print_iterations=options.print_iterations,
        checkpoint_iterations=options.checkpoint_iterations
    ):
        output_file = None
        combined_rgb = image
        if iteration is not None:
            if options.checkpoint_output:
                output_file = options.checkpoint_output % iteration
        else:
            output_file = options.output
        if output_file:
            imsave(output_file, combined_rgb)


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
