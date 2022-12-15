# -*- coding: utf-8 -*-
"""
todo: write
"""

from tensorflow.keras.models import Model
import tensorflow as tf

from models.GAN import GAN
from models.binary_classification import discriminator
from models.super_resolution import rcan


class CAGAN(GAN):
    def __init__(self, args):
        GAN.__init__(self, args)
        print('CAGAN')

        self.g_input = Input(self.data.input_dim)
        self.d_input = Input(self.data.output_dim)

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        optimizer_d = self.args.d_opt
        optimizer_g = self.args.g_opt

        # --------------------------------------------------------------------------------
        #                              define combined model
        # --------------------------------------------------------------------------------
        disc = self.discriminator()
        frozen_d = Model(inputs=self.d_input,
                         outputs=disc.outputs)
        frozen_d.trainable = False

        gen = self.generator()
        print(gen)

        fake_hp = gen(self.input)

        judge = frozen_d(fake_hp)
        label = np.zeros(self.args.batch_size)

        # g.compile(loss=self.generator_loss(judge, ),
        #           optimizer=optimizer_g,
        #           loss_weights=[1, weight_wf_loss])
        # if weight_wf_loss > 0:
        #     combined = Model(input_lp, [judge, fake_hp, fake_hp])
        #     loss_wf = create_psf_loss(psf)
        #     combined.compile(loss=['binary_crossentropy', loss_mse_ssim_3d, loss_wf],
        #                      optimizer=optimizer_g,
        #                      loss_weights=[0.1, 1, weight_wf_loss])  # 0.1 1
        # else:
        #     combined = Model(input_lp, [judge, fake_hp])
        #     combined.compile(loss=['binary_crossentropy', loss_mse_ssim_3d],
        #                      optimizer=optimizer_g,
        #                      loss_weights=[0.1, 1])  # 0.1 1

        # lr_controller_g = ReduceLROnPlateau(model=g, factor=lr_decay_factor,
        #                                     patience=10, mode='min', min_delta=1e-4,
        #                                     cooldown=0, min_learning_rate=g_start_lr * 0.1, verbose=1)
        # lr_controller_d = ReduceLROnPlateau(model=d, factor=lr_decay_factor,
        #                                     patience=10, mode='min', min_delta=1e-4,
        #                                     cooldown=0, min_learning_rate=d_start_lr * 0.1, verbose=1)

    def discriminator(self):
        print('input', self.data.input_dim)
        disc = Model(discriminator(self.data.output_dim))
        disc.compile(loss='binary_crossentropy',
                     optimizer=self.args.d_opt,
                     metrics=['accuracy'])

        # frozen_disc = Model(inputs=disc.inputs, outputs=disc.outputs)
        # frozen_disc.trainable = False
        return disc

    def generator(self):

        gen = Model(rcan(self.input_dim))
        return gen

    def generator_loss(self, disc_generated_output, gen_output, target):
        alpha = 100
        beta = 1
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (alpha * l1_loss)

        return total_gen_loss, gan_loss, l1_loss
