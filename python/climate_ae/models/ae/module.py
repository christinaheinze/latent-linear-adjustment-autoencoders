import tensorflow as tf


class _ResidualInnerBlock(tf.keras.Model):
    def __init__(self, num_filters_resnet_conv1, num_filters_resnet_conv2, 
        kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_filters_resnet_conv1, kernel_size=[kernel_size, kernel_size], 
            strides=[1, 1], name="res3x3", padding="SAME", activation=None)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_filters_resnet_conv2, kernel_size=[1, 1], 
            strides=[1, 1], name="res1x1", padding="SAME", activation=None)

    def call(self, inputs, *args, **kwargs):
        h_i = tf.nn.relu(inputs)
        h_i = self.conv1(h_i)
        h_i = tf.nn.relu(h_i)
        h_i = self.conv2(h_i)
        return h_i


class ResidualBlock(tf.keras.Model):
    def __init__(self, num_residual_layers, num_filters_resnet_conv1,
        num_filters_resnet_conv2, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = []
        for _ in range(0, num_residual_layers):
            self.blocks.append(_ResidualInnerBlock(
                num_filters_resnet_conv1=num_filters_resnet_conv1, 
                num_filters_resnet_conv2=num_filters_resnet_conv2,
                kernel_size=kernel_size))

    def call(self, inputs, *args, **kwargs):
        h = inputs
        for block in self.blocks:
            h = tf.keras.layers.add([h, block(h)])
        return h


class ConvEncoderLayer(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.residual_block = ResidualBlock(
            num_residual_layers=config.num_residual_layers,
            num_filters_resnet_conv1=config.num_filters_resnet_conv1,
            num_filters_resnet_conv2=config.num_filters_resnet_conv2,
            kernel_size=self.config.kernel_size)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                self.bn_resnet_block = tf.keras.layers.BatchNormalization(
                    momentum=config.bn_momentum, renorm=config.bn_renorm)

        self.blocks = []
        self.bn_blocks = []
        for i in range(0, config.num_conv_layers):
            self.blocks.append(tf.keras.layers.Conv2D(
                filters=config.filter_sizes[i], 
                kernel_size=[self.config.kernel_size, self.config.kernel_size],
                strides=[2, 2], padding="SAME", 
                activation=tf.nn.relu if self.config.activation == "relu" 
                    else tf.nn.leaky_relu))
            if 'batch_norm' in self.config._fields:
                if self.config.batch_norm:
                    self.bn_blocks.append(tf.keras.layers.BatchNormalization(
                    momentum=config.bn_momentum, renorm=config.bn_renorm))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                for block, bn_block in zip(self.blocks, self.bn_blocks):
                    x = block(x)
                    x = bn_block(x, *args, **kwargs)
                x = self.residual_block(x)
                x = self.bn_resnet_block(x, *args, **kwargs)
            else:
                for block in self.blocks:
                    x = block(x)
                x = self.residual_block(x)
        else:
            for block in self.blocks:
                x = block(x)
            x = self.residual_block(x)
        return x


class FcEncoderLayer(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.blocks = tf.keras.Sequential()
        if self.config.use_dropout:
            self.do_blocks = tf.keras.Sequential()
        for i in range(0, config.num_fc_layers):
            self.blocks.add(tf.keras.layers.Dense(
                units=config.num_hiddens[i], 
                activation=None if self.config.architecture == "linear" 
                        else (tf.nn.relu if self.config.activation=="relu" 
                            else tf.nn.leaky_relu)))
            if self.config.use_dropout:
                self.do_blocks.add(
                    tf.keras.layers.Dropout(self.config.dropout_rate))

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.config.use_dropout:
            raise NotImplementedError
        else:
            x = self.blocks(x)
        return x


class ConvDecoderLayer(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.residual_block = ResidualBlock(
            num_residual_layers=config.num_residual_layers,
            num_filters_resnet_conv1=config.num_filters_resnet_conv1,
            num_filters_resnet_conv2=config.num_filters_resnet_conv2,
            kernel_size=self.config.kernel_size)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                self.bn_resnet_block = tf.keras.layers.BatchNormalization(
                    momentum=config.bn_momentum, renorm=config.bn_renorm)

        self.blocks = []
        self.bn_blocks = []
        for i in range(0, config.num_conv_layers-1): 
            self.blocks.append(
                tf.keras.layers.Convolution2DTranspose(
                    filters=config.filter_sizes[config.num_conv_layers-2-i], 
                    kernel_size=[self.config.kernel_size, self.config.kernel_size], 
                    strides=[2, 2], padding="SAME", 
                    activation=tf.nn.relu if self.config.activation == "relu" 
                        else tf.nn.leaky_relu))
            if 'batch_norm' in self.config._fields:
                if self.config.batch_norm:
                    self.bn_blocks.append(tf.keras.layers.BatchNormalization(
                    momentum=config.bn_momentum, renorm=config.bn_renorm))

        self.output_layer = \
            tf.keras.layers.Convolution2DTranspose(
                filters=config.num_filters_out, 
                kernel_size=[self.config.kernel_size, self.config.kernel_size], 
                strides=[2, 2], padding="SAME",
                activation=tf.nn.sigmoid if config.sigmoid_activation else None)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.residual_block(x)

        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                x = self.bn_resnet_block(x, *args, **kwargs)
                for block, bn_block in zip(self.blocks, self.bn_blocks):
                    x = block(x)
                    x = bn_block(x, *args, **kwargs)
            else:
                for block in self.blocks:
                    x = block(x)
        else:
            for block in self.blocks:
                    x = block(x)

        x = self.output_layer(x)
        return x


class FcDecoderLayer(tf.keras.Model):
    def __init__(self, config, units_out, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.units_out = units_out

    def build(self, input_shape):
        self.blocks = tf.keras.Sequential()
        if self.config.use_dropout:
            self.do_blocks = tf.keras.Sequential()
        for i in range(0, self.config.num_fc_layers-1): 
            self.blocks.add(
                tf.keras.layers.Dense(
                    units=self.config.num_hiddens[self.config.num_fc_layers-2-i], 
                    activation=None if self.config.architecture == "linear" 
                        else (tf.nn.relu if self.config.activation=="relu" 
                            else tf.nn.leaky_relu)))
            if self.config.use_dropout:
                self.do_blocks.add(
                    tf.keras.layers.Dropout(self.config.dropout_rate))
        self.last_block = tf.keras.layers.Dense(units=self.units_out, 
            activation=tf.nn.sigmoid if self.config.sigmoid_activation else None)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = inputs
        if self.config.use_dropout:
            raise NotImplementedError
        else:
            x = self.blocks(x)
        x = self.last_block(x)
        return x


class StochasticLayer(tf.keras.Model):
    def __init__(self, config, dim_latent, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.flatten = tf.keras.layers.Flatten()
        self.dim_latent = dim_latent
        self._input_shape = None

    def build(self, input_shape):
        units = 1
        for i in range(1, len(input_shape)):
            units *= input_shape[i]
        self._input_shape = input_shape
        self.dense_in = tf.keras.layers.Dense(units=self.dim_latent*2, 
            activation=None)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                self.bn1 = tf.keras.layers.BatchNormalization(
                    momentum=self.config.bn_momentum, renorm=self.config.bn_renorm)
        if self.config.use_dropout:
            self.drop1 = tf.keras.layers.Dropout(self.config.dropout_rate)
        self.dense_out = tf.keras.layers.Dense(units=units, activation=None)
        if self.config.use_dropout:
            self.drop2 = tf.keras.layers.Dropout(self.config.dropout_rate)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                self.bn2 = tf.keras.layers.BatchNormalization(
                    momentum=self.config.bn_momentum, renorm=self.config.bn_renorm)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x_shape = tf.shape(inputs)
        x = self.flatten(inputs)
        mu_log_sigma2 = self.dense_in(x)
        if self.config.use_dropout:
            mu_log_sigma2 = self.drop1(mu_log_sigma2, *args, **kwargs)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                mu_log_sigma2 = self.bn1(mu_log_sigma2, *args, **kwargs)
        mu, log_sigma2 = tf.split(mu_log_sigma2, num_or_size_splits=2, 
            axis=-1)
        eps = tf.random.normal(tf.shape(mu), 0, 1)
        z_std = tf.exp(log_sigma2 / 2)
        z = eps*z_std + mu 
        inter_z = self.dense_out(z)
        if self.config.use_dropout:
            inter_z = self.drop2(inter_z, *args, **kwargs)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                inter_z = self.bn2(inter_z, *args, **kwargs)
        output = tf.reshape(inter_z, x_shape)
        kld_z = 0.5 * tf.reduce_sum(
            tf.square(mu) + tf.exp(log_sigma2) - log_sigma2 - 1, axis=1)
        out_dict = {"z": z, "log_sigma_2": log_sigma2, "mu": mu, 
            "kld_z": kld_z}
        return output, out_dict

    def set_encodings(self, z, *args, **kwargs):
        inter_z = self.dense_out(z)
        if self.config.use_dropout:
            inter_z = self.drop2(inter_z, *args, **kwargs)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                inter_z = self.bn2(inter_z, *args, **kwargs)
        output_shape = [-1]
        for i in range(1, len(self._input_shape)):
            output_shape.append(self._input_shape[i])
        output = tf.reshape(inter_z, output_shape)
        return output

    def generate(self, num_samples, *args, **kwargs):
        z = tf.random_normal([num_samples, self.dim_latent], 0, 1)
        inter_z = self.dense_out(z)
        if self.config.use_dropout:
            inter_z = self.drop2(inter_z, *args, **kwargs)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                inter_z = self.bn2(inter_z, *args, **kwargs)
        output_shape = [-1]
        for i in range(1, len(self._input_shape)):
            output_shape.append(self._input_shape[i])
        output = tf.reshape(inter_z, output_shape)
        return output, {"z": z}

    def mean_encode(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        mu_log_sigma2 = self.dense_in(x)
        if self.config.use_dropout:
            mu_log_sigma2 = self.drop1(mu_log_sigma2, *args, **kwargs)
        if 'batch_norm' in self.config._fields:
            if self.config.batch_norm:
                mu_log_sigma2 = self.bn1(mu_log_sigma2, *args, **kwargs)
        mu, _ = tf.split(mu_log_sigma2, num_or_size_splits=2, axis=-1)
        return {"z": mu}


class DeterministicLayer(tf.keras.Model):
    def __init__(self, config, dim_latent, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.dim_latent = dim_latent
        self._input_shape = None
        self.config = config

    def build(self, input_shape):
        units = 1
        for i in range(1, len(input_shape)):
            units *= input_shape[i]
        self._input_shape = input_shape
        self.dense_in = tf.keras.layers.Dense(units=self.dim_latent, 
            activation=None)
        if self.config.use_dropout:
            self.drop1 = tf.keras.layers.Dropout(self.config.dropout_rate)
        self.dense_out = tf.keras.layers.Dense(units=units, activation=None)
        if self.config.use_dropout:
            self.drop2 = tf.keras.layers.Dropout(self.config.dropout_rate)
        self.built = True

    def call(self, inputs, generate=False, *args, **kwargs):
        x_shape = tf.shape(inputs)
        x = self.flatten(inputs)
        z = self.dense_in(x)
        if self.config.use_dropout:
            z = self.drop1(z, *args, **kwargs)
        inter_z = self.dense_out(z)
        if self.config.use_dropout:
            inter_z = self.drop2(inter_z, *args, **kwargs)
        output = tf.reshape(inter_z, x_shape)
        return output, {"z": z}

    def set_encodings(self, z, *args, **kwargs):
        inter_z = self.dense_out(z)
        if self.config.use_dropout:
            inter_z = self.drop2(inter_z, *args, **kwargs)
        output_shape = [-1]
        for i in range(1, len(self._input_shape)):
            output_shape.append(self._input_shape[i])
        output = tf.reshape(inter_z, output_shape)
        return output


class FcDetAutoencoderModel(tf.keras.Model):
    def __init__(self, config, dim_latent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dim_latent = dim_latent
        self.ae_type = "deterministic"
        self._input_shape = None
        self.flatten = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self._input_shape = input_shape
        units = input_shape[1]*input_shape[2]*input_shape[3]
        self.encoder = FcEncoderLayer(config=self.config)
        self.decoder = FcDecoderLayer(config=self.config, units_out=units)
        self.deterministic_layer = DeterministicLayer(config=self.config, 
            dim_latent=self.dim_latent)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        x = self.encoder(inputs=x, *args, **kwargs)
        x, output_dict = self.deterministic_layer(inputs=x, *args, **kwargs)
        x = self.decoder(inputs=x, *args, **kwargs)
        output_shape = [-1, self._input_shape[1], self._input_shape[2],
            self._input_shape[3]]
        x = tf.reshape(x, output_shape)
        output_dict.update({"output": x, "inputs": inputs})
        return output_dict

    def encode(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        x = self.encoder(inputs=x, *args, **kwargs)
        _, output_dict = self.deterministic_layer(inputs=x, *args, **kwargs)
        return output_dict

    def mean_encode(self, inputs, *args, **kwargs):
        return self.encode(inputs, *args, **kwargs)

    def decode(self, x, *args, **kwargs):
        x = self.deterministic_layer.set_encodings(x, *args, **kwargs)
        x = self.decoder(inputs=x, *args, **kwargs)
        output_shape = [-1, self._input_shape[1], self._input_shape[2],
            self._input_shape[3]]
        x = tf.reshape(x, output_shape)
        return {"output": x}

    def autoencode(self, inputs, *args, **kwargs):
        return self.call(inputs, *args, **kwargs)


class FcVarAutoencoderModel(tf.keras.Model):
    def __init__(self, config, dim_latent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dim_latent = dim_latent
        self.ae_type = "variational"
        self._input_shape = None
        self.flatten = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self._input_shape = input_shape
        units = input_shape[1]*input_shape[2]*input_shape[3]
        self.encoder = FcEncoderLayer(config=self.config)
        self.decoder = FcDecoderLayer(config=self.config, units_out=units)
        self.stochastic_layer = StochasticLayer(config=self.config, 
            dim_latent=self.dim_latent)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        x = self.encoder(inputs=x, *args, **kwargs)
        x, output_dict = self.stochastic_layer(inputs=x, *args, **kwargs)
        x = self.decoder(inputs=x, *args, **kwargs)
        output_shape = [-1, self._input_shape[1], self._input_shape[2],
            self._input_shape[3]]
        x = tf.reshape(x, output_shape)
        output_dict.update({"output": x, "inputs": inputs})
        return output_dict

    def encode(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        x = self.encoder(inputs=x, *args, **kwargs)
        _, output_dict = self.stochastic_layer(inputs=x, *args, **kwargs)
        return output_dict

    def mean_encode(self, inputs, *args, **kwargs):
        x = self.flatten(inputs)
        x = self.encoder(inputs=x, *args, **kwargs)
        output_dict = self.stochastic_layer.mean_encode(inputs=x, *args, **kwargs)
        return output_dict

    def decode(self, x, *args, **kwargs):
        x = self.stochastic_layer.set_encodings(x, *args, **kwargs)
        x = self.decoder(inputs=x, *args, **kwargs)
        output_shape = [-1, self._input_shape[1], self._input_shape[2],
            self._input_shape[3]]
        x = tf.reshape(x, output_shape)
        return {"output": x}

    def generate(self, num_samples, *args, **kwargs):
        x, output_dict = self.stochastic_layer.generate(num_samples, *args, **kwargs)
        x = self.decoder(inputs=x, *args, **kwargs)
        output_dict.update({"output": x})
        return output_dict

    def autoencode(self, inputs, *args, **kwargs):
        output_dict = self.mean_encode(inputs, *args, **kwargs)
        return self.decode(output_dict["z"], *args, **kwargs)


class ConvVarAutoencoderModel(tf.keras.Model):
    def __init__(self, config, dim_latent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.dim_latent = dim_latent
        self.ae_type = "variational"
        self.encoder = ConvEncoderLayer(config=config)
        self.decoder = ConvDecoderLayer(config=config)
        self.stochastic_layer = StochasticLayer(config=config, 
            dim_latent=self.dim_latent)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.encoder(inputs=x)
        x, output_dict = self.stochastic_layer(inputs=x, *args, **kwargs)
        x = self.decoder(inputs=x)
        output_dict.update({"output": x, "inputs": inputs})
        return output_dict

    def encode(self, inputs, *args, **kwargs):
        x = inputs
        x = self.encoder(inputs=x)
        _, output_dict = self.stochastic_layer(inputs=x, *args, **kwargs)
        return output_dict

    def mean_encode(self, inputs, *args, **kwargs):
        x = inputs
        x = self.encoder(inputs=x)
        output_dict = self.stochastic_layer.mean_encode(inputs=x, *args, **kwargs)
        return output_dict

    def decode(self, x, *args, **kwargs):
        x = self.stochastic_layer.set_encodings(x, *args, **kwargs)
        x = self.decoder(inputs=x)
        return {"output": x}

    def generate(self, num_samples, *args, **kwargs):
        x, output_dict = self.stochastic_layer.generate(num_samples, *args, **kwargs)
        x = self.decoder(inputs=x)
        output_dict.update({"output": x})
        return output_dict

    def autoencode(self, inputs, *args, **kwargs):
        output_dict = self.mean_encode(inputs)
        return self.decode(output_dict["z"])


class ConvDetAutoencoderModel(tf.keras.Model):
    def __init__(self, config, dim_latent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.ae_type = "deterministic"
        self.dim_latent = dim_latent
        self.encoder = ConvEncoderLayer(config=config)
        self.decoder = ConvDecoderLayer(config=config)
        self.deterministic_layer = DeterministicLayer(config=config, 
            dim_latent=self.dim_latent)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        x = self.encoder(inputs=x)
        x, output_dict = self.deterministic_layer(inputs=x)
        x = self.decoder(inputs=x)
        output_dict.update({"output": x, "inputs": inputs})
        return output_dict

    def encode(self, inputs, *args, **kwargs):
        x = inputs
        x = self.encoder(inputs=x)
        _, output_dict = self.deterministic_layer(inputs=x)
        return output_dict

    def mean_encode(self, inputs, *args, **kwargs):
        return self.encode(inputs)

    def decode(self, x, *args, **kwargs):
        x = self.deterministic_layer.set_encodings(x)
        x = self.decoder(inputs=x)
        return {"output": x}

    def autoencode(self, inputs, *args, **kwargs):
        return self.call(inputs)


class LinearModelPredictLatents(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dim_latent = config.dim_latent

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(units=self.dim_latent, 
            activation=None)
        self.built = True

    def call(self, inputs):
        return self.dense(inputs)
