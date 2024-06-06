import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE, KLD, CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Conv1D, GlobalMaxPooling1D
from spektral.layers import GraphAttention, GraphConvSkip, TAGConv, GraphConv
from tensorflow.keras.initializers import GlorotUniform
import numpy as np
from sklearn import metrics


class GCNTree(tf.keras.Model):

    def __init__(self, X, adj, adj_n, num_classes, batch_size=32, hidden_dim=16, latent_dim=8, dec_dim=None,
                 adj_dim=32, depth=4):
        super(GCNTree, self).__init__()
        if dec_dim is None:
            dec_dim = [8, 16]

        batch_size = X.shape[0]
        self.latent_dim = latent_dim
        self.X = X
        self.adj = np.float32(adj)
        self.adj_n = np.float32(adj_n)
        self.in_dim = X.shape[1]
        self.n_sample = X.shape[0]
        self.sparse = False
        self.num_classes = num_classes
        initializer = GlorotUniform(seed=7)

        # Encoder
        X_input = Input(shape=self.in_dim)
        h = X_input
        self.sparse = True
        A_in = Input(shape=self.n_sample,
                     sparse=True)
        h = GraphConv(channels=hidden_dim, kernel_initializer=initializer, activation="relu")(
            [h, A_in])
        z_mean = GraphConv(channels=latent_dim, kernel_initializer=initializer, activation="relu")(
            [h, A_in])

        h = Dense(units=2 ** depth, activation="relu")(z_mean)
        features = h
        self.depth = depth
        self.num_leaves = 2 ** depth #leavas number

        used_features_rate = 1.0
        num_features = features.shape[1]
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )
        features = tf.matmul(
            features, self.used_features_mask
        )
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )
        decisions = tf.keras.layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )
        mu = tf.reshape(features[:, 1], [batch_size, 1, 1])
        begin_idx = 1
        end_idx = 2
        for level in range(self.depth):
            mu = tf.reshape(features[:, begin_idx:end_idx], [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                              :, begin_idx:end_idx, :
                              ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = tf.keras.activations.relu(self.pi)  # [num_leaves, num_classes]
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        self.tmodel = Model(inputs=[X_input, A_in], outputs=outputs, name="model")


    def train(self, X, adj_n, train_index, test_index, y_train, epochs, batch_size, lr=0.001, verbose=0):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        for epoch in range(1, epochs + 1):
            train_dataset = tf.data.Dataset.from_tensor_slices((X, adj_n))
            with tf.GradientTape(persistent=True) as tape:
                outputs = self.tmodel([X, adj_n])
                loss =  tf.reduce_mean(MSE(y_train, tf.gather(outputs, indices=train_index, axis=0)))
            vars = self.trainable_weights
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))

        pre = self.tmodel.predict([X, adj_n], batch_size=adj_n.shape[0])[test_index]
        return pre

