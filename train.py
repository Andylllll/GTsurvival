import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import keras.backend as K
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from GTsurv import GCNTree
from graph_function import get_adj
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def Gsimulation(X_ori, xx, S_area, prob, tau):
    scaler = MinMaxScaler()
    scaler.fit(xx)
    xx_norm = scaler.transform(xx)
    transfer = StandardScaler()
    X = transfer.fit_transform(X_ori)

    batch_size = X.shape[0]
    ntau = xx_norm.shape[1]
    from sklearn.model_selection import train_test_split
    random_state = 8
    indices = np.arange(X.shape[0])
    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_index,
        test_index,
    ) = train_test_split(X, xx_norm, indices, train_size=0.7, random_state=random_state)

    adj, adj_n = get_adj(X, k=10, pca=False)
    model = GCNTree(X, adj=adj, adj_n=adj_n, depth=3, batch_size=batch_size, num_classes=ntau)

    pre = model.train(X=X, adj_n=adj_n,
                      train_index=train_index,
                      test_index=test_index,
                      y_train=y_train,
                      epochs=500, batch_size=batch_size)
    scaler.inverse_transform(np.array(pre))
    true_RM = S_area[test_index]
    metrx = {
             'MAE': mean_absolute_error(true_RM, pre),
             'MSE': mean_squared_error(true_RM, pre)
     }

    K.clear_session()
  #  tf.reset_default_graph()


    return metrx
