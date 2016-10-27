import numpy as np

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# df = pd.DataFrame({'x' : [v[0] for v in vectors_set], 'y' : [v[1] for v in vectors_set]})
# sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
# plt.show(block=True)

import tensorflow as tf

vectors = tf.constant(vectors_set)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0,0], [k,-1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

diff = tf.sub(expanded_vectors, expanded_centroides)
sqr = tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)

new_centers = []
for c in range(k):
    locations = tf.equal(assignments, c)
    clustered_points = tf.gather(vectors, tf.reshape(tf.where(locations), [1,-1]))
    new_centers.append(tf.reduce_mean(clustered_points, reduction_indices=[1]))

means = tf.concat(0, new_centers)

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

for step in range(100):
    _, centroid_values, assignments_values = sess.run([update_centroides, centroides, assignments])

data = {'x' : [], 'y' : [], 'cluster' : []}

for i in range(len(assignments_values)):
    data['x'].append(vectors_set[i][0])
    data['y'].append(vectors_set[i][1])
    data['cluster'].append(assignments_values[i])

df = pd.DataFrame(data)
sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, hue='cluster', legend=False)
plt.show()
