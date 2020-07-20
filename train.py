import tensorflow as tf

def train_model(model, dataset, epochs):
    for epoch in range(epochs):
        print('doing epoch',epoch)
        
        bar = tf.keras.utils.Progbar(tf.data.experimental.cardinality(dataset).numpy())

        for step, (x, y) in enumerate(dataset):
            l = model.step(x, y)
            bar.add(1, values = [('loss', l)])
        
        
        
