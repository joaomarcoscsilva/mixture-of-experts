import tensorflow as tf

def train_model(model, dataset, epochs, test_dataset = None, wandb_config = None):
    
    # Since for classification the Mixture of Experts loss is the same as categorical_crossentropy, I just call fit here
    # However, there is some code for a custom training loop that must be used if this code is to be adapted for regression
    # This is because the loss for regression in the MoE model is different than the normally used MSE loss.
    if wandb_config is not None:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project = 'mixture-of-experts', config = wandb_config)
        model.fit(dataset, epochs = epochs, validation_data = test_dataset, callbacks = [WandbCallback(save_weights_only = True, log_batch_frequency = 128)])
    else:
        model.fit(dataset, epochs = epochs, validation_data = test_dataset)

    """
    for epoch in range(epochs):
        
        bar = tf.keras.utils.Progbar(tf.data.experimental.cardinality(dataset).numpy())

        for step, (x, y) in enumerate(dataset):
            l = model.step(x, y)
            bar.add(1, values = [('loss', l)])
    """ 
        
        
