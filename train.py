import tensorflow as tf
import tqdm

def train_model(model, dataset, epochs):
    for epoch in range(epochs):
        print('doing epoch',epoch)

        for step, (x, y) in enumerate(tqdm.tqdm(dataset)):
            model.step(x, y)
        
        
        
