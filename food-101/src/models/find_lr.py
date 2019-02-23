
'''
This function is used to find lr by training over one epoch 
and varying the lr from 1e-8 to 10 to find the best lr
'''

# import necessary libraries
import keras.backend as K
import math
from tqdm import tqdm

def find_lr(model, train_generator, batch_size, init_value = 1e-8, final_value=10., beta = 0.98):
    
    num = len(train_generator)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    K.set_value(model.optimizer.lr, lr)
    
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    
    for i in tqdm(range(train_generator.samples // batch_size)):
        batch_num += 1

        data = next(train_generator)
    
        history = model.fit(data[0], data[1])
        loss = history.history['loss'][0]
        
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        #Update the lr for the next step
        lr *= mult
        K.set_value(model.optimizer.lr, lr)

    # return the log10 lrs and smoothed losses    
    return log_lrs, losses