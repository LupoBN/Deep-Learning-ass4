import random

import numpy as np
import dynet as dy

def train(strain, mtrain, dev, num_of_iterations, trainer, model, save_file, batch_size=32):
    dev_results = list()
    train_results = list()
    best_acc = 0.0
    take_from_s_train = int(0.15 * len(strain))
    for epoch in range(num_of_iterations):

        train = mtrain + random.sample(strain, take_from_s_train)
        np.random.shuffle(train)

        mini_batches = [train[i:i + batch_size] for i in range(0, len(train), batch_size)]
        correct = 0.0
        incorrect = 0.0
        sum_of_losses = 0.0

        for mini_batch in mini_batches:
            dy.renew_cg()
            losses = []
            for (premise, hypothesis), label in mini_batch:
                prediction, loss = model.forward(premise, hypothesis, label)
                losses.append(loss)
                if prediction == label:
                    correct += 1.0
                else:
                    # print "Wrong, Predicted:", pred, "True Label:", label[i + 2]
                    incorrect += 1.0
            batch_loss = dy.esum(losses)
            sum_of_losses += batch_loss.value()
            batch_loss.backward()
            trainer.update()
            print "Current acc:", correct / (correct + incorrect)
            print "Current loss:", sum_of_losses / (correct + incorrect)

        print "Itertation:", epoch + 1
        train_acc = correct / (correct + incorrect)
        train_loss = sum_of_losses / (correct + incorrect)
        print "Training accuracy:", train_acc
        print "Training loss:", train_loss
        dev_acc, dev_loss = test(dev, model)
        dev_results.append(dev_acc)
        if dev_acc > best_acc:
            best_acc = dev_acc
            model.save_model(save_file)
        print "Test accuracy:", dev_acc
        print "Test loss:", dev_loss
    return train_results, dev_results


def test(dev, model, batch_size = 1):
    np.random.shuffle(dev)
    sum_of_losses = 0.0
    correct = 0.0
    incorrect = 0.0
    mini_batches = [dev[i:i + batch_size] for i in range(0, len(dev), batch_size)]
    for mini_batch in mini_batches:
        dy.renew_cg()
        losses = list()
        for (premise, hypothesis), label in mini_batch:
            prediction, loss = model.forward(premise, hypothesis, label)
            losses.append(loss)
            if prediction == label:
                correct += 1.0
            else:
                # print "Wrong, Predicted:", pred, "True Label:", label[i + 2]
                incorrect += 1.0
        batch_loss = dy.esum(losses)
        sum_of_losses += batch_loss.value()
    return correct / (correct + incorrect), sum_of_losses / (correct + incorrect)
