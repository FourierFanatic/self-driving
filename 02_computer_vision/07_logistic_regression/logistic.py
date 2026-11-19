import tensorflow as tf
from utils import check_softmax, check_acc, check_model, check_ce

def softmax(logits):
    """
    softmax implementation
    args:
    - logits [tensor]: 1xN logits tensor
    returns:
    - soft_logits [tensor]: softmax of logits
    """
    # IMPLEMENT THIS FUNCTION
    sum = tf.reduce_sum(tf.math.exp(logits))
    soft_logits = tf.math.exp(logits)/sum

    return soft_logits


def cross_entropy(scaled_logits, one_hot):
    """
    Cross entropy loss implementation
    args:
    - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
    - one_hot [tensor]: one hot tensor
    returns:
    - loss [tensor]: cross entropy 
    """
    # IMPLEMENT THIS FUNCTION


    return -1*tf.reduce_sum(tf.math.log(scaled_logits)*one_hot)


def model(X, W, b):
    """
    logistic regression model
    args:
    - X [tensor]: input HxWx3
    - W [tensor]: weights
    - b [tensor]: bias
    returns:
    - output [tensor]
    """
    # IMPLEMENT THIS FUNCTION
#     print(X.shape)
#     print(W.shape)
#     print(b.shape)
#     print("b",b.numpy())
#     print("X@W",(tf.reshape(X,[1,-1])@W).numpy())
#     print("X@W+b",((tf.reshape(X,[1,-1])@W) + b).numpy())
    return (tf.reshape(X,[1,-1])@W) + b


def accuracy(y_hat, Y):
    """
    calculate accuracy
    args:
    - y_hat [tensor]: NxC tensor of models predictions
    - y [tensor]: N tensor of ground truth classes
    returns:
    - acc [tensor]: accuracy
    """
    # IMPLEMENT THIS FUNCTION
#     print(y_hat.numpy())
#     print(tf.argmax(y_hat, axis=1).shape)
#     print(Y.shape)
    args = tf.argmax(y_hat,axis=1)
#     print("args", args.numpy())
#     print("Y",Y.numpy())
    comp = tf.equal(tf.cast(args,Y.dtype),Y)
#     print("Comp",comp)
    right = tf.reduce_sum(tf.cast(comp,tf.int32))
#     print(comp.shape[0])
#     print("Sum",right.numpy())
    return right/comp.shape[0]

if __name__ == '__main__':
    # checking the softmax implementation
    check_softmax(softmax)

    # checking the NLL implementation
    check_ce(cross_entropy)

    # check the model implementation
    check_model(model)

    # check the accuracy implementation
    check_acc(accuracy)