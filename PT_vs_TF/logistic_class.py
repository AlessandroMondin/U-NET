import tensorflow as tf
import torch

# this class contains 3 methods: 1-softmax, 2-cross_entropy, 3-accuracy
# the only reasons why I created a class is to expertise OOP and to create consistency
# avoid adding an extra-parameter "lib" for each of the functions


class Logistic:
    def __init__(self, library):
        self.library = library
        assert self.library in ["tf", "pt"], 'You must choose between "tf" and "pt"'

    def softmax(self, logits):
        """
        softmax implementation
        args:
        - logits [tensor]: 1xN logits te|nsor
        returns:
        - soft_logits [tensor]: softmax of logits
        """

        if self.library == "tf":
            exp = tf.exp(logits)
            denom = tf.math.reduce_sum(exp, 1, keepdims=True)

        else:
            exp = torch.exp(logits)
            denom = torch.sum(exp, dim=1, keepdim=True)

        return exp/denom

    def cross_entropy(self, scaled_logits, one_hot):
        """
        CE implementation
        args:
        - scaled_logits [tensor]: NxC tensor where N batch size / C number of classes
        - one_hot [tensor]: one hot tensor
        returns:
        - loss [tensor]: cross entropy
        """
        if self.library == "tf":
            masked_logits = tf.boolean_mask(scaled_logits, one_hot)
            ce = -tf.math.log(masked_logits)
        else:
            masked_logits = torch.masked_select(scaled_logits, one_hot)
            ce = -torch.log(masked_logits)
        return ce

    def accuracy(self, y_hat, Y):
        """
        calculate accuracy
        args:
        - y_hat [tensor]: NxC tensor of models predictions
        - y [tensor]: N tensor of ground truth classes
        returns:
        - acc [tensor]: accuracy
        """

        if self.library == "tf":
            # calculate argmax
            argmax = tf.cast(tf.argmax(y_hat, axis=1), Y.dtype)
            # calculate acc
            acc = tf.math.reduce_sum(tf.cast(argmax == Y, tf.int32)) / Y.shape[0]
        else:
            argmax = torch.argmax(y_hat, dim=1)
            acc = torch.sum(torch.eq(argmax, Y)) / Y.shape[0]

        return acc