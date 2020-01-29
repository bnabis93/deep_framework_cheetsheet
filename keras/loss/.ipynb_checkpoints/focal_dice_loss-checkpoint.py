from keras import backend as K


def hybrid_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def hybrid_focal_dice_loss(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        focal_loss = K.sum(focal_loss, axis = 1)
        
        # Sum the losses in mini_batch
        Ncl = y_pred.shape[-1]
        w = K.zeros(shape=(Ncl,))
        w = K.sum(y_true, axis=(1,2))
        w = 1/(w**2+0.000001)
        # Compute gen dice coef:
        numerator = y_true*y_pred
        numerator = w*K.sum(numerator,(1,2))
        numerator = K.sum(numerator)

        denominator = y_true+y_pred
        denominator = w*K.sum(denominator,(1,2))
        denominator = K.sum(denominator)

        gen_dice_coef = 2*(numerator+K.epsilon()) / (denominator + K.epsilon())
        gen_dice_loss = (1- gen_dice_coef)
        
        
        return focal_loss + gen_dice_loss

    return hybrid_focal_dice_loss
