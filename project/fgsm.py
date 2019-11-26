import tensorflow as tf

def create_adversarial_pattern(input_image, input_label,models,loss_object = tf.keras.losses.CategoricalCrossentropy()):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = models(input_image)
        loss = loss_object(input_label, prediction)

      # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
      # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)
    return signed_grad