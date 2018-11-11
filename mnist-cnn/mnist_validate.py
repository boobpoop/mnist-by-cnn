import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import time
import forward_propagation
import mnist_train

#EVAL_INTERVAL_SECS = 10

def valuate(mnist):
    input_data_x = tf.placeholder(tf.float32, [None, forward_propagation.INPUT_NODE], name = "input_data_x")
    input_data_y = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE], name = "input_data_y")

    y = forward_propagation.forward_propagation(input_data_x, False, None)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_data_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    validate_feed = {input_data_x: mnist.validation.images, input_data_y: mnist.validation.labels}
    print(validate_feed)
    print("*")
    for variables in tf.all_variables():
        print(variables)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
            accuracy_prediction = sess.run(accuracy, feed_dict = validate_feed)
            print("accuracy on validation data is %g" %(accuracy_prediction))
        else:
            print("No checkpoint file found")
            return
     

def main(argc = None):
    mnist = input_data.read_data_sets("mnist_data", one_hot = True)
    valuate(mnist)

if __name__ == "__main__":
    tf.app.run()
