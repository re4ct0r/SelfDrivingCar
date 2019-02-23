#Computation
import numpy

#Image resizing & saving
from skimage import io
from skimage import transform as skitransform

#Keyboard reading keypresses
import keyboard

#Keyboard control
from pynput.keyboard import Key, Controller

#Threading & Timing
from threading import Thread
from Queue import Queue
import time
from time import sleep

#Fast screenshots
import mss
import mss.tools

#TensorFlow and TFLearn
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf

LEARN_LEFT_RIGHT_MODE = True
LEARN_UP_DOWN_MODE = False

#LEFT RIGHT network
tf.reset_default_graph()
network_left_right = input_data(shape=[None, 128, 128, 6], name='input')
network_left_right = conv_2d(network_left_right, 8, 3, activation='relu', name='l1')
network_left_right = max_pool_2d(network_left_right, 2)
network_left_right = conv_2d(network_left_right, 16, 3, activation='relu', name='l2')
network_left_right = max_pool_2d(network_left_right, 2)
network_left_right = conv_2d(network_left_right, 32, 3, activation='relu', name='l3')
network_left_right = max_pool_2d(network_left_right, 2)
network_left_right = conv_2d(network_left_right, 64, 3, activation='relu', name='l4')
network_left_right = max_pool_2d(network_left_right, 2)
network_left_right = conv_2d(network_left_right, 128, 3, activation='relu', name='l5')
network_left_right = max_pool_2d(network_left_right, 2)
network_left_right = conv_2d(network_left_right, 256, 3, activation='relu', name='l6')
network_left_right = max_pool_2d(network_left_right, 2)
network_left_right = fully_connected(network_left_right, 2, activation='softmax')
network_left_right = regression(network_left_right, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)
model_left_right = tflearn.DNN(network_left_right, tensorboard_verbose=0)
#load a model either for more training or just for controlling the game
#model_left_right.load("best_model_left_right/model_left_right_17400.tfl")



#UP DOWN network
tf.reset_default_graph()
network_up_down = input_data(shape=[None, 128, 128, 6], name='input')
network_up_down = conv_2d(network_up_down, 8, 3, activation='relu', name='l1_2')
network_up_down = max_pool_2d(network_up_down, 2)
network_up_down = conv_2d(network_up_down, 16, 3, activation='relu', name='l2_2')
network_up_down = max_pool_2d(network_up_down, 2)
network_up_down = conv_2d(network_up_down, 32, 3, activation='relu', name='l3_2')
network_up_down = max_pool_2d(network_up_down, 2)
network_up_down = conv_2d(network_up_down, 64, 3, activation='relu', name='l4_2')
network_up_down = max_pool_2d(network_up_down, 2)
network_up_down = conv_2d(network_up_down, 128, 3, activation='relu', name='l5_2')
network_up_down = max_pool_2d(network_up_down, 2)
network_up_down = conv_2d(network_up_down, 256, 3, activation='relu', name='l6_2')
network_up_down = max_pool_2d(network_up_down, 2)
network_up_down = fully_connected(network_up_down, 2, activation='softmax')
network_up_down = regression(network_up_down, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)
model_up_down = tflearn.DNN(network_up_down, tensorboard_verbose=0)
#load a model either for more training or just for controlling the game
#model_up_down.load("best_model_up_down/model_up_down_9200.tfl")

#screenshot area
monitor = {"top": 52, "left": 1, "width": 1280, "height": 720}

#this is needed for controlling the keyboard
keyboardController = Controller()


############################# KEYBOARD CONTROL

steering_factor_left_right = 0
steering_factor_up_down = 0

def keyboardControl():

  up_status = 0
  steering_counter_left_right = 0
  steering_counter_up_down = 0
  required_speed = 1
  global steering_factor_left_right
  global steering_factor_up_down

  while 1:
    steering_counter_left_right = steering_counter_left_right + 1
    steering_counter_up_down = steering_counter_up_down + 1
    steering_factor_left_right2 = abs(steering_factor_left_right)
    steering_factor_up_down2 = abs(steering_factor_up_down)

    if steering_factor_left_right < 0 and (int(steering_counter_left_right) % int(10)) < steering_factor_left_right2:
      #print("TURN LEFT")
      keyboardController.press(Key.left)
      keyboardController.release(Key.right)
    elif steering_factor_left_right > 0 and (int(steering_counter_left_right) % int(10)) < steering_factor_left_right2:
      #print("TURN RIGHT")
      keyboardController.release(Key.left)
      keyboardController.press(Key.right)
    else:
      #print("DON'T TURN")
      keyboardController.release(Key.left)
      keyboardController.release(Key.right)
      #steering_counter_left_right = 0

    if steering_factor_up_down < 0 and (int(steering_counter_up_down) % int(10)) < steering_factor_up_down2:
      #print("UP")
      keyboardController.press(Key.up)
      keyboardController.release(Key.down)
    elif steering_factor_up_down > 0 and (int(steering_counter_up_down) % int(10)) < steering_factor_up_down2:
      #print("DOWN")
      keyboardController.release(Key.up)
      keyboardController.press(Key.down)
    else:
      #print("DON'T UP DOWN")
      keyboardController.release(Key.up)
      keyboardController.release(Key.down)
      #steering_counter_up_down = 0
      #print(steering_factor_up_down)
    sleep(0.05)




worker = Thread(target=keyboardControl)
worker.setDaemon(True)
worker.start()

############################# END OF KEYBOARD CONTROL


############################# CONTROL AND LEARN

training_keys_left_right = [0,0]
training_keys_up_down = [0,0]

required_speed = 0

control_iteration = 0
def control(float_image, previous_float_image):
  global control_iteration
  control_iteration = control_iteration + 1
  #start = time.time()
  global steering_factor_left_right
  global steering_factor_up_down
  global required_speed

  concatenated = numpy.concatenate((float_image, previous_float_image), axis=2)

  # calculate predictions from the model
  prediction_left_right = model_left_right.predict([concatenated])
  steering_factor_left_right = (int)(prediction_left_right[0][1]*20-10)

  print("steering_factor_left_right = %d" % steering_factor_left_right)

  ## Use this piece of code if you want to control the acceleration/deceleration with the model
  '''
  prediction_up_down = model_up_down.predict([concatenated])
  steering_factor_up_down = (int)(prediction_up_down[0][1]*20-10)
  '''
  steering_factor_up_down = -2


  learn_left_right = False
  learn_up_down = False

  # use "a" and "d" to teach the model to steer left and right
  # use "q" and "w" to override models decision without teaching it
  if LEARN_LEFT_RIGHT_MODE:
    if keyboard.is_pressed('a'):
      steering_factor_left_right = -10
      training_keys_left_right[0] = 1
      training_keys_left_right[1] = 0
      learn_left_right = True
    elif keyboard.is_pressed('d'):
      steering_factor_left_right = 10
      training_keys_left_right[0] = 0
      training_keys_left_right[1] = 1
      learn_left_right = True

    if keyboard.is_pressed('q'):
      steering_factor_left_right = -10
    elif keyboard.is_pressed('e'):
      steering_factor_left_right = 10
  
  if LEARN_UP_DOWN_MODE:
    if keyboard.is_pressed('o'):
      training_keys_up_down[0] = 1
      training_keys_up_down[1] = 0
      learn_up_down = True
    elif keyboard.is_pressed('l'):
      training_keys_up_down[0] = 0
      training_keys_up_down[1] = 1
      learn_up_down = True

  '''
  if keyboard.is_pressed('r'):
    required_speed = required_speed + 1
    if required_speed > 10:
      required_speed = 10
  if keyboard.is_pressed('f'):
    required_speed = required_speed - 1
    if required_speed < 0:
      required_speed = 0

  print("required_speed = %d" % required_speed )

  steering_factor_up_down = required_speed * (-1)
  '''

  if LEARN_LEFT_RIGHT_MODE:  
    if learn_left_right:
      if training_keys_left_right[1] == 1:
        print("LEARNING: RIGHT")
      else:
        print("LEARNING: LEFT")
      model_left_right.fit([concatenated], [training_keys_left_right], n_epoch=1, show_metric=False, batch_size=1, snapshot_epoch=False)


  if LEARN_UP_DOWN_MODE:      
    if learn_up_down:
      if training_keys_up_down[1] == 1:
        print("LEARNING: DOWN")
      else:
        print("LEARNING: UP")
      model_up_down.fit([concatenated], [training_keys_up_down], n_epoch=1, show_metric=False, batch_size=1, snapshot_epoch=False)

  if (control_iteration % 200 == 0):
    if LEARN_LEFT_RIGHT_MODE:
      model_left_right.save("model_left_right/model_left_right_%d.tfl" % control_iteration)
    if LEARN_UP_DOWN_MODE:
      model_up_down.save("model_up_down/model_up_down_%d.tfl" % control_iteration)


  #end = time.time()
  #print(end - start)


def crawlControl(q):
  while 1:
    work = q.get()
    control(work[0], work[1])
    q.task_done()

controlQueue = Queue(maxsize=0)

worker = Thread(target=crawlControl, args=(controlQueue,))
worker.setDaemon(True)
worker.start()

############################# END OF CONTROL AND LEARN


############################# RESIZE

global previous_float_image
previous_float_image = None

def resizeImage(sct_img):
  global previous_float_image

  img = numpy.array(sct_img)
  shrinked_img = skitransform.resize(img, (128, 128))
  shrinked_img = shrinked_img[:,:,0:3]

  #convert from BGR to RGB
  shrinked_img = numpy.concatenate( ( shrinked_img[:,:,2:3], shrinked_img[:,:,1:2], shrinked_img[:,:,0:1] ), axis = 2 )

  float_image = shrinked_img.astype(numpy.float32)

  if previous_float_image is None:
    previous_float_image = float_image
    return

  controlQueue.put((float_image, previous_float_image))  

  previous_float_image = float_image


def crawlResize(q):
  while 1:
    work = q.get()
    resizeImage(work)
    q.task_done()


resizeQueue = Queue(maxsize=0)

worker = Thread(target=crawlResize, args=(resizeQueue,))
worker.setDaemon(True)
worker.start()

############################# END OF RESIZE




############################# MAIN LOOP

while True:
  start = time.time()

  with mss.mss() as sct:
    sct_img = sct.grab(monitor)

    if resizeQueue.qsize()<3:
      resizeQueue.put((sct_img))
    else:
      print("SKIPPED!!!!!!!!!!")

    sleep(0.05)

    end = time.time()
    #print(end - start)


############################# END OF MAIN LOOP






