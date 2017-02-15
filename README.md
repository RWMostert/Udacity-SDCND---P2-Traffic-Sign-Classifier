
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

## Step 1: Dataset Summary & Exploration

The dataset used: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The csv file from the dataset contains a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 2D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 

```python
# Number of training examples
n_train = len(X_train)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = X_test[58].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 39209
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


We see that the dataset is a list of images (traffic signs), with a list of their actual labels, represented as an integer.  We can convert these integers to descriptions using the signnames.csv file.

First, let's take a look at some of the images and what their labels look like:

```python

sign_names = {}
with open('signnames.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
         sign_names[row[0]] = row[1]

def show_sign_at_index(i, images, classifications):
    img = images[i]
    sign_prediction = classifications[i]
    print(str(sign_prediction) + ":  " + sign_names[str(sign_prediction)])
    plt.axis('off')
    plt.imshow(img)
    plt.show();
    print(img.shape)
```


```python
interact(show_sign_at_index, i=(0,len(X_train)), classifications=fixed(y_train), images = fixed(X_train));
```

    12:  Priority road



![png](output_8_1.png)


    (32, 32, 3)


----

## Step 2: Design and Test a Model Architecture


### Implementation

##### Preprocessing
I tried multiple normalisation techniques for this project, however I ultimately settled for the most simple of all: dividing each pixel value by 255 and subtracting 0.5 to center it around zero.
    <br>
The reason I picked such a simple normalisation procedure is mostly for ease of use. Subtracting the mean of the image set and dividing by the standard deviation didn't make sense in this context, since it didn't provide any significant performance improvements and it creates difficulty when working with new images to classify.

```python
def rgb_to_normalized_gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray/255. - 0.5

def normalize(imgs):
    imgs = np.array(imgs, dtype=np.float64)
    return imgs/255. - 0.5
```


```python
X_train_batches_gray = []
X_train_batches_color = []
y_train_batches = []

perm = np.random.permutation(len(X_train))[:40*980]
perm = perm.reshape((40,980))

images_gray = np.array([rgb_to_normalized_gray(image) for image in X_train])
images_color = normalize(X_train)

for j in range(40):
    x_train_batch_gray = images_gray[perm[j]]
    x_train_batch_color = images_color[perm[j]]
    y_train_batch = y_train[perm[j]]
    
    X_train_batches_gray.append(x_train_batch_gray)
    X_train_batches_color.append(x_train_batch_color)
    y_train_batches.append(y_train_batch)
```


```python
X_validation_gray =  np.array([rgb_to_normalized_gray(image) for image in X_test])[7500:]
X_validation_color =  normalize(X_test)[7500:]
y_validation = y_test[7500:]

X_test_gray =  np.array([rgb_to_normalized_gray(image) for image in X_test])[:7500]
X_test_color =  normalize(X_test)[:7500]
y_test = y_test[:7500]
```

##### Data Augmentation (credit: Vivek Yadav)

Beyond normalisation, I also followed Dr. Vivek Yadav's guide on dataset augmentation, which helps to prevent overfitting on the small dataset provided, and helps generalise the model to many different angles. In addition to Dr. Yadav's tilting/shearing procedure, I did brightness variation, which helped to further generalise the dataset provided to many different lighting conditions.
<br>
My data augmentation is called from my training routine, as it allows for fine grain control over the amount of change to the dataset at different stages in the training procedure. I gradually reduce the augmented dataset as training progresses, in order to allow the model to learn the more subtle features of the dataset.

```python
def transform_image(img,ang_range,shear_range,trans_range, reduction_coeff):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over. 
    
    A Random uniform distribution is used to generate different parameters for transformation
    
    '''
    ang_range*=reduction_coeff
    shear_range*=reduction_coeff
    trans_range*=reduction_coeff
    
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    
    rows,cols, colours = 0,0,0
    if len(img.shape) == 2:
        rows,cols = img.shape
    else:
        rows,cols, colours = img.shape
        
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    # Brightness Variation
    
    if len(img.shape) == 3:
        value = np.random.uniform(high=100.) - 50
        value*=reduction_coeff
        coef = 1
        if value < -1:
            coef = -1
            value = abs(value)

        value = np.array(value, dtype=np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
        if coef == 1:
            hsv[:,:,2] = np.where((255 - hsv[:,:,2]) < value,255,hsv[:,:,2]+value)
        else:
            hsv[:,:,2] = np.where( hsv[:,:,2] < value,0,hsv[:,:,2]-value)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img
```

# Architecture

### Sermanet Model (Sermanet Team, LeCun paper)

**This model is based on the model proposed by Yann LeCun in his paper: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf.**
<br>
I read through and tried to mimick the network proposed by Yann LeCun in the paper referenced above.  He describes a very simple 2-layer convolutional neural network submitted by the Sermanet team in the traffic sign classification competition. This conv-net consists of two convolutional layers, of size 108 each, both feeding into a single fully-connected layer of size 100.  I mimicked this approach (Sermanet model).


```python
def conv2d(name, l_input, w, b):
    return tf.nn.elu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], 
                                                  padding='SAME'),b), name=name)

def max_pool(name, l_input, k, stride):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], 
                          padding='SAME', name=name)

def serma_net(_X, _weights, _biases, _dropout):
    
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 32, 32, 1])
    
    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2, stride=2)
    
    # Apply Dropout
    conv1_out = tf.nn.dropout(pool1, _dropout)
    
    # Convolution Layer
    conv2 = conv2d('conv2', conv1_out, _weights['wc2'], _biases['bc2'])
    
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=2, stride=2)
    
    # Apply Dropout
    conv2_out = tf.nn.dropout(pool2, _dropout)
    
    # Max Pooling (down-sampling) applied to the first convolutional layer
    conv1_bridge = max_pool('pool1', conv1_out, k=5, stride=3)
    
    # Fully connected layer
    flat1 = tf.contrib.layers.flatten(conv1_bridge)
    flat2 = tf.contrib.layers.flatten(conv2_out)
    concat = tf.concat(1, [flat1, flat2])
    dense1 = tf.contrib.layers.flatten(concat)
    
    # Elu activation
    dense1 = tf.nn.elu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)
                       
    # Output, class prediction
    out = tf.matmul(dense1, _weights['out']) + _biases['out']
    return out
```

###### Tensorflow Graph 1


```python
tf.reset_default_graph()

g1 = tf.Graph()
with g1.as_default() as g:
    with g.name_scope("EBLearn") as g1_scope:
        
        # Parameters
        learning_rate = 0.001
        n_input = 32*32*1 # Traffic Signs data input (img shape)
        n_classes = 43 # Traffic Signs total classes (43 different types of signs)
        dropout = 0.5 # Dropout
        
        # Input Placeholders
        EB_x = tf.placeholder(tf.float32, [None, 32, 32])
        EB_y = tf.placeholder(tf.float32, [None, n_classes])
        EB_keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

        # Model Weights & Biases
        EB_weights = {
            'wc1': tf.get_variable("conv1_w", shape=[5, 5, 1, 108], initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable("conv2_w", shape=[5, 5, 108, 108], initializer=tf.contrib.layers.xavier_initializer()),
            
            'wd1': tf.get_variable("dense1_w", shape=[10800, 100], initializer=tf.contrib.layers.xavier_initializer()),
            
            'out': tf.get_variable("out_w", shape=[100, n_classes], initializer=tf.contrib.layers.xavier_initializer())
        }
        EB_biases = {
            'bc1': tf.get_variable("conv1_b", shape=[108], initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': tf.get_variable("conv2_b", shape=[108], initializer=tf.contrib.layers.xavier_initializer()),

            'bd1': tf.get_variable("dense1_b", shape=[100], initializer=tf.contrib.layers.xavier_initializer()),
            'bd2': tf.get_variable("dense2_b", shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),

            'out': tf.get_variable("out_b", shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer())
        }

        # Loss
        EB_model_output = serma_net(EB_x, EB_weights, EB_biases, EB_keep_prob)
        EB_prediction = tf.nn.softmax(EB_model_output)
        EB_cross_entropy = -tf.reduce_sum(EB_y * tf.log(EB_prediction + 1e-6), reduction_indices=1)
        EB_cost = tf.reduce_mean(EB_cross_entropy)

        # Optimization
        EB_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(EB_cost)

        # Model Evaluation
        EB_correct_pred = tf.equal(tf.argmax(EB_prediction,1), tf.argmax(EB_y,1))
        EB_accuracy = tf.reduce_mean(tf.cast(EB_correct_pred, tf.float32))
```

#### Training SermaNet
For training the model, I used the well known and loved Adam Optimizer with a learning rate of 0.001, set to minimize the cross entropy between the actual labels and the predicted labels.  My training data is split into 40 batches of 980 images each. 50 epochs seems to be more than enough to train this specific model on the training data provided.
<br><br>
I evaluated my model using both an accuracy figure (which indicates the amount of labels predicted correctly), as well as the cross-entropy figure - which indicates to what extent the model was "sure" about the predicted label (and whether it got it right).
<br><br>
As mentioned before, I augmented the training dataset, and gradually reduced the augmentation as training progressed - which allows the conv-net to learn the more subtle features of the (very low resolution) images.

```python
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "./SERMANET.ckpt")
    
    current_X_train_batches = X_train_batches_gray

    for step in range(100):
        print("Iteration "+str(step))
        for i in range(40):
            # Further split the batches due to memory constraints
            for j in range(20):
                
                batch_xs = current_X_train_batches[i][49*j:49*(j+1)]
                batch_ys = tf.one_hot(y_train_batches[i][49*j:49*(j+1)], n_classes).eval(session=sess)
                
                # Fit training using batch data
                sess.run(EB_optimizer, feed_dict={EB_x: batch_xs, EB_y: batch_ys, EB_keep_prob: dropout})

                # Accuracy & Loss
                acc = sess.run(EB_accuracy, feed_dict={EB_x: batch_xs, EB_y: batch_ys, EB_keep_prob: 1.})
                loss = sess.run(EB_cost, feed_dict={EB_x: batch_xs, EB_y: batch_ys, EB_keep_prob: 1.})

                print("Batch " + str(i) + ", Minibatch Loss= " \
                          + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                
        save_path = saver.save(sess, "./SERMANET.ckpt")
        print("Model saved in file: %s" % save_path)
        
        #Test Set performance:
        test_accuracy = []
        test_loss = []
        
        for j in range(20):
            test_xs = X_test_gray[50*j:50*(j+1)]
            test_ys = tf.one_hot(y_test[j*50:(j+1)*50], n_classes).eval(session=sess)
            
            acc = sess.run(EB_accuracy, feed_dict={EB_x: test_xs, EB_y: test_ys, EB_keep_prob:1.})
            loss = sess.run(EB_cost, feed_dict={EB_x: test_xs, EB_y: test_ys, EB_keep_prob: 1.})
        
            test_accuracy.append(acc)
            test_loss.append(loss)
            
        print("Test Set Accuracy:  " + str(np.mean(test_accuracy)) + \
              ", Loss: " + str(np.mean(test_loss)))

        # Augmented data generation:
        reduction_coefficient = 0.9**step
        current_X_train_batches = normalize([[transform_image(np.array((image+0.5)*255, dtype=np.uint8),10,5,3,reduction_coefficient) for image in batch] for batch in X_train_batches_gray])
```

#### SermaNet Validation


```python
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "./SERMANET.ckpt")
    
    validation_accuracy = []
    
    # Split up the validation set due to memory restrictions.
    for i in range(100):
        test_ys = tf.one_hot(y_validation[i*50:(i+1)*50], n_classes).eval(session=sess)
        acc = sess.run(EB_accuracy, feed_dict={EB_x: X_validation_gray[i*50:(i+1)*50], EB_y: test_ys, EB_keep_prob:1.})
        validation_accuracy.append(acc)
        
    print("Validation Accuracy for SermaNet:" + str(np.mean(validation_accuracy)))
```

    Validation Accuracy for SermaNet:0.9464


### Alexnet-based Model

I started with a simple convolutional neural network, based on Alexnet. It consists of 3 3x3 convolution layers (of sizes 32, 64 and 128 respectively), each with an elu activation function. The convolution layers are each fed into a 3x3 max pooling layer, followed by normalisation and dropout.  The final convolutional layer is flattened into a fully connected layer, which is followed by another fully connected layer, from which we then gather the output values.  These are then converted to class predictions using a standard sigmoid function.  Overall, a standard convolutional model.

```python
def conv2d(name, l_input, w, b):
    return tf.nn.elu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], 
                                                  padding='SAME'),b), name=name)

def max_pool(name, l_input, k, stride=1):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], 
                          padding='SAME', name=name)

def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def alex_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 32, 32, 3])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=3, stride=3)
    # Apply Normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Apply Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Convolution Layer
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = max_pool('pool2', conv2, k=3, stride=3)
    # Apply Normalization
    norm2 = norm('norm2', pool2, lsize=4)
    # Apply Dropout
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer
    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # Max Pooling (down-sampling)
    pool3 = max_pool('pool3', conv3, k=3, stride=3)
    # Apply Normalization
    norm3 = norm('norm3', pool3, lsize=4)
    # Apply Dropout
    norm3 = tf.nn.dropout(norm3, _dropout)
    
    # Fully Connected Layers
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.elu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    dense1 = tf.nn.dropout(dense1, _dropout)
    
    dense2 = tf.nn.elu(tf.matmul(dense1, _weights['wd2']), name='fc2')
    dense2 = dense2  + _biases['bd2']
    dense2 = tf.nn.dropout(dense2, _dropout)
    
    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out
```

###### Tensorflow Graph 2


```python
g2 = tf.Graph()
with g2.as_default() as g:
    with g.name_scope("AlexNet") as g2_scope:
        
        # Parameters
        learning_rate = 0.001
        n_input = 32*32*3 # Traffic Signs data input (img shape)
        n_classes = 43 # Traffic Signs total classes (43 different types of signs)
        dropout = 0.5 # Dropout
        
        # Input Placeholders
        AN_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        AN_y = tf.placeholder(tf.float32, [None, n_classes])
        AN_keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

        # Model Weights & Biases
        AN_weights = {
            'wc1': tf.get_variable("conv1_w", shape=[3, 3, 3, 32], initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable("conv2_w", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
            'wc3': tf.get_variable("conv3_w", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer()),
            
            'wd1': tf.get_variable("dense1_w", shape=[512, 1024], initializer=tf.contrib.layers.xavier_initializer()),
            'wd2': tf.get_variable("dense2_w", shape=[1024, 1024], initializer=tf.contrib.layers.xavier_initializer()),

            'out': tf.get_variable("out_w", shape=[1024, n_classes], initializer=tf.contrib.layers.xavier_initializer())
        }
    
        AN_biases = {
        'bc1': tf.get_variable("conv1_b", shape=[32], initializer=tf.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable("conv2_b", shape=[64], initializer=tf.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable("conv3_b", shape=[128], initializer=tf.contrib.layers.xavier_initializer()),
    
        'bd1': tf.get_variable("dense1_b", shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),
            'bd2': tf.get_variable("dense2_b", shape=[1024], initializer=tf.contrib.layers.xavier_initializer()),
    
            'out': tf.get_variable("out_b", shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer())
        }

        # Loss
        AN_model_output = alex_net(AN_x, AN_weights, AN_biases, AN_keep_prob)
        AN_prediction = tf.nn.softmax(AN_model_output)
        AN_cross_entropy = -tf.reduce_sum(AN_y * tf.log(AN_prediction + 1e-6), reduction_indices=1)
        AN_cost = tf.reduce_mean(AN_cross_entropy)

        # Optimization
        AN_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(AN_cost)

        # Model Evaluation
        AN_correct_pred = tf.equal(tf.argmax(AN_prediction,1), tf.argmax(AN_y,1))
        AN_accuracy = tf.reduce_mean(tf.cast(AN_correct_pred, tf.float32))

tf.reset_default_graph()
```

#### Training AlexNet

```python
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "./ALEXNET.ckpt")
    
    current_X_train_batches = X_train_batches_color
    
    for step in range(100):
        print("Iteration "+str(step))
        for i in range(40):
            # Fit training using batch data
            
            batch_xs = current_X_train_batches[i]
            batch_ys = tf.one_hot(y_train_batches[i], n_classes).eval(session=sess)
                
            sess.run(AN_optimizer, feed_dict={AN_x: batch_xs, AN_y: batch_ys, AN_keep_prob: dropout})

            acc = sess.run(AN_accuracy, feed_dict={AN_x: batch_xs, AN_y: batch_ys, AN_keep_prob: 1.})
            loss = sess.run(AN_cost, feed_dict={AN_x: batch_xs, AN_y: batch_ys, AN_keep_prob: 1.})

            print("Batch " + str(i) + ", Minibatch Loss= " \
                      + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        if step%10 == 0:
            save_path = saver.save(sess, "./ALEXNET.ckpt")
            print("Model saved in file: %s" % save_path)

            
        #Test Set performance:
        test_xs = X_test_color
        test_ys = tf.one_hot(y_test, n_classes).eval(session=sess)
            
        acc = sess.run(AN_accuracy, feed_dict={AN_x: test_xs, AN_y: test_ys, AN_keep_prob:1.})
        loss = sess.run(AN_cost, feed_dict={AN_x: test_xs, AN_y: test_ys, AN_keep_prob: 1.})
        
        print("Test Set Accuracy:  " + str(acc))
            
        
        # Augmented data generation:
        reduction_coefficient = 1
        if step > 10:
            reduction_coefficient = 0.9**(step-10)
        current_X_train_batches = normalize([[transform_image(np.array((image+0.5)*255, dtype=np.uint8),20*4,10*4,6*4,reduction_coefficient) for image in batch] for batch in X_train_batches_color])
```

###### Alexnet Validation


```python
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "./ALEXNET.ckpt")
    
    validation_accuracy = []
    
    # Split up the validation set due to memory restrictions.
    for i in range(100):
        test_ys = tf.one_hot(y_validation[i*50:(i+1)*50], n_classes).eval(session=sess)
        acc = sess.run(AN_accuracy, feed_dict={AN_x: X_validation_color[i*50:(i+1)*50], AN_y: test_ys, AN_keep_prob:1.})
        validation_accuracy.append(acc)
        
    print("Validation Accuracy for AlexNet:" + str(np.mean(validation_accuracy)))
```

    Validation Accuracy for AlexNet:0.9182


### Ensemble Model


```python
def ensemble_classify(imgs):
    colour_imgs = normalize(imgs)
    gray_imgs = [rgb_to_normalized_gray(image) for image in imgs]
    
    with tf.Session(graph = g1) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, "./SERMANET.ckpt")
        pred_1 = sess.run(EB_prediction, feed_dict={EB_x:gray_imgs, EB_keep_prob: 1.})
        
    with tf.Session(graph = g2) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, "./ALEXNET.ckpt")
        pred_2 = sess.run(AN_prediction, feed_dict={AN_x:colour_imgs, AN_keep_prob: 1.})
        
    prediction = (pred_1 + pred_2)/2
    return prediction
```

#### Ensemble Model Accuracy on the Validation Set


```python
predictions = np.empty(shape=(0, 43))
    
for i in range(102):
    pred = ensemble_classify((X_validation_color[i*50:(i+1)*50] + 0.5)*255)
    predictions = np.append(predictions, pred, axis=0)
```


```python
test_accuracy = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    y = tf.placeholder(tf.int32, [None])
    predicted_ys = tf.placeholder(tf.float32, [None, n_classes])
    actual_ys = tf.one_hot(y, n_classes)
    correct_pred = tf.equal(tf.argmax(predicted_ys,1), tf.argmax(actual_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    acc = sess.run(accuracy, feed_dict={predicted_ys: predictions, y: y_validation[:5100]})
    test_accuracy.append(acc)
    print("Ensemble Test Accuracy: "+str(acc*100)+"%")
```

    Ensemble Test Accuracy: 95.5348908901%

---

## Step 3: Test a Model on New Images


### Implementation


<strong>
The images below were fetched from a streetView drive around the Maximilaneum in Munich, and around the Breitscheidplatz in Berlin:
</strong>



```python
raw_images =[Image.open("./Breitscheidplatz/sign1.png"), \
            Image.open("./Breitscheidplatz/sign2.png"),  \
            Image.open("./Breitscheidplatz/sign3.png"),  \
            Image.open("./Breitscheidplatz/sign4.png"),  \
            Image.open("./Breitscheidplatz/sign6.png"),  \
            Image.open("./Breitscheidplatz/sign7.png"),  \
            Image.open("./Breitscheidplatz/sign8.png"),  \
            Image.open("./Breitscheidplatz/sign9.png"),  \
            Image.open("./Maximilaneum/sign1.png"),      \
            Image.open("./Maximilaneum/sign2.png"),      \
            Image.open("./Maximilaneum/sign3.png"),      \
            Image.open("./Maximilaneum/sign4.png"),      \
            Image.open("./Maximilaneum/sign5.png"),      \
            Image.open("./Maximilaneum/sign7.png"),      \
            Image.open("./Maximilaneum/sign8.png"),      \
            Image.open("./Maximilaneum/sign9.png"),      \
            Image.open("./Maximilaneum/sign10.png"),     \
            Image.open("./Maximilaneum/sign11.png"),     \
            Image.open("./Maximilaneum/sign12.png"),     \
            Image.open("./Maximilaneum/sign13.png"),     \
            Image.open("./Maximilaneum/sign14.png")]

maximilaneum_signs = []
streetview_signs = []

for image in raw_images:
    streetview_signs.append(np.asarray(ImageOps.fit(image, (32,32), Image.ANTIALIAS)))

streetview_classifications = ensemble_classify(streetview_signs)
```


```python
def show_sign_at_index(i, images, classifications):
    img = images[i]
    sign_prediction = classifications[i]
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    
    sorted_classification = np.argsort(sign_prediction)[::-1][:5]
    for j in range(len(sorted_classification)):
        print(str(j+1) + ": " + sign_names[str(sorted_classification[j])])
    
    fig = plt.figure()
    plt.bar( np.arange(1, 44), sign_prediction )
    plt.show()

interact(show_sign_at_index, i=(0,len(streetview_classifications)-1), classifications=fixed(streetview_classifications), images = fixed(streetview_signs));
```


![png](output_59_0.png)


    1: Keep right
    2: No passing for vechiles over 3.5 metric tons
    3: Turn left ahead
    4: Speed limit (100km/h)
    5: Speed limit (80km/h)



![png](output_59_2.png)


### Misclassifications

**Answer:**

## <font color='red'> #1: Misclassification: "Children Crossing" classified as "Road Work"
</font>

<font color='red'>
The image below was classified as "Road work", rather than "Children Crossing".  When looking at the scaled-down image, it's easy to see why (very low image resolution). This suggests that it might help to train the model on a higher definition set of images (or reduce the size of the cropping region). 
<br>
What's encouraging though, is that the network clearly indicates that it is uncertain, giving the classification a p-value below 0.5. In practice we might reject classifications with low p-values, try grab another image from the video and try to classify the image again.
</font>


```python
show_sign_at_index(15, classifications=streetview_classifications, images=streetview_signs)
```


![png](output_64_0.png)


    1: Road work
    2: Bicycles crossing
    3: Beware of ice/snow
    4: Wild animals crossing
    5: Children crossing



![png](output_64_2.png)


## <font color='red'> #2: Correct Classification: 60 km/h Speed Limit
</font>

<font color='red'>
The 60km/h speed limit sign below, however classified correctly, reveals a lot about the classification procedure. It's encouraging to see that the top 5 are all speed limit signs (which are all visually similar).
</font>


```python
show_sign_at_index(16, classifications=streetview_classifications, images=streetview_signs)
```


![png](output_67_0.png)


    1: Speed limit (60km/h)
    2: Speed limit (80km/h)
    3: Speed limit (50km/h)
    4: Speed limit (30km/h)
    5: Speed limit (100km/h)



![png](output_67_2.png)


## <font color='red'> #3: Correct Classification: No passing for vehicles over 3.5 metric tonnes
</font>

<font color='red'>Something to note with the classification below, is the top 5 classifications.  It's encouraging to see that the "no passing" sign (which is visually similar to the "no passing for vehicles over 3.5 metric tonnes" sign) was it's second choice.
</font>


```python
show_sign_at_index(17, classifications=streetview_classifications, images=streetview_signs)
```


![png](output_70_0.png)


    1: No passing for vechiles over 3.5 metric tons
    2: No passing
    3: No entry
    4: Stop
    5: Vechiles over 3.5 metric tons prohibited



![png](output_70_2.png)


## <font color='red'> #4: Uncertain Classification: Double Curve
</font>

<font color='red'>
The classifier was rather uncertain about the classification below as a double curve - and rightly so. At low definition, the sign is very similar to a "right of way at next intersection" sign.  A possible solution, as mentioned before, might be to train the model on higher definition images.
</font>


```python
show_sign_at_index(18, classifications=streetview_classifications, images=streetview_signs)
```


![png](output_73_0.png)


    1: Double curve
    2: Right-of-way at the next intersection
    3: Slippery road
    4: Beware of ice/snow
    5: Wild animals crossing



![png](output_73_2.png)


## <font color='red'> #4: Correct Classification: Traffic Signals
</font>

<font color='red'>
In earlier models, the sign below was constantly classified as a general caution sign. In the diagnostics I noticed that the mistake was SermaNet's fault.  This might have been due to SermaNet being trained on grayscale images (in which case the 3 traffic sign dots might melt together to form an exclamation mark). <br>
Reducing augmentation at later training stages (using the reduction_coefficient variable), helped SermaNet learn the more subtle differences between a traffic sign and a general caution sign, even with lack of color. It still assigns some probability to the sign being a general caution sign though - and rightly so, since they are visually similar when color is disregarded.
</font>


```python
show_sign_at_index(20, classifications=streetview_classifications, images=streetview_signs)
```


![png](output_76_0.png)


    1: Traffic signals
    2: General caution
    3: Road narrows on the right
    4: Pedestrians
    5: Children crossing



![png](output_76_2.png)



```python
## Diagnostics: We try see how each individual model in the ensemble classified the example, 
# in order to label the problematic model.

with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "./SERMANET.ckpt")
    gray_image = [rgb_to_normalized_gray(streetview_signs[20])]
    
    prediction = sess.run(EB_prediction, feed_dict={EB_x: gray_image, EB_keep_prob:1.})
        
    print("SERMANET prediction:  "+sign_names[str(np.argmax([prediction]))])
        
with tf.Session(graph = g2) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    saver.restore(sess, "./ALEXNET.ckpt")
    
    prediction = sess.run(AN_prediction, feed_dict={AN_x:[streetview_signs[20]], AN_keep_prob: 1.})
    
    print("ALEXNET prediction:  "+sign_names[str(np.argmax([prediction]))])
```

    SERMANET prediction:  Traffic signals
    ALEXNET prediction:  Traffic signals


One would expect the model to perform slightly worse on the new images than those from the set it was trained on, simply because it's an entirely different datasource taken with a different camera, and from entirely different angles (My candidate images were taken from Google StreetView, which means the camera was mounted on top of the car, and we would expect the images to be much more distorted than those from a front-mounted, standard camera).
<br><br>
However, the model performed well and seems to have learned the features of the traffic signs (and not the subtle condition differences), which is great.  I validated this by also using a standard online sign dataset (which contains no distortion and no "outside effects") and still the model performed well, correctly classifying each of the images. This tells us that the model was able to capture the features of the traffic signs and not overfit (too severely) to the data it was trained with.
<br><br>
Overall, the model achieved 18/20 correct predictions on the captured pictures, which translates to 90% accuracy which is slightly lower than the dataset trained on (~96%).

