import numpy as np
from PIL import Image
import sys
from keras.models import load_model
#360 824
"""

This program takes the LFW test set
folder as the input, and returns the 
number of correct predictions,
total number of predictions,
the accuracy percentage,
and the confusion matrix for the 
LFW test set

This program also creates a file called
predictions_lfw.txt, which consists of 
the image name followed by the model's 
top three predictions.

"""

print "Loading trained CNN model"
model = load_model('my_model_360_iter_batch_100.h5')
print "Loading Complete"
import os
files = os.listdir(sys.argv[1])

tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

def targets(filename):
    targets = []
    for f in filename:
        if tag_list[0] in f:
            targets.append(0)
        if tag_list[1] in f:
            targets.append(1)
        if tag_list[2] in f:
            targets.append(2)
        if tag_list[3] in f:
            targets.append(3)
        if tag_list[4] in f:
            targets.append(4)
        if tag_list[5] in f:
            targets.append(5)
        if tag_list[6] in f:
            targets.append(6)
    return np.array(targets)

true_labels = targets(files)

#for i in range(0, 7):
    #print np.sum(true_labels == i)
predictions = []

#print files
writer = open('predictions_lfw.txt', 'w')
for f in files:
    #print f
    current = Image.open(sys.argv[1] + f)
    #current
    current = current.convert('L')
    current = current.resize((48, 48))
    data = np.array(current.getdata())
#10 on top
#11 on top
#13 okay
#15 okay
    data = np.reshape(data, (1, 48, 48, 1))
#from keras.models import load_model
#360 82.4
#model = load_model('my_model_360_iter_batch_100.h5')

    prediction = model.predict(data)
    order = np.argsort(prediction)[0,:]
    #print order
    #print prediction
#first = np.argmax(prediction)
#m keras.models import load_model
#360 82.4
#model = load_model('my_model_360_iter_batch_100.h5')


#prediction[0, first] = 0
#second = np.argmax(prediction)

    tag_dict = {0: 'Angry', 1: 'Sadness', 2: 'Surprise', 3: 'Happiness', 4: 'Disgust', 5: 'Fear', 6: 'Neutral'}
 
    prediction = prediction[0, :]
    writer.write(f + '  ' + tag_dict[order[-1]]+ '  ' + str( prediction[order[-1]]) + '  ' + tag_dict[order[-2]] + '  ' + str(prediction[order[-2]]) + '  ' +  tag_dict[order[-3]] + '  ' +str( prediction[order[-3]]))
    writer.write('\n')
    predictions.append([order[-1], order[-2], order[-3]])
  
#print predictions
writer.close()
true_predictions = []
count = 0
tot_count = 0
for i in range(0, len(true_labels)):
    tot_count +=1
    if true_labels[i] in predictions[i]:
       count += 1
       true_predictions.append(true_labels[i])
    else:
       true_predictions.append(predictions[i][2])

print "Number of Correct Predictions: " +str(count)
print "Total number of Images: " +str(tot_count)
#print count, tot_count
accuracy =  1.0 * count / tot_count
print "Accuracy: "  +str(accuracy)
from sklearn.metrics import confusion_matrix
print "\nConfusion Matrix\n"
print confusion_matrix(true_labels, true_predictions)
