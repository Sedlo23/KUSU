import os
from tkinter import Button, Tk, Canvas
from tkinter.messagebox import askyesno

from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import joblib
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# specify the path to the directory containing the dataset
path = "final_symbols_split_ttv/train"

# specify the size of the images in pixels
image_size = 50

# define a function to load the dataset and preprocess the images
def load_dataset():
    # create empty arrays for the images and labels
    images = []
    labels = []

    # loop over the directories in the dataset
    for dir_name in os.listdir(path):
        if not os.path.isdir(f"{path}/{dir_name}"):
            continue

        # loop over the files in each directory
        for filename in os.listdir(f"{path}/{dir_name}"):
            # load the image file and resize it to the desired size
            image = Image.open(f"{path}/{dir_name}/{filename}")
            image = image.convert("L")  # convert to grayscale
            image = image.resize((image_size, image_size))

            # add the image and label to the arrays
            images.append(np.array(image).flatten())
            labels.append(dir_name)

    # convert the arrays to numpy arrays
    images = np.array(images)
    labels = np.array(labels)



    # split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# define a function to train an SVM model on the training set and test it on the testing set
def train_svm(X_train, X_test, y_train, y_test,setTrain):

    if(setTrain):
        # create an SVM classifier with a linear kernel
        clf = svm.SVC(kernel='linear',probability=True, verbose=10)

        # train the classifier on the training set
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'svm_model.joblib')
        # test the classifier on the testing set
        y_pred = clf.predict(X_test)

        # calculate the accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        plt.imshow(confusion, cmap=plt.cm.Blues,
                   extent=[-0.5, len(os.listdir(path)) - 0.5, len(os.listdir(path)) - 0.5, -0.5])
        plt.colorbar()
        tick_marks = np.arange(len(os.listdir(path)))
        plt.xticks(tick_marks, os.listdir(path), rotation=90)
        plt.yticks(tick_marks, os.listdir(path))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for i in range(len(os.listdir(path))):
            for j in range(len(os.listdir(path))):
                plt.text(j, i, confusion[i, j], ha='center', va='center')
        plt.show()

    else:
        # load the model from the file
        clf = joblib.load('svm_model.joblib')
        accuracy = ""
        confusion = ""

    return clf, accuracy, confusion

setTrain = askyesno(title='Učení',message='Přeješ si znovu učit model?')

# load the dataset and preprocess the images
X_train, X_test, y_train, y_test = load_dataset()

# train an SVM model on the training set and test it on the testing set
clf, accuracy, confusion = train_svm(X_train, X_test, y_train, y_test,setTrain)


# create a window with a canvas to draw on
window = Tk()
canvas = Canvas(window, width=400, height=400, bg="gray")
canvas.pack()

# create an empty image and a draw object
image2 = Image.new("RGB", (400, 400), (255, 255, 255))
draw = ImageDraw.Draw(image2)

# define a function to draw lines on the canvas and on the image
def draw_line(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x+15, y+15, fill="black")
    draw.line((x, y, x+15, y+15), fill=(0, 0, 0), width=20)

# bind the canvas to the draw_line function
canvas.bind("<B1-Motion>", draw_line)


# define a function to save the image as a JPEG file
def save_image():

    image = image2.convert("L") # convert to grayscale
    image = image.resize((image_size, image_size))
    X_new = np.array(image).flatten()
    probs = clf.predict_proba([X_new])
    probs = probs[0]

    # create a bar chart of the probabilities
    fig, ax = plt.subplots()
    x_pos = np.arange(len(clf.classes_))
    ax.bar(x_pos, probs, align='center', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(clf.classes_)
    ax.set_ylabel('Probability')
    ax.set_title('Predicted probabilities')

    # The OffsetBox is a simple container artist.
    # The child artists are meant to be drawn at a relative position to its #parent.
    imagebox = OffsetImage(image.resize((200, 200)), zoom=0.15)
    # Annotation box for solar pv logo
    # Container for the imagebox referring to a specific position *xy*.

    ind = np.argmax(probs)
    ab = AnnotationBbox(imagebox, (ind, probs[ind]/2), frameon=False)
    ax.add_artist(ab)

    # display the chart
    plt.show()

    w, h = 600, 600
    shape = [(0, 0), (w , h )]
    draw.rectangle(shape, fill ="#FFF")
    canvas.delete('all')

if __name__ == '__main__':
    button = Button(window, text="Calc prob", command=save_image)
    button.pack()

    window.mainloop()


