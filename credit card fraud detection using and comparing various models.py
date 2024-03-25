import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec #used to specify the geometry of the grid to place a subplot.
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier # Xtreme Gradient Boost
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix

import tensorflow as tf
from tensorflow import keras

import time
from datetime import date
import tkinter
from tkinter import *
from PIL import Image,ImageTk
#loading the data
data=pd.read_csv(r'C:\Users\Msi\Desktop\creditcard.csv')

x=data.drop(['Class'],axis=1)           # print(x.shape)  #(row,column)
y=data['Class']                         # print(y.shape)

x=x.values                           #print(xData)    #numpy array with values only
y=y.values                          #print(yData)

# Splitting into Training & Testing Dataset
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# CNN Implementation-
def CNN():
    # First Layer:
    cnn = tf.keras.Sequential()    # Initiating CNN Classifier Implementation as Sequential Neural Network
    cnn.add(tf.keras.layers.Conv1D(filters=32,kernel_size=2, activation = "relu", input_shape = (x.shape[1],1))) # 1st layer   | activation func introduces NON-LINEARITY to NN, value for above threshold, neuron will be activated else not  | so enables the n/w to learn complex pattern
    cnn.add(tf.keras.layers.Dropout(0.1))     # drop-out prevents over-fitting (randomly removing some neurons)
    # Second Layer:
    cnn.add(tf.keras.layers.BatchNormalization())   #batchnormalization standardizes the each mini-batch as input to next layer, stabilizing the learning process, reducing the number of training epochs required to train deep networks
    cnn.add(tf.keras.layers.Conv1D(filters=64,kernel_size=2, activation = "relu")) # 2nd layer as Convolution layer
    cnn.add(tf.keras.layers.Dropout(0.2))    
    # 3rd layer as flattening layer
    cnn.add(tf.keras.layers.Flatten())    
    cnn.add(tf.keras.layers.Dropout(0.4)) 
    cnn.add(tf.keras.layers.Dense(64, activation = "relu")) 
    cnn.add(tf.keras.layers.Dropout(0.5)) 
    # Last Layer: cnn structure definition ends here
    cnn.add(tf.keras.layers.Dense(1, activation = "sigmoid"))

    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'Precision','Recall','TruePositives','FalsePositives','FalseNegatives','TrueNegatives'])  
    cnn.fit(xtrain, ytrain, epochs = 1, validation_data=(xtest,ytest), verbose=0)   #fitting cnn with scaled data  #CNN uses validation_data(xtest,ytest) is passed as argument in CNN for evaluation its performance in each epoch.  || verbose is for showing the progress of CNN, verbose=0: silent (not to show), 1= show progress of CNN, =2 show progress of CNN by epoch no.
    global loss,accuracy,precision,recall,TP,FP,FN,TN 
    loss,accuracy,precision,recall,TP,FP,FN,TN=cnn.evaluate(x,y)    #'TruePositives','FalsePositives','FalseNegatives','TrueNegatives
    TP,FP,FN,TN =  int(TP),int(FP),int(FN),int(TN)


clf=[RandomForestClassifier(random_state=42),SVC(),KNeighborsClassifier(),DecisionTreeClassifier(),LogisticRegression(),GaussianNB(),AdaBoostClassifier(),XGBClassifier(),CNN()] 
clfname=['Random Forest Classifier:','SVM Classifier:','KNN Classifier:','Decision Tree Classifier:','Logistic Regression Classifier:','Gaussian Naive Bayes Classifier:','AdaBoostClassifier:','XGBoost Classifier:','Convolutional Neural Network (CNN) Classifier:']


def callmodel(i):
    Label(frame2,text=clfname[i],bg="#00ffff",font="Times 18 bold",fg="#ff0000").place(x=10,y=45)   
    if i==8:
        Label(frame2,text=('Accuracy Score=',format(accuracy,'.4f')),bg="#00ffff",font="Times 16 bold",fg="#ff4500").place(x=100,y=200)
        Label(frame2,text=('Precision Score=',format(precision,'.4f')),bg="#00ffff",font="Times 16 bold",fg="#ff4500").place(x=100,y=250)
        Label(frame2,text=('Recall Score=',format(recall,'.4f')),bg="#00ffff",font="Times 16 bold",fg="#0000FF").place(x=100,y=300)
        Label(frame2,text=("F1 Score=",format(2*precision*recall/(precision + recall), '.4f')),bg="#00ffff",font="Times 16 bold",fg="#ff4500").place(x=100,y=350)
        conf_matrix=[[TN,FP],[FN,TP]]
        conf_matrix1=f'{TN}\t\t{FP}\n{FN}\t{TP}'
        Label(frame2,text=('Confusion_matrix:\n',conf_matrix1),bg="#00ffff",font="Times 16 bold",fg="#0000FF").place(x=100,y=80)
        plt.figure(figsize=(6.7,3.2))
        plt.title(f"{clfname[i]} - Confusion Matrix") 
        heatmp=sns.heatmap(conf_matrix, annot = True,fmt='d',cmap='Blues')
        heatmp.get_figure().savefig(r'C:\Users\Msi\Desktop\xyz\conf_matrix.png')
        cm=PhotoImage(file=r'C:\Users\Msi\Desktop\xyz\conf_matrix.png')
        Label(frame1,image=cm).place(x=5,y=5)
        plt.show()        
    else:
        clf[i].fit(xtrain,ytrain)
        ypred=clf[i].predict(xtest)
        
        acc=format(accuracy_score(ytest,ypred),'.4f')
        Label(frame2,text=('Accuracy Score=',acc),bg="#00ffff",font="Times 16 bold",fg="#ff4500").place(x=100,y=170)
        
        Label(frame2,text=('Classification Report:\n',classification_report(ytest,ypred)),bg="#00ffff",font="Times 15 bold",fg="#0000FF").place(x=100,y=300)
        
        prec=format(precision_score(ytest,ypred),'.4f')   
        Label(frame2,text=('Precision Score=',prec),bg="#00ffff",font="Times 16 bold",fg="#ff4500").place(x=100,y=200)
        
        rec=format(recall_score(ytest,ypred),'.4f')
        Label(frame2,text=('Recall Score=',rec),bg="#00ffff",font="Times 16 bold",fg="#0000FF").place(x=100,y=230)
        
        f1=format(f1_score(ytest,ypred),'.4f')
        Label(frame2,text=("F1 Score=",f1),bg="#00ffff",font="Times 16 bold",fg="#ff4500").place(x=100,y=260)
        
        conf_matrix=confusion_matrix(ytest,ypred)    
        Label(frame2,text=('Confusion_matrix:\n',conf_matrix),bg="#00ffff",font="Times 16 bold",fg="#0000FF").place(x=100,y=80)
        plt.figure(figsize=(6.7,3.2))
        plt.title(f"{clfname[i]} - Confusion Matrix") 
        heatmp=sns.heatmap(conf_matrix, annot = True,fmt='d',cmap='Blues')
        heatmp.get_figure().savefig(r'C:\Users\Msi\Desktop\xyz\conf_matrix.png')
        cm=PhotoImage(file=r'C:\Users\Msi\Desktop\xyz\conf_matrix.png')
        Label(frame1,image=cm).place(x=5,y=5)
        plt.show()



root=Tk()
root.title("Credit Card Fraud Detection System using Machine Learning")
root.geometry("1500x800")
photo=PhotoImage(file=r'C:\Users\Msi\Desktop\th.png')
root.iconphoto(True,photo)
root.config()
root.minsize(1510,750)
root.maxsize(1510,750)
root.config(bg='#ffdab9')

def quit():
    root.destroy()

def clock():
        current=time.strftime("%H:%M:%S")
        label1 ["text"]=current
        root.after(1000,clock)


Label(root,text=" Credit Card Fraud Detection System using Machine Learning ",bd=10,highlightthickness=1,font="Times 25 bold",fg="#ff4500",relief=RIDGE).place(x=120,y=10)
Label(root,text=" Mr. Jitendra Kumar ",font="Times 18 bold",fg="#0000ff",relief=SUNKEN).place(x=45,y=80)
Label(root,text=" M.Tech. (ML & DS), TMU, Moradabad ",font="Times 18 bold",fg="#cd853f",relief=SUNKEN).place(x=45,y=110)
Label(root,text="under the guidance of",bg='#ffdab9',font="Times 18 normal",).place(x=470,y=80)
Label(root,text=" Dr. Pankaj Kumar Goswami ",font="Times 18 bold",fg="#0000ff",relief=SUNKEN).place(x=760,y=80)
Label(root,text=" Faculty of Engineering, TMU, Moradabad ",font="Times 18 bold",fg="#cd853f",relief=SUNKEN).place(x=680,y=110)
logo=PhotoImage(file=r'C:\Users\Msi\Desktop\TMU.png')
Label(root,image=logo,bg='#ffdab9').place(x=1150,y=10)

frame1=Frame(root,height="350",width="700",bd=10,bg="#33A9CE",highlightthickness=1,relief=SUNKEN)
frame1.place(x=40,y=150)
frame2=Frame(root,height="560",width="720",bg="#00ffff",bd=10,highlightthickness=1,relief=SUNKEN)
frame2.place(x=760,y=150) #
Label(frame2,text='Performance Evaluation Metrics & Confusion Matrix of Models',font="Times 19 bold underline",bg="#ff4500",fg='#ffffff',relief=RIDGE).place(x=7,y=5)
clf_frame=Frame(root,height="200",width="700",bd=10,bg="#008b8b",highlightthickness=1,relief=SUNKEN)
clf_frame.place(x=40,y=510)

frame_time=Frame(root,height="50",width="150",bd=10,highlightthickness=1,relief=SUNKEN)
frame_time.place(x=1300,y=50)
label1=Label(frame_time,font="article 30",bg="black",fg="#ffd700")
label1.grid(row=2000,column=10)
clock()

Button(clf_frame,text="RFC",bg='#33A9CE',command=lambda:callmodel(0),font="arial 15 bold",bd=10).place(x=20,y=20)
Button(clf_frame,text="SVM",bg='#33A9CE',command=lambda:callmodel(1),font="arial 15 bold",bd=10).place(x=20,y=110)
Button(clf_frame,text="KNN",bg='#33A9CE',command=lambda:callmodel(2),font="arial 15 bold",bd=10).place(x=160,y=20)
Button(clf_frame,text="GNB",bg='#33A9CE',command=lambda:callmodel(5),font="arial 15 bold",bd=10).place (x=160,y=110)
Button(clf_frame,text="DT",bg='#33A9CE',command=lambda:callmodel(3),font="arial 15 bold",bd=10).place(x=310,y=20)
Button(clf_frame,text="LR",bg='#33A9CE',command=lambda:callmodel(4),font="arial 15 bold",bd=10).place(x=310,y=110)
Button(clf_frame,text="ADBT",bg='#33A9CE',command=lambda:callmodel(6),font="arial 15 bold",bd=10).place(x=430,y=20)
Button(clf_frame,text="XGBT",bg='#33A9CE',command=lambda:callmodel(7),font="arial 15 bold",bd=10).place(x=430,y=110)
Button(clf_frame,text="CNN",fg='#ff0000',bg='#33A9CE',command=lambda:callmodel(8),font="arial 15 bold",bd=10).place(x=570,y=20)  

Button(clf_frame,text="Quit",command=quit,font="arial 15 bold",bd=10).place(x=570,y=110)

root.mainloop()