#Import Modules
import math
import pickle
import random
import numpy as np
from Model_NoDef import DFNet
from keras.utils import np_utils
from keras.optimizers import Adamax
from keras.models import load_model

#Calculate Total Distance between source and target trace
def calc_dist(source,target):
    prev=1
    curr=0
    j=0
    k=0
    total=0   
    flag=0
    
    while(j<len(source) and k<len(target)):

        count1=0
        count2=0

        while(j<len(source)):
                if(source[j]==prev):
                    count1+=1
                elif(source[j]==0):
                    j=len(source)
                else:
                    curr=-prev
                    break
                j+=1

        while(k<len(target)):
                if(target[k]==prev):
                    count2+=1
                elif(target[k]==0):
                    k=len(source)
                    break
                else:
                    break
                k+=1

        prev=curr
        dis=count2-count1
        total+=dis*dis
    
    total=math.sqrt(total)
    return total

# Training the DF model
#NB_EPOCH = 10   # Number of training epoch
NB_EPOCH = 1   # Number of training epoch
BATCH_SIZE = 128 # Batch size
VERBOSE = 2 # Output display mode
LENGTH = 5000 # Packet sequence length
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer
NB_CLASSES = 95 # number of outputs = number of classes
INPUT_SHAPE = (LENGTH,1)


#Create Model
def create_model():
    model = DFNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
    model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,metrics=["accuracy"])
    return model

OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Optimizer
with open('D:\\dataset\\ClosedWorld\\NoDef\\X_test_NoDef.pkl', 'rb') as handle:
	X_test = np.array(pickle.load(handle,encoding="bytes"))
with open('D:\\dataset\\ClosedWorld\\NoDef\\y_test_NoDef.pkl', 'rb') as handle:
	y_test = np.array(pickle.load(handle,encoding="bytes"))
size=X_test.shape[0]
outfile1 = open("D:\\dataset\\GAT\\Adversarial_Traces_x_alpha30_only1.pkl",'wb')
outfile2 = open("D:\\dataset\\GAT\\Adversarial_Traces_y_alpha30_only1.pkl",'wb')
datax=[]
datay=[]
model=create_model()
totalpadding=0
packets=0
model.load_weights('D:\\dataset\\nodef_model_weights_trainer.h5')
for l in range(0,9500):
    
    dummy_x=X_test[l]
    dummy_y=y_test[l]
    source=dummy_x.tolist()
    packetcount=0
    for j in range(0,5000):
        if source[j]==0:
            break
        else:
            packetcount+=1
    for t in range(0,100):
        Xtarget_pool=[]
        ytarget_pool=[]
        i=0
        while(i<100):
            index=random.randrange(size)
            if(index!=l):
                i+=1
                Xtarget_pool.append(X_test[index])
                ytarget_pool.append(y_test[index])

        target_length=len(ytarget_pool)
        
        Xtarget_pool=np.array(Xtarget_pool)
        ytarget_pool=np.array(ytarget_pool)
        

        distmin=1000000000
        closest=0

        j=0

        while(j<len(source)):
            if source[j]==1:
                break
            j+=1
        source=source[j:]

        for i in range(len(source),5000):
                source.append(0)
        dummy_x=source[:]
        
        for o in range(len(Xtarget_pool)):
            target=Xtarget_pool[o].tolist()
            k=0

            while (k<len(target)):
                if target[k]==1:
                    break
                k+=1
            target=target[k:]

            for i in range(len(target),5000):
                target.append(0)
            Xtarget_pool[o]=target[:]
        
        for i in range(0,target_length):
            dist=calc_dist(Xtarget_pool[i],dummy_x)
            if dist<distmin:
                distmin=dist
                closest=i
            
        #print("Source Class",dummy_y,' ',"Target Class",ytarget_pool[closest])
        val=ytarget_pool[closest]
        target=Xtarget_pool[closest].tolist()
        
        Xtarget_pool=Xtarget_pool.astype('float32')
        ytarget_pool=ytarget_pool.astype('float32')
        
        Xtarget_pool=Xtarget_pool[:,:,np.newaxis]
        ytarget_pool = np_utils.to_categorical(ytarget_pool, NB_CLASSES)
        score_test = model.evaluate(Xtarget_pool, ytarget_pool, verbose=VERBOSE)

        answer=source[:]
        original=source[:]

        u=0
        flag=0
        
        #Running 400 Iterations
        for x in range (0,10):
            prev=1
            j=0
            k=0
            curr=0
            total=0
            o=0
            arr=[]
            index=[0]    
        
            while(j<len(source) and k<len(target)):

                count1=0
                count2=0

                while(j<len(source)):
                    if(source[j]==prev):
                        count1+=1
                    elif(source[j]==0):
                        j=len(source)
                    else:
                        curr=-prev
                        break
                    j+=1

                while(k<len(target)):
                    if(target[k]==prev):
                        count2+=1
                    elif(target[k]==0):
                        k=len(target)
                    else:
                        break
                    k+=1

                prev=curr
                dis=count2-count1
                arr.append(30*dis)
                index.append(j)
                o+=1
                total+=dis*dis
            
            total=math.sqrt(total)
            value=0
            added_dis=0
            
            for y in range(len(arr)):
                if(arr[y]>0):
                    for i in range(int(arr[y]/total)):
                        answer.insert(value+index[y+1],1)

                elif(arr[y]<0):
                    for i in range(int(-arr[y]/total)):
                        answer.insert(value+index[y+1],1)
                value+=abs(int(arr[y]/total))
        
            #print("Total Padding ",value)
            answer2=answer[:]
            length=len(answer2)

            for i in range(length,5000):
                answer2.append(0)
            answer2=answer2[:5000]
            answer3=answer2[:]
            answer2=np.array(answer2)
            answer2=np.array([answer2])
            answer2=answer2.astype('float32')
            answer2=answer2[:,:,np.newaxis]
            dis=calc_dist(answer,source)
            if(dis<0.0001 and u>0):
                dis+=1
                u+=1
            else:
                u=0
            if(u==10):
                break
            
            predict=np.argmax(model.predict(answer2), axis=-1)
            prediction=int(predict[0])

            if(val==prediction or dummy_y!=prediction):
                flag=1
                break
            source=answer
        
        if(x==500 or flag==1):
            break
    datax.append(np.array(answer3))
    datay.append(prediction)
    totalpadding=totalpadding+value
    packets=packets+packetcount
    if (l%50==0):
        print("For ",l,"Class is ",value)

print("Total padding ",totalpadding)
print("total packets",packets)
print("percentage",(totalpadding/(totalpadding+packets))*100,"%")
pickle.dump(datax,outfile1)
pickle.dump(datay,outfile2)

outfile1.close()
outfile2.close()

