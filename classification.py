import pandas as pd
import numpy as np
class naive_bayes:
    def __init__(self):
        pass
    # calculates probability of x_features for every target value and returns probability in the form of dictionary
    def prob(self,col,val):
        prob={}
        denom=0
        for i in self.target_uni:
            r=len(self.data[(self.data[col]==val) & (self.data[self.y_col]==i)])
            denom+=r
            prob[i]=r
        for i in prob.keys():
            prob[i]/=denom
        return prob
    # fit method takes the data in the form of x_train and y_train
    def fit(self,x,y):
        self.data=pd.concat([x,y],axis=1)
        self.x_col=list(x.columns)
        self.y_col=list(y.columns)[0]
        self.target_uni=list(self.data[self.y_col].unique())
    # predict method takes the x_test data and returns the predicted values of target feature in the form of array
    def predict(self,x_test):
        self.y_pred=[]
        y_prob={}
        for i in self.target_uni:
            y_prob[i]=len(self.data[self.data[self.y_col]==i])
        y_prob_denom=sum(list(y_prob.values()))
        for h in range(len(x_test)):
            cols=self.x_col
            col_val=list(x_test.iloc[h,:])
            x_prob={}
            for i,j in zip(cols,col_val):
                x_prob[i]=self.prob(i,j)
            target_prob={}
            for y_val in self.target_uni:
                p=1
                for k in list(x_prob.values()): 
                    p*=k[y_val]
                target_prob[y_val]=p*(y_prob[y_val])/y_prob_denom
            max_prob=0
            val=0
            for i in list(target_prob.keys()):
                if target_prob[i]>max_prob:
                    max_prob=target_prob[i]
                    val=i
            self.y_pred.append(val)
        return np.array(self.y_pred)
    # calculates accuracy and generates confusion matrix,classification report and actual,predicted values of target features
    def score(self,y_test):
        import pandas as pd
        self.actual_pred_table=pd.DataFrame({'Actual':list(y_test[list(y_test.columns)[0]]),'Predicted':self.y_pred})
        t=self.actual_pred_table
        d={}
        TP=len(t[(t['Actual']==1) &(t['Predicted']==1)])
        FP=len(t[(t['Actual']==1) &(t['Predicted']==0)])
        FN=len(t[(t['Actual']==0) &(t['Predicted']==1)])
        TN=len(t[(t['Actual']==0) &(t['Predicted']==0)])
        Accuracy=(TP+TN)/(FP+FN+TP+TN)
        self.confusion_matrix=pd.DataFrame({"Actual_Positive":[TP,FP],"Actual_Negative":[FN,TN]},index=["Predicted_Positive","Predicted_Negative"])
        d['Precision']=[TN/(FN+TN),TP/(TP+FP)]
        d['Recall']=[TN/(TN+FP),TP/(TP+FN)]
        d['F1_Score']=[2*d['Precision'][0]*d['Recall'][0]/(d['Precision'][0]+d['Recall'][0]),2*d['Precision'][1]*d['Recall'][1]/(d['Precision'][1]+d['Recall'][1])]
        d['Support']=[FN+TN,TP+FP]
        self.classification_report=pd.DataFrame(d)
        return f"Accuracy:{Accuracy}"