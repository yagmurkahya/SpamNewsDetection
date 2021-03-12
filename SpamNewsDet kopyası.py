import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


dataframe = pd.read_excel("turknews.xlsx")
dataframe.head()
labels=dataframe.label
labels.head()
x = dataframe['title']
y = dataframe['label']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
y_train


tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)
with open('tfid.pickle','wb') as f:
    pickle.dump(tfvect,f)

from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)

y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)


fake_news_det("Süleyman Soylu: Elimde öyle istihbaratlar var ki")



