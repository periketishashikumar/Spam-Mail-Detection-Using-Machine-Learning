import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split#splitting the dataset
from sklearn.naive_bayes import MultinomialNB#prediction model
import nltk#natural language processing
import re #cleaning the data for natural language processing
from nltk.stem import WordNetLemmatizer#formation of tokens(lemma)
from nltk.corpus import stopwords#detects stopwords like 'and' 'the' 'an' and soon
from sklearn.impute import SimpleImputer#handling missing values
from sklearn.feature_extraction.text import TfidfVectorizer#encoding categorivcal data
from sklearn.metrics import confusion_matrix,accuracy_score

dataset1=pd.read_csv('True.csv')
dataset2=pd.read_csv('Fake.csv')
dataset1['label']=0#assaigning labels for dataset
dataset2['label']=1 
df1=dataset1[['text','label']]#forming dataset with only required columns
df2=dataset2[['text','label']]
df=pd.concat([df1,df2])#merging two datasets 
df=df.sample(frac=1)#sheffling the dataset after the concatination of teo datasets 

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer and stopwords
lemma = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))

# Define the tokenizer function
def tokenizer(row):
    # Convert to lowercase
    row = row.lower()
    # Remove non-alphanumeric characters
    row = re.sub(r'[^a-zA-Z\s]', ' ', row)
    # Remove stopwords
    row = ' '.join([word for word in row.split() if word not in stopwords_set])
    # Lemmatization
    row = ' '.join([lemma.lemmatize(word) for word in row.split()])
    return row

# Apply the tokenizer function to the 'text' column of your DataFrame
df['text'] = df['text'].apply(tokenizer)

x=df.iloc[:,0]
y=df.iloc[:,-1]
#data splitting
# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorizing the data using the same vectorizer
vt = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
x_train_transformed = vt.fit_transform(x_train)
x_test_transformed = vt.transform(x_test)

# Converting to arrays if needed
x_train_transformed = x_train_transformed.toarray()
x_test_transformed = x_test_transformed.toarray()

# Forming dataframes
x_train = pd.DataFrame(x_train_transformed, columns=vt.get_feature_names_out())
x_test = pd.DataFrame(x_test_transformed, columns=vt.get_feature_names_out())
#MODEL
model=MultinomialNB()
model.fit(x_train,y_train)
pred=model.predict(x_train)
#accuracy
cf=confusion_matrix(y_train,pred)
print(cf)
ac=accuracy_score(y_train,pred)
print(ac)
#predicting for a mail
text=str(input('enter subject of recieved mail : '))
text=tokenizer(text)
x_mail_transformed = vt.transform([text]).toarray()
pred_test = model.predict(x_mail_transformed)
print(f"Predicted label for the mail: {pred_test}")