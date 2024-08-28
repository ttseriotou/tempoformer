import pandas as pd
from os import listdir

##################################
##########Data reading############
##################################

#source: https://github.com/Konigari/Mixed-Initiative/tree/master/Switchboard-Corpus/Annotation%20Data/Dataset
data_dir = '/import/nlp/tt003/topic-shift-mi/data/topic-shift/'
files = [f for f in listdir(data_dir)]

df_list = []
for i, f in enumerate(files):
    if i!=42:
        df_part = pd.read_csv(data_dir+f, encoding='windows-1252')
        df_list.append(df_part)

df = pd.DataFrame(columns=['Unnamed: 0', 'timeline_id', 'Person', 'Conversation', 'Class'])
for i in range(len(df_list)):
    df_list[i]['timeline_id'] = str(i)
    df = pd.concat([df, df_list[i][['Unnamed: 0', 'timeline_id', 'Person', 'Conversation', 'Class']]])

#rename columns
df = df.rename(columns={'Unnamed: 0': 'utterance_id', 'Conversation': 'text', 'Class':'label'})

#remove nulls
df = df[~df['text'].isnull()]
df = df[~df['Person'].isnull()]

#label type conversion
df['label'] = df['label'].astype('int')
df.loc[df['label']==2,'label']=1

#reset index
df = df.reset_index(drop=True)

text_col = 'text'
label_col = 'label'
timeline_col = 'timeline_id'