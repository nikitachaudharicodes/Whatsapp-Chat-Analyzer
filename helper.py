from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import seaborn as sns
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

extract = URLExtract()

def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    #1. fetch number of messages
    num_messages = df.shape[0]
    #2. fetch number of words
    words = []
    for message in df['message']:
        words.extend(message.split())
    #3. fetch number of media messages
    num_media_messages = df[df['message'] == 'sticker omitted'].shape[0]
    #4. fetch number of links shared
    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))


    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    temp = df[df['user'] != 'group_notification']
    temp[temp['message'] != "image omitted"]
    temp[temp['message'] != "sticker omitted"]

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=""))
    return df_wc

def most_common_words(selected_user, df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp[temp['message'] != "image omitted"]
    temp[temp['message'] != "sticker omitted"]

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline
def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap

def sentiment_analysis(selected_user, df):

    if selected_user != 'Overall':
        df=df[df['user'] == selected_user]
    sid_obj = SentimentIntensityAnalyzer()
    d1 = df[['date', 'user', 'message']].copy()
    d1['negative'] = d1['message'].apply(lambda x: sid_obj.polarity_scores(x)['neg'])
    d1['neutral'] = d1['message'].apply(lambda x: sid_obj.polarity_scores(x)['neu'])
    d1['positive'] = d1['message'].apply(lambda x: sid_obj.polarity_scores(x)['pos'])
    d1['compound'] = d1['message'].apply(lambda x: sid_obj.polarity_scores(x)['compound'])

    data = {'Polarity': ['Negative', 'Neutral', 'Positive'],
            'Mean': [d1["negative"].mean(), d1['neutral'].mean(), d1['positive'].mean()]}

    d4 = pd.DataFrame(data)

    return(d1, d4)

def tokenize(selected_user, df):

    if selected_user != 'Overall':
        df=df[df['user'] == selected_user]
    d2 = df[['date', 'user', 'message']].copy()
    d2['tokens'] = d2['message'].apply(lambda x: nltk.word_tokenize(x))

    d8 = d2[['user', 'message', 'tokens']].copy()
    lem = WordNetLemmatizer()
    def lemmatize_text(s):
        s=[lem.lemmatize(word) for word in s]
        return s

    d8['lemmatized'] = d2['tokens'].apply(lambda x: lemmatize_text(x))

    return d2, d8






