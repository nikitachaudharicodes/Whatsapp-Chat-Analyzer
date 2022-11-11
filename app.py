import streamlit as st
import seaborn as sns
import preprocessor
import helper
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.title("IN DEPTH ANALYSIS OF SELECTED TEXTS :D")


    st.header("Table of texts with deets:")
    st.dataframe(df)

    #fetch unique users
    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        df88=df.copy()

        if selected_user != 'Overall':
            df88 = df[df['user'] == selected_user]
            st.title("User's texts:")
            st.dataframe(df88)

        #STATS AREA

        num_messages, words, num_media_messages,num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")

        col1, col2, col3, col4=st.columns(4) #st.beta_colums ---> st.columns


        with col1:
            st.header("Total Messages:")
            st.title(num_messages)
        with col2:
            st.header("Total Words:")
            st.title(words)
        with col3:
            st.header("Media Messages:")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared:")
            st.title(num_links)

        #TIMELINE
            # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # DAILY TIMELINE
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ACTIVITY MAP
        st.title('Activity Map')
        col1, col2 = st.beta_columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        #FINDING BUSIEST USERS IN GROUP (grp level)

        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        #TOKENIZE, STEMMING, LEMMATIZE


        #WORDCLOUD

        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        plt.imshow(df_wc)
        st.pyplot(fig)

        #MOST COMMON WORDS

        most_common_df = helper.most_common_words(selected_user, df)

        fig,ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common Words: ")
        st.pyplot(fig)

        #EMOJI ANALYSIS

        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.beta_columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            if len(emoji_df)!=0:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1], labels=emoji_df[0])
                st.pyplot(fig)

        # TOKENIZE
        st.title("Tokenize Texts!")

        d6, d8 = helper.tokenize(selected_user, df)
        st.dataframe(d6)

        st.title("Lemmatization")

        st.dataframe(d8)

        #SENTIMENT ANALYSIS

        st.title("Sentiment Analysis!")


        d2,d4=helper.sentiment_analysis(selected_user, df)
        st.dataframe(d2)

        st.dataframe(d4)

        mean = d4["Mean"].tolist()
        polarity = d4["Polarity"].tolist()

        fig, ax = plt.subplots()
        ax.pie(mean, labels=polarity)
        st.pyplot(fig)

        st.title("Person/People is/are overall:")

        if d2["compound"].mean() >= 0.05:
            st.header("Positive!!")

        elif d2["compound"].mean() <= - 0.05:
            st.header("Negative!!")

        else:
            st.header("Neutral...")







