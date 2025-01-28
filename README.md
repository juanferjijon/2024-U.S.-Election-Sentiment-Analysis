# 2024-U.S.-Election-Sentiment-Analysis
Welcome to the 2024 U.S. Election Sentiment Analysis repository! This project aims to analyze public sentiment on the 2024 U.S. Election using tweets from the social media platform X. The analysis is based on three datasets: train.csv, test.csv, and val.csv, which contain tweets related to various candidates and political parties.
Dataset: https://www.kaggle.com/datasets/emirhanai/2024-u-s-election-sentiment-on-x?select=val.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
val_df = pd.read_csv('val.csv')

# Combine the datasets for a comprehensive analysis
df = pd.concat([train_df, test_df, val_df])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# Plot Overall Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Overall Sentiment Distribution')
plt.show()

# Plot Candidate Sentiment Analysis
plt.figure(figsize=(12, 8))
sns.countplot(x='candidate', hue='sentiment', data=df)
plt.title('Candidate Sentiment Analysis')
plt.show()

# Plot Party Sentiment Analysis
plt.figure(figsize=(12, 8))
sns.countplot(x='party', hue='sentiment', data=df)
plt.title('Party Sentiment Analysis')
plt.show()

# Plot Most Popular Tweets
most_liked = df.nlargest(5, 'likes')
most_retweeted = df.nlargest(5, 'retweets')
print("Most Liked Tweets:\n", most_liked[['user_handle', 'tweet_text', 'likes']])
print("Most Retweeted Tweets:\n", most_retweeted[['user_handle', 'tweet_text', 'retweets']])

# Plot Common Keywords
positive_tweets = ' '.join(df[df['sentiment'] == 'positive']['tweet_text'])
neutral_tweets = ' '.join(df[df['sentiment'] == 'neutral']['tweet_text'])
negative_tweets = ' '.join(df[df['sentiment'] == 'negative']['tweet_text'])

wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
wordcloud_neutral = WordCloud(width=800, height=400, background_color='white').generate(neutral_tweets)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_tweets)

plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Common Keywords in Positive Tweets')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.title('Common Keywords in Neutral Tweets')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Common Keywords in Negative Tweets')
plt.axis('off')

plt.show()

# Plot Engagement Analysis
plt.figure(figsize=(12, 8))
sns.scatterplot(x='retweets', y='likes', hue='sentiment', data=df)
plt.title('Engagement Analysis: Retweets vs Likes')
plt.show()

# Plot Candidate Comparison
candidate_comparison = df.groupby('candidate')[['retweets', 'likes']].mean()
print("Candidate Comparison:\n", candidate_comparison)

# Plot Sentiment Over Time for Each Candidate with Color Coding
plt.figure(figsize=(14, 8))
candidates = df['candidate'].unique()
colors = sns.color_palette("hsv", len(candidates))

for candidate, color in zip(candidates, colors):
    candidate_data = df[df['candidate'] == candidate]
    sentiment_over_time = candidate_data.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    sentiment_over_time.plot(kind='line', color=color, ax=plt.gca(), label=candidate)

plt.title('Sentiment Over Time for Each Candidate')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.legend(title='Candidate')
plt.show()
