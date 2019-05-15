from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel

import pandas as pd

def parse_raw_message(raw_message):
    try:
        lines = raw_message.split('\n')
        email = {}
        message = ''
        keys_to_extract = ['from', 'to']
        for line in lines:
            if ':' not in line:
                message += line.strip()
                email['body'] = message
            else:
                pairs = line.split(':')
                key = pairs[0].lower()
                val = pairs[1].strip()
                if key in keys_to_extract:
                    email[key] = val
    except:
        email = {}
    return email


def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results


def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }


def read_email_bodies():
  emails = pd.read_csv('splitemails.csv')
  email_df = pd.DataFrame(parse_into_emails(emails.message))
  email_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)
  email_df.drop_duplicates(inplace=True)
  return email_df['body']


class EmailDataset: 
  def __init__(self):
    stopwords = ENGLISH_STOP_WORDS.union(['ect', 'hou', 'com', 'recipient'])
    self.vec = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_df=0.3, min_df=2)
    self.emails = read_email_bodies() 

    # train on the given email data.
    self.train()
  
  def train(self):
    self.vec_train = self.vec.fit_transform(self.emails)
  
  def query(self, keyword, limit):
    vec_keyword = self.vec.transform([keyword])
    cosine_sim = linear_kernel(vec_keyword, self.vec_train).flatten()
    related_email_indices = cosine_sim.argsort()[:-limit:-1]
    print(related_email_indices)
    return related_email_indices

  def find_email_by_index(self, i):
    return self.emails.as_matrix()[i]


if __name__ == "__main__":
    ds = EmailDataset()
    results = ds.query('network', 100)
    # print out the first result.
    print(ds.find_email_by_index(results[0]))