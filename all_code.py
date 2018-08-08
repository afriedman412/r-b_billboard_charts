# its all your code ... in one place!
import pandas as pd
import billboard
from bs4 import BeautifulSoup
import re
import requests
import unidecode
import string
import datetime
import time
from IPython.display import clear_output
import urllib3
import certifi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# prep 'http'
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

# pull 'all_dates' for 'date_o'
all_dates = pd.read_csv('data_safe/all_dates.csv')

# set to print whole entries
pd.set_option('display.max_colwidth', -1)

#####################
# WORKFLOW AS OF 6/30:
# megachart (uses get_credits, charto, searchy, find_link, allmusic_credits and texty)
# megachart -> bulk_reverse_expander (uses reverse_credit_expander, needs output from group_extract_1to1)
# 'date_o' in there too
#
#####################

### rewriting on 6/7/30 so 'megachart' no longer cares about chart position (because it's in 'mega_track')
### DATA SCRAPING
def megachart(start, delta, end, chart_name='r-b-hip-hop-albums', track=False, starter=None):

    # billboard-approve 'start'
    start_1 = date_o(start, all_dates)

    d1 = datetime.datetime.strptime(start_1, '%Y-%m-%d')
    dd = datetime.timedelta(days=delta)

    # 'mega' collects all data ... prime it with 'starter' or with empty df
    if type(starter) != pd.DataFrame:
        mega = pd.DataFrame(columns=['key', 'artist', 'title', 'credits', 'status'])

    # if 'starter' is a dataframe, put it into 'mega'
    else:
        mega = starter

    # while 'end' is true...
    while eval(end):

        # read d0
        d0 = d1.strftime('%Y-%m-%d')
        print(d0)

        # set 'df' to 'get_credits' of given chart at 'd0' with 'mega' as starter
        df = get_credits(chart_name=chart_name, date=d0, track=track, df=True, starter=mega)
        print(df.shape)

        # add 'chart_date' column to 'df' with all entries equal to d0
        # df['chart_date'] = d0

        # add 'df' to 'mega'
        mega = pd.concat([mega, df])
        d1 += dd
    return mega.reset_index(drop=True)

# given chart and date, return dictionary of 'artist', 'title', 'credits'
# rewriting
def get_credits(chart_name='r-b-hip-hop-albums', date='1984-01-22', track=True, df=True, starter=None):

    # define 'chart' with given 'chart_name' and 'date'
    chart = charto(chart_name, date)

    # initiate 'credits_dict' with empty lists
    credits_dict = {'artist':[],'title':[],'key':[],'credits':[]}

    # pull year
    year = datetime.datetime.strptime(date, '%Y-%m-%d').year

    # if starter, pull keys to prevent duplicates
    if (type(starter) == pd.DataFrame):
        starter_keys = starter['key'].values
    else:
        starter_keys = []

    # iterate through 'chart' entries
    for c in chart:
        if track:
            print(c)

        # pull artist, title and position
        a = c.artist
        t = c.title
        k = ' - '.join([a,t])

        # check for dupes in 'starter'
        if len(starter_keys) > 0:
            if track:
                print('dupe check...')

            if k in starter_keys:
                continue

        credits_dict['artist'].append(a)
        credits_dict['title'].append(t)
        credits_dict['key'].append(k)

        # run search and retrieve
        if track:
            print('searchy...')
        search = searchy(a, t, year)
        if track:
            print('find link...')
        url = find_link(search, a, t, track=track)
        if track:
            print(url)
            print('find credits...')

        # if no url found, declare an error
        if url == None:
            credits_dict['credits'].append('error')
            continue
        credits = allmusic_credits(url)
        credits_dict['credits'].append(credits)
        # if track:
        #   clear_output()
    if df:
        # credits_df = pd.DataFrame(credits_dict)[['key','artist','title','credits']]
        credits_df = pd.DataFrame(credits_dict)
        credits_df['status'] = credits_df['credits'].map(lambda x: 'error' if x == 'error' else 'good')
        return credits_df
    return credits_dict

# takes 'artist' and 'title'
# returns 'title', 'artist', 'year' and 'album_link' for album search for artist and title
def searchy(artist, title, year):
    # process artist and title
    a = texty(artist)
    t = texty(title)

    search = f'{a} {t}'.replace(' ', '+')
    url = f'https://www.allmusic.com/search/albums/{search}'

    r = http.request('GET', url)

    if r.status == 200:
        soup = BeautifulSoup(r.data, 'lxml')
        zults = []
        for d in soup.find_all('div', {'class':'info'}):
            entry = {}
            for tag in ['title','artist','year']:
                try:
                    entry[tag] = d.find('div', {'class':tag}).text.strip()
                except AttributeError:
                    entry[tag] = d.find('div', {'class':tag})
                if tag == 'title':
                    entry['album_link'] = d.find('div', {'class':tag}).find('a')['href']
            zults.append(entry)
        return zults

# searches 'searchy' results for albums that match artist, title and year
# returns url for credits page
def find_link(zults, artist, title, track=True):
    for l in zults:
        if track:
            print(l['artist'], l['title'])

        if (l['artist'] == None) or (l['title'] == None):
            continue

        data = []
        for d in [artist, title, l['artist'], l['title']]:
            data.append(set(texty(d)))

        if track:
            print(data)

        if data[1] == data[3]:
            names = [data[0], data[2]]
            names.sort()
            # print(names)
            if [w for w in names[0] if w not in names[1]] == []:
                url = l['album_link']
                if track:
                    print(url)
                return (f'{url}/credits')


# takes a 'url'
# returns scraped 'credits'
def allmusic_credits(url):

    # define 'r' as http request
    r = http.request('GET', url)

    # if status is 200...
    if r.status == 200:

        # pull site into 'soup'
        soup = BeautifulSoup(r.data, 'lxml')

        # initiate 'credits'
        credits = {'artist':[], 'credit':[]}

        # iterate thru artist 'td', strip whitespace and append
        for t in soup.find_all('td', {'class':'artist'}): 
            credits['artist'].append(t.text.strip())
        
        # iterate thru class 'td', strip whitespace and append
        for t in soup.find_all('td', {'class':'credit'}):
            credits['credit'].append(t.text)

        return credits
    else:
        # print error status
        print(r.status)

# text processing
def texty(input, skip=['and','the', 'featuring']):

    # prep for punctuation removal
    exclude = set(string.punctuation)

    output = (''.join(ch for ch in input if ch not in exclude)).lower().split()
    output = [w for w in output if w not in skip]

    return output


# takes a dict from the scraped data, converts it to a dict of credit/list of artists pairs
# if given a 'credit_index', returns credit groups, not unique credits
def reverse_credit_expander(dicto, credit_index=None, track=False):
    # pivot dicto into d2, dictionary of artist/credit list pairs
    try:
        d2 = dict(zip(dicto['artist'],dicto['credit']))
    except TypeError:
        print('type error')
        return

    # clean credits
    for k, v in d2.items():
        v_clean = v.replace('\n ','').strip().split(',')
        d2[k] = v_clean
        
    d3 = {}
    for k, v in d2.items():

        # if artist has a list of credits ...
        if type(v) == list:
            for l in v:

                # if credit_index is supplied (and is a dict)
                if type(credit_index) == dict:

                    # ignore anything not in credit_index
                    l = l.strip()
                    if l not in credit_index.keys():
                        if track:
                            print(f'skipping: {l}')
                        continue

                    # otherwise, convert credit to credit_group and assing to 'role'
                    role = credit_index[l]
                else:

                    # just strip and set 'role' to credit
                    role = l.strip()

                # either initiate 'role' entry with 'artist' or add 'artist' to 'role' list
                if role not in d3.keys():
                    d3[role] = [k]
                else:
                    if k not in d3[role]:
                        d3[role].append(k)

        # same but if artist only has one credit
        else:
            if type(credit_index) == dict:
                role = credit_index[v]
            else:
                role = v.strip()

            if role not in d3.keys():
                d3[role] = [k]
            else:
                if k not in d3[role]:
                    d3[role].append(k)
    return d3

# takes in scraped chart data, returns a df with features for every credit and row for every album
# joins lists with '|'
# skips any credits in 'skip'
# if provided a credit_index, uses credit groups for features instead of unique credits
# fills any empty cells with 'none' (string -- not np.NaN etc!!!!)
def bulk_reverse_expander(chart, credit_index=None, skip=None, track=False):
    if type(credit_index) != dict:
        print('no credit index provided!')
        return

    df = pd.DataFrame(columns=['key', 'peak', 'year'])
    for n in range(len(chart)):

        a = chart.loc[n,'artist']
        t = chart.loc[n,'title']

        # combines 'artist' and 'title' into single 'key'
        key = ' - '.join([a,t])

        # skips any albums with missing 'credits'
        # adjusted to specifically skip where 'credits' == 'error'
        if chart.loc[n,'credits'] == 'error':
            if track:
                print(f'skipping {key}')
            continue

        else:
            entry = {}
            entry['key'] = key
            entry['peak'] = chart.loc[n,'position']

            # add year
            entry['year'] = datetime.datetime.strptime(chart.loc[n,'chart_date'], '%Y-%m-%d').year

            cred_raw = chart.loc[n,'credits'] if type(chart.loc[n,'credits']) == dict else eval(chart.loc[n,'credits'])

            credits = reverse_credit_expander(chart.loc[n,'credits'], credit_index=credit_index)

            # equalize 'entry' features in 'df'
            # convert lists in credits into strings on the fly
            for k,v in credits.items():
                if type(skip) == list and k in skip:
                    continue

                if k not in df.columns:
                    df[k] = 'none'
                
                # assumes v is a list
                entry[k] = '|'.join(v)

            # equalize 'df' features in 'entry'
            for c in df.columns:
                if c not in entry.keys():
                    entry[c] = 'none'

        df = pd.concat([df, pd.DataFrame(entry, index=[0])])
    return df.reset_index(drop=True)

# extract groups from painstakingly labled google sheet file
# can skip blanks and skips anything in 'skip'
# returns a 1:1 match for easier indexing!
def group_extract_1to1(df, skip_blank=True, skip=['huh']):
    credit_index = {}
    for i,r in df.iterrows():
        group = r['credit group']
        credit = r['credit']
        if type(group) != str:
            if not skip_blank:
                credit_index[credit] = credit
            else:
                continue
        if (credit not in credit_index.keys()) and (group not in skip):
            credit_index[credit] = group
    return credit_index

# adjusts 'd' to be the next date with a billboard chart
def date_o(d, dates):
    for n in dates:
        if n >= d:
            return n

# takes a start date in '&Y-&m-&d' format, 'delta' days, and 'end' end condition
# 'end' should take the format of d1.{unit <>= value}
# supply a chart if need be, but it defaults to the r&b chart
# returns only 'artist','title', 'chart_date' and 'position', keeping all duplicates
# 7/7: baking 'pos_shift' into this
# 7/10: baking 'peak' and 'total_weeks'
def mega_track(start, delta, end, chart_name='r-b-hip-hop-albums', track=False):
    d1 = datetime.datetime.strptime(start, '%Y-%m-%d')
    dd = datetime.timedelta(days=delta)

    mega = pd.DataFrame(columns=['artist', 'title', 'chart_date', 'position'])

    # while 'end' is true...
    while eval(end):

        # read d0
        d0 = d1.strftime('%Y-%m-%d')
        print(d0)

        chart = charto(chart_name, d0)
        c_dict = {'artist':[],'title':[],'position':[]}
        for c in chart:
            c_dict['artist'].append(c.artist)
            c_dict['title'].append(c.title)
            c_dict['position'].append(c.rank)

        df = pd.DataFrame(c_dict)

        # add 'chart_date' column to 'df' with all entries equal to d0
        df['chart_date'] = chart.date

        # add 'df' to 'mega'
        mega = pd.concat([mega, df])
        d1 += dd

    mega2 = pd.DataFrame(columns=['artist', 'title', 'key', 'chart_date', 'position', 'weeks_on_chart'])

    # add 'weeks' on chart
    mega['key'] = mega[['artist','title']].apply(lambda x: ' - '.join(x), axis=1)
    for k in mega['key'].unique():
        df = mega[mega['key'] == k]
        df = df.reset_index(drop=True).reset_index().rename(columns={'index':'weeks_on_chart'})
        df['lag_1'] = df['position'].shift()
        df['lag_2'] = df['position'].shift(2)
        df['next_pos'] = df['position'].shift(-1)
        df.fillna('51', inplace=True)

        # add 'total_weeks' and 'peak' columns
        df['peak'] = df['position'].min()
        df['total_weeks'] = df['weeks_on_chart'].max()

        mega2 = pd.concat([mega2, df])

    # convert columns to numerical if possible
    for c in mega2.columns:
        try:
            mega2[c] = pd.to_numeric(mega2[c])
        except ValueError:
            continue

    # add 'delta' columns and 'move'
    mega2['next_delta'] = mega2['position'] - mega2['next_pos']
    mega2['delta_1'] = mega2['lag_1'] - mega2['position']
    mega2['delta_2'] = mega2['lag_2'] - mega2['lag_1']
    mega2['move'] = mega2['next_delta'].map(lambda x: 0 if x == 0 else 1)
    mega2['move_2'] = mega2['next_delta'].map(lambda x: 'up' if x > 0 else ('stay' if x == 0 else 'down'))


    return mega2.reset_index(drop=True)

# takes a 'mega_track' type df
# returns a df with 'peak_lag', 'weeks_lag' and album info
def album_track(mt, track=False):
    mt_cull = mt[['key','artist','title','peak','total_weeks', 'debut']].drop_duplicates().sort_values(['artist','debut'])
    mt2 = pd.DataFrame(columns=['artist'])
    for a in mt_cull['artist'].unique():
        if track:
            print(a)
        df = mt_cull[mt_cull['artist'] == a]
        df['peak_lag'] = df['peak'].shift()
        df['weeks_lag'] = df['total_weeks'].shift()
        mt2 = pd.concat([mt2, df])
    if track:
        print('adding lags')
    mt2['peak_lag'] = mt2['peak_lag'].fillna(51)
    mt2['weeks_lag'] = mt2['weeks_lag'].fillna(0)
    return mt2

# generates a chart object
def charto(chart='r-b-hip-hop-albums', date='1984-01-22'):
    return billboard.ChartData(chart, date)

# takes a megachart
# returns number of different artists and titles each artist played for
def session_counter(chart):
    exp = bulk_expander(chart)
    artist_count = exp[['artist', 'album_artist']].drop_duplicates().groupby('artist').count()
    title_count = exp[['artist', 'title']].drop_duplicates().groupby('artist').count()
    df = pd.concat([artist_count, title_count['title']], axis=1).reset_index()
    df.rename(columns={'index':'who','album_artist':'artist_count', 'title':'title_count'}, inplace=True)
    return df


# re-adding credit_expander and bulk_expander for counting how many credits an artist has
def credit_expander(dicto):

    # pivot dicto into d2, dictionary of artist/credit list pairs
    try:
        d2 = dict(zip(dicto['artist'],dicto['credit']))
    except TypeError:
        return

    # iterate through d2 pairs
    for k, v in d2.items():

        # took the replace out to fix spacing issue
        d2[k] = v.replace('\n ','').split(',')
    d3 = {'artist':[], 'credit':[]}
    for k, v in d2.items():
        for value in v:
            d3['credit'].append(value)
            d3['artist'].append(k)

    return pd.DataFrame(d3)

def bulk_expander(chart):
    df = pd.DataFrame(columns=['artist','credit', 'album_artist', 'title', 'peak'])
    for n in range(len(chart)):
        if type(chart.loc[n,'credits']) != dict:
            continue
        else:
            a = chart.loc[n,'artist']
            t = chart.loc[n,'title']
            peak = chart.loc[n,'position']
            df2 = credit_expander(chart.loc[n,'credits'])
            df2['album_artist'] = a
            df2['title'] = t
            df2['peak'] = peak
        df = pd.concat([df, df2])
    return df

# takes the name of an 'artist', a 'date', a bulk-expanded 'bulk_chart' and 'mega_track' chart data
# returns stats for that 'artist' at that 'date'
def artist_parz(artist, chart_date, bulk_chart, mega_track):

    # convert date to billboard date
    date_1 = date_o(chart_date, all_dates)

    # check if 'date' is valid
    if date_1 not in mega_track['chart_date'].values:
        print('bad date!')
        return
    else:
        # pull 'artist' data
        df = bulk_chart[bulk_chart['artist'] == artist]

        # pull chart data
        chart = mega_track[mega_track['chart_date'] == date_1]
        chart.rename(columns={'artist':'album_artist'}, inplace=True)

        # combine the two data sets
        df2 = df.merge(chart, on=['title', 'album_artist'])

        # skip if no data
        if df2.shape[0] < 1:
            return
        
        # initiate 'sesh' and add data
        sesh = {}
        sesh['artist'] = artist
        sesh['chart_date'] = date_1
        sesh['avg_weeks_on_chart'] = df2['weeks_on_chart'].mean()
        sesh['avg_position'] = df2['position'].mean()
        sesh['unique_credits'] = int(df2['credit'].nunique())
        sesh['total_credits'] = int(df2.shape[0])
        sesh['total_albums'] = int(df2['title'].nunique())
        return sesh

# returns a data frame of all 'sesh' data for all artists across all dates in 'mega_track'
# if given 'manual_t': 
def bulk_parz(bulk_chart, mega_track, manual_t=None):
    
    # initiate dataframe
    df = pd.DataFrame(columns=[
        'artist', 'chart_date', 'avg_weeks_on_chart', 'avg_position', 'unique_credits', 'total_credits', 'total_albums'])

    if type(manual_t) == list:

        # extract date params from 'manual_t'
        print('using manual_t')
        start = manual_t[0]
        delta = manual_t[1]
        end =   manual_t[2]

        # convert start date to billboard date
        start_1 = date_o(start, all_dates)

        d1 = datetime.datetime.strptime(start_1, '%Y-%m-%d')
        dd = datetime.timedelta(days=delta)

        while eval(end):
            d = d1.strftime('%Y-%m-%d')
            print(d)

            counter = 0
            l = len(bulk_chart['artist'].unique())
            for a in bulk_chart['artist'].unique():
                if counter % 50 == 0:
                    print(f'{counter} of {l}')
                sesh = artist_parz(a, d, bulk_chart, mega_track)
                df = pd.concat([df, pd.DataFrame(sesh, index=[0])])
                if counter % 500 == 0:
                    # clear_output()
                    print(d)
                counter += 1
            d1 += dd


    else:
        print('not using manual_t!')

        # shouldnt need to billboard-approve date because mega_track f'n does it
        for d in mega_track['chart_date'].unique():
            print(d)
            counter = 0
            l = len(bulk_chart['artist'].unique())
            for a in bulk_chart['artist'].unique():
                if counter % 50 == 0:
                    print(f'{counter} of {l}')
                sesh = artist_parz(a, d, bulk_chart, mega_track)
                df = pd.concat([df, pd.DataFrame(sesh, index=[0])])
                if counter % 500 == 0:
                    # clear_output()
                    print(d)
                counter += 1
    return df

# DEPRECIATED
# adds rows to subset of 'mega_track' for when an album leaves the charts
# def off_charter(m_t, key):

#   # subset 'm_t' on album in question
#   df = m_t[m_t['key'] == key].sort_values('chart_date').drop('weeks_on_chart', axis=1)
#   df = df[['key', 'artist', 'title', 'chart_date', 'month', 'position']]

#   # duplicate last row
#   df = pd.concat([df, df.tail(1)])
#   df.reset_index(inplace=True, drop=True)

#   # change 'chart_date' of last row to date of next chart (using 'all_dates')
#   t_index = df.tail(1).index[0]
#   charty = df.tail(1)['chart_date'][t_index]
#   i = list(all_dates).index(charty)
#   df.loc[t_index,'chart_date'] = list(all_dates)[i+1]

#   # change 'position' of last row to 51
#   df.loc[t_index,'position'] = 51

#   # reset index twice, using first index as 'weeks_on_chart'
#   df.reset_index(inplace=True, drop=True)
#   df.reset_index(inplace=True)
#   df.rename(columns={'index':'weeks_on_chart'}, inplace=True)
#   df = df[['key', 'artist', 'title', 'chart_date', 'month', 'position', 'weeks_on_chart']]

#   return df

# combines 'artist' and 'title' of a given df to make 'key'
def keyed(df):
    df['key'] = df[['artist','title']].apply(lambda x: ' - '.join(x), axis=1)
    return df

# adds 'last_pos' and 'next_pos' to each 'key' subset of 'mega_track'
def pos_shifts(m_t):

    # initialize output
    df = pd.DataFrame(columns=m_t.columns)

    for k in m_t['key'].unique():
        print(k)
        subset = m_t[m_t['key'] == k].sort_values('chart_date')
        subset = subset[['key', 'artist', 'title', 'chart_date', 'month', 'year', 'position', 'weeks_on_chart']]

        # add t = -1 and t = 1 columns
        subset['last_pos'] = subset['position'].shift()
        subset['next_pos'] = subset['position'].shift(-1)

        # fill NA with '51' (for off the chart values)
        subset.fillna(51, inplace=True)

        df = pd.concat([df, subset])

    return df

# calculates average personnel scores for an album at a given date from 'parz' and 'mega'
# if 'megachart' 'credits' are str, runs eval to convert to dict
# return a series of means
def scorer(key, date, parz, mega):

    # extract personnel from 'mega'
    i = mega[mega['key'] == key].index[0]
    cr_dict = mega[mega['key'] == key]['credits'][i]
    if (type(cr_dict) == str) and (cr_dict != 'error'):
        cr_dict = eval(cr_dict)
    dicto = cr_dict['artist']

    # pull personnel scores from 'parz'
    scores = parz[(parz['chart_date'] == date) & parz['artist'].isin(dicto)]

    # format scores for output
    scores = scores.set_index('artist').drop('chart_date', axis=1)

    return scores.describe().T['mean']

# adds personnel scores to one key of a 'mega_track'
def super_scorer(key, m_t, parz, mega):

    # subset 'm_t' on album in question
    df = m_t[m_t['key'] == key].sort_values('chart_date')

    scores = pd.DataFrame(columns=['chart_date','avg_weeks_on_chart', 'avg_position', 'unique_credits', 'total_credits', 'total_albums'])

    for d in df['chart_date']:
        s = scorer(key, d, parz, mega)
        s['chart_date'] = d
        scores = pd.concat([scores, pd.DataFrame(s).T])

    return scores

# addes personel scores to entre 'mega_track'
# creates 'pos_diff' feature, difference between avg position and actual position
# adds 'peak_lag' and 'weeks_lag' for the last album by the artist
def mega_scorer(m_t, parz, mega, a_t):
    df= pd.DataFrame(columns=m_t.columns)

    for k in m_t['key'].unique():
        if k not in mega['key'].values:
            print(f'skipping {k}, no credits found')
            continue
        try:
            # print(f'pulling {k}')
            s_s = super_scorer(k, m_t, parz, mega)
        except TypeError:
            print(f'skipping {k}')
            continue
        subset = m_t[m_t['key'] == k].sort_values('chart_date')
        subset_scored = pd.merge(subset, s_s, on='chart_date', how='left')
        df = pd.concat([df, subset_scored])

    df['pos_diff'] = df['position'] - df['avg_position']

    # df = pd.merge(df, a_t[['key', 'peak_lag', 'weeks_lag']])

    return df

def classy(x, y, model):
    y_pred = model.predict(x)
    conf_mat = pd.DataFrame(
        confusion_matrix(y, y_pred),
        index=['is_down','is_stay','is_up'], columns=['pred_down','pred_stay','pred_up'])
    
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(conf_mat.apply(lambda x: x/x.sum(),axis=1), annot=True, cmap='Greens')
    
    print(classification_report(y, y_pred))

    print(pd.DataFrame(y_pred)[0].value_counts())
    return
