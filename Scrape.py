import pandas as pd
from bs4 import BeautifulSoup
from pip._vendor import requests

# url and year for the stats we want
url = 'https://www.pro-football-reference.com'
max_players = 310
max_year = 2019
starting_year = 2018


def scrape_top_players(start, end):
    for year in range(start, end):
        overall_rank = 1
        print(year)
        r = requests.get(url + '/years/' + str(year) + '/fantasy.htm')
        soup = BeautifulSoup(r.content, 'html.parser')
        parsed_table = soup.find_all('table')[0]

        df = pd.DataFrame([],
                          columns=['Name', 'Team', 'Pos', 'Age', 'Games Played', 'Games Started', 'Pro Bowl', 'All Pro',
                                   'Completions', 'Passing Attempts', 'Passing Yards', 'Passing TD',
                                   'Interceptions',
                                   'Rushing Attempts', 'Rushing Yards', 'Rushing Y/A', 'Rushing TD',
                                   'Targets', 'Receptions', 'Receiving Yards', 'Y/R', 'Receiving TD',
                                   'Fumbles', 'Fumbles Lost', 'Total TD', '2PM', '2PP',
                                   'Fantasy Points', 'PPR', 'DraftKings Points', 'FanDuel Points',
                                   'VBD', 'Position Rank', 'Overall Rank'])

        for i, row in enumerate(parsed_table.find_all('tr')[2:]):
            #print(year, ' ', i)

            if i == max_players / 2:
                print(i)

            if i >= max_players:
                print('\nComplete.')
                break

            try:
                # Get all the text values from the row and format them a little bit
                row_as_text = [i.text for i in row]
                row_as_text.insert(7, 0)
                row_as_text.insert(7, 0)

                if row_as_text[1][len(row_as_text[1]) - 1] == '+':
                    row_as_text[1] = row_as_text[1][:len(row_as_text[1]) - 1]
                    row_as_text[8] = 1

                if row_as_text[1][len(row_as_text[1]) - 1] == '*':
                    row_as_text[1] = row_as_text[1][:len(row_as_text[1]) - 1]
                    row_as_text[7] = 1

                for index in range(len(row_as_text)):
                    if row_as_text[index] == '':
                        row_as_text[index] = 0

                row_as_text = row_as_text[1:]

                row_as_text[len(row_as_text) - 1] = overall_rank
                overall_rank = overall_rank + 1

                # Add them to the DataFrame
                df.loc[len(df)] = row_as_text

            except Exception as e:
                pass

        df.to_csv('fantasy-full' + str(year) + '.csv')


scrape_top_players(2010, 2019)


def scrape_per_game():
    for year in range(starting_year, max_year):
        print(year)

        r = requests.get(url + '/years/' + str(year) + '/fantasy.htm')
        soup = BeautifulSoup(r.content, 'html.parser')
        parsed_table = soup.find_all('table')[0]

        df = []

        # first two tables are bad
        for i, row in enumerate(parsed_table.find_all('tr')[2:]):
            print(year, ' ', i)
            # if i % 10 == 0:
            #    print(i)
            # if i % 100 == 0:
            #    print(i)
            if i >= max_players:
                print('\nComplete.')
                break

            try:
                # from the whole row of their name, get the <td> thing that has
                # data-stat="player", because this has the player name and link stub which we can
                # use to get to their fantasy page
                dat = row.find('td', attrs={'data-stat': 'player'})

                # get the name from the row
                name = dat.a.get_text()

                # gets the stub link then adds /fantasy/ and the year which
                # sends it to the players fantasy stats for that year
                stub = dat.a.get('href') + '/fantasy/' + str(year)

                # find the <td> thing with data-stat="fantasy_pos" and assign it to their position
                pos = row.find('td', attrs={'data-stat': 'fantasy_pos'}).get_text()

                # grab this players stats
                tdf = pd.read_html(url + stub)[0]

                # get rid of MultiIndex, just keep last row
                tdf.columns = tdf.columns.get_level_values(-1)

                # fix the away/home column
                tdf = tdf.rename(columns={'Unnamed: 4_level_2': 'Away'})
                tdf['Away'] = [1 if r == '@' else 0 for r in tdf['Away']]

                # keep columns wanted
                tdf = tdf.iloc[:, [1, 2, 3, 4, 5, -3]]

                # drop "Total" row
                tdf = tdf.query('Date != "Total"')

                # add other info
                # tdf['Name'] = name
                # tdf['Position'] = pos
                tdf.insert(1, 'Name', name)
                tdf.insert(2, 'Position', pos)
                tdf['Season'] = year

                df.append(tdf)
            except:
                pass

        df = pd.concat(df)
        df.head()

        df.to_csv('fantasy' + str(year) + '.csv')
