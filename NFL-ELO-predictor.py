
"""
ELO rater for NFL teams

To Add:
    Standings
    Simulate Season
"""
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import pandas as pd
import re
import datetime as dt
# from dateutil.relativedelta import relativedelta
from haversine import haversine
import math
import json
from functools import reduce
# import heapq
import itertools
import jellyfish
import os
import urllib
import seaborn as sns
from matplotlib import pyplot as plt
import time
# =============================================================================
# Directory
# =============================================================================
path = '/Users/graemepm/Dropbox/pgam/NFL/Predict-V2/'
# =============================================================================
# Outstanding
# =============================================================================

# 1) Division changes
# 2) Distance for moved changes
# 3) Expansion teams

# =============================================================================
# Class
# =============================================================================
class FootballSeason:
    
    '''
    Rank NFL teams and predict games
    '''
    
    def __init__(self, year=None, qb_changes=None, season=None): #Move QB changes to load-starting values
        
        self.qb_val_to_elo = 3.3
        self.hist_stats_until = 2019
        self.team_records = {}
        self.wl_records = {}
        self.k = 20 # Change over season
        self.quarterbacks = {}
        self.qb_teams = {}
        self.qb_changes = {}
        self.prediction_results = None
        
        for week in ['w' + str(i) for i in range(1,18)] + ['WildCard', 'Division', 'ConfChamp', 'SuperBowl']:
            self.qb_changes[week] = {}
        if qb_changes:
            self.qb_changes.update(qb_changes)
        
        self.season = season #!!!
        
        if year:
            if isinstance(year, int) and 2000 <= year <= dt.datetime.today().date().year:
                self.year = year
            else:
                raise ValueError('Year must be integer after 2000 and not future year')
        else:
            tday = dt.datetime.today().date()
            year = tday.year
            if tday.month in [1,2]:
                year += 1
            self.year = year
                
        self.__league_info()   
        self.__get_season()#!!!
        self.__load_qb_stats()           
        self.__load_records()
        self.__load_wl_records()
        self.__load_byes()
        self.__calculate_raw_standings()
        self.__load_division_standings()
        self.__load_qb_starters()
        
    def __league_info(self):
        
        '''Load dictionary of NFL teams, team names, city GPS, divisions, conferences'''
        
        with open(path + 'NFL_info.json', 'r') as f:
            info = json.loads(f.read())
        
        self.divisions = {}
        self.conferences = {}
        self.team_names = info['team_names']
        self.league = info['league']
        self.gps = info['gps']
        self.team_colors = info['team_colors']
        self.__info = info
        
        ## Name/acronym lookup
        self.name_lookup = {name: team for team, names in 
                            self.team_names.items() for name in names 
                            if len(name) > 3}
        
        self.acronym_lookup = {name: team for team, names in 
                               self.team_names.items() for name in names 
                               if len(name) <= 3}
            
        ## Calculate distance between cities
        def calculate_distance(team1, team2):
            
            return haversine(self.gps[team1], self.gps[team2], unit='mi')
        
        self.distances = pd.DataFrame(index=self.gps.keys(), 
                                      columns=self.gps.keys())
        
        for team, row in self.distances.iterrows():
            for opp, col in row.items():
                self.distances.at[team, opp] = calculate_distance(team, opp)
                
        ## Load Divisions
        divnames = set([team['div'] for team in self.league.values()])
        for division in divnames:
            teams = []
            for team, spot in self.league.items():
                if division==spot['div']:
                    teams.append(team)
            self.divisions.update({division: teams})
            
        self.team_divisions = {team: division for division, teams in 
                               self.divisions.items() for team in teams}
        
        ## Load Conferences
        for conference in ['NFC', 'AFC']:
            teams = []
            for team, spot in self.league.items():
                if conference==spot['conf']:
                    teams.append(team)
            self.conferences.update({conference: teams})
        
        ## Load week numbers
        weeks = ['w' + str(i) for i in range(18)] + ['WildCard', 'Division', 'ConfChamp', 'SuperBowl']
        self.week_numbers  = {week: i for i, week in enumerate(weeks)}
        
    def __standardize_team(self, team):
        
        '''Find standard team acronym given team name input (from 2000 on)'''
        
        def comp_str(s1, s2):
            S1 = str(s1).upper()
            S2 = str(s2).upper()
            ed = jellyfish.damerau_levenshtein_distance(S1, S2)
        
            return 1 - (ed / max(len(S1), len(S2)))
        
        if team in self.team_names.keys():
            return team
        
        if team in self.acronym_lookup.keys():
            return self.acronym_lookup[team]
        
        name_comp = {name: comp_str(team, name) for name in self.name_lookup.keys()}
        
        if max(name_comp.values()) >= 0.9:
            return self.name_lookup[max(name_comp, key=name_comp.get)]
            
        raise ValueError(team + ' unknwown team name')
        
    def __get_season(self):
        
        '''
        Download NFL season schedule from profootballreference.com
        with live win-loss info for games played through course of season.
        For seasons in years 2000-2018, upload dataset file of NFL records
        '''
        
        start_time = time.time()
        
        def format_schedule(row, year):
            if len(row.Week) <= 2:
                week = 'w' + str(row.Week)
            else:
                week = row.Week
            season = year
            if row.Date in ['January', 'February']:
                date_year = str(int(year) + 1)
            else:
                date_year = str(year)
                
            if pd.isnull(row.Time):
                row.Time = '8:00AM'
                
            date_string = ' '.join([row.Date, date_year, row.Time])
            date_time = dt.datetime.strptime(date_string, '%B %d %Y %I:%M%p')
            date = date_time.date()
            
            if row['Unnamed: 5']=='@':
                away = row['Winner/tie']
                score_away = float(row['PtsW'])
                yds_away = float(row['YdsW'])
                to_away = float(row['TOW'])
                home = row['Loser/tie']
                score_home = float(row['PtsL'])
                yds_home = float(row['YdsL'])
                to_home = float(row['TOL'])
            else:
                home = row['Winner/tie']
                score_home = float(row['PtsW'])
                yds_home = float(row['YdsW'])
                to_home = float(row['TOW'])        
                away = row['Loser/tie']
                score_away = float(row['PtsL'])
                yds_away = float(row['YdsL'])
                to_away = float(row['TOL'])
                
            if score_home == score_away:
                winner = 'TIE'
            elif score_home > score_away:
                winner = home
            elif score_away > score_home:
                winner = away
            else:
                winner = np.NaN
                
            return (season, week, date_time, date, away, home, winner, score_away, 
                    score_home, yds_away, yds_home, to_away, to_home)
       
        if self.year < self.hist_stats_until:
            schedule = pd.read_csv(path + 'hist_records_thru2018.txt', 
                                   sep='\t', parse_dates=['date_time'])

            schedule = schedule[schedule.season.eq(self.year)]
            
        else:
            
            url = os.path.join('https://www.pro-football-reference.com/years', 
                               str(self.year), 'games.htm')
            lines = urllib.request.urlopen(url, timeout = 20).read()
            schedule = pd.read_html(lines)[0]
            schedule = schedule[schedule.Week.astype(str).ne('Week')]
            schedule = schedule.dropna(subset=['Week'])
            schedule = [format_schedule(row, self.year) for i, row in schedule.iterrows()]    
            schedule = pd.DataFrame(schedule)
            schedule.columns = ['season', 'week', 'date_time', 'date', 'away', 'home', 'winner', 
                                'score_away', 'score_home', 'yds_away', 'yds_home', 
                                'to_away', 'to_home']
        
        if type(self.season) is not pd.DataFrame: #!!!
            self.season = schedule
        
        self.week_matches = {}
        for i, row in self.season.iterrows():
            away = self.__standardize_team(row.away)
            home = self.__standardize_team(row.home)
            away_result, home_result = np.nan, np.nan
            if pd.notnull(row.winner):
                if str(row.winner).upper() == 'TIE':
                    away_result, home_result = 0.5, 0.5
                elif row.winner == row.away:
                    away_result, home_result = 1.0, 0.0
                elif row.winner == row.home:
                    away_result, home_result = 0.0, 1.0
                
            if row.week in self.week_matches.keys():
                self.week_matches[row.week].append((away, home, away_result, home_result))
            else:
                self.week_matches[row.week] = [(away, home, away_result, home_result)]
        
        self.team_matches = {}
        for team in self.team_names.keys():
            self.team_matches[team] = []
        for week, games in self.week_matches.items():
            for game in games:
                self.team_matches[game[0]].append((week,) + game)
                self.team_matches[game[1]].append((week,) + game)
                
                
        this_week = dt.datetime(self.year, 1, 1), np.NaN
        for i, row in self.season.iterrows():
            if row.date_time > this_week[0] and pd.notnull(row.winner):
                this_week = row.date_time, row.week
        self.this_week = this_week[1]
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        
        func_msg = 'Loading ' + str(self.year) + ' game stats completed: '
        sec_msg = str(elapsed_time) + ' sec'
        print(func_msg + sec_msg)
                
    def __load_records(self):
        
        '''
        Populate team records with win, loss, tie
        '''
        
        # Load empty records from schedule
        def record_dict():
            
            weeks = ['w' + str(i) for i in range(1,18)]
            weeks += ['WildCard', 'Division', 'ConfChamp', 'SuperBowl']
            week_dict = {}
            for week in weeks:
                week_dict[week] = {'fd': None, 'opp': None, 'result': None,
                         'pf': None, 'pa': None, 'win_prob': None,
                         'elo_pre': None, 'elo_post': None, 'elo_diff': None}
                
            return week_dict
                            
        self.team_records = {}
        for team in self.team_names.keys():
            self.team_records[team] = record_dict()
        
        for i, row in self.season.iterrows():
            
            week = row.week
            home = self.__standardize_team(row.home)
            away = self.__standardize_team(row.away)
            
            self.team_records[home][week]['fd'] = 'home'
            self.team_records[home][week]['opp'] = away
            
            self.team_records[away][week]['fd'] = 'away'
            self.team_records[away][week]['opp'] = home          
                
            if pd.notnull(row.winner):
                
                self.team_records[home][week]['pf'] = row.score_home
                self.team_records[home][week]['pa'] = row.score_away
                
                self.team_records[away][week]['pf'] = row.score_away
                self.team_records[away][week]['pa'] = row.score_home
            
                if row.winner == 'TIE':
                    self.team_records[home][week]['result'] = 'tie'
                    self.team_records[away][week]['result'] = 'tie'
                elif row.winner==row.home:
                    self.team_records[home][week]['result'] = 'win'
                    self.team_records[away][week]['result'] = 'loss'                
                elif row.winner==row.away:
                    self.team_records[home][week]['result'] = 'loss'
                    self.team_records[away][week]['result'] = 'win'
                    
    def get_record_team(self, team, group='overall'):
        
        '''
        Get win-loss record for specific team from FootballSeason class team records
        '''
        
        if group not in ['overall', 'division', 'conference']:
            raise ValueError('Group must be overall, division, or conference')
        
        team = self.__standardize_team(team)
        div = self.league[team]['div']
        conf = self.league[team]['conf']
        
        win, loss, tie = 0, 0, 0
        reg_weeks = ['w' + str(i) for i in range(1,18)]
        
        if group=='overall':
            for week, game in self.team_records[team].items():
                if week in reg_weeks:
                    if game['result']=='win':
                        win += 1
                    elif game['result']=='loss':
                        loss += 1
                    elif game['result']=='tie':
                        tie += 1
        
        if group=='division':
            for week, game in self.team_records[team].items():
                if week in reg_weeks:
                    if game['result']:
                        if self.league[game['opp']]['div'] == div:
                            if game['result']=='win':
                                win += 1
                            elif game['result']=='loss':
                                loss += 1
                            elif game['result']=='tie':
                                tie += 1
                        
        if group=='conference':
            for week, game in self.team_records[team].items():
                if week in reg_weeks:
                    if game['result']:
                        if self.league[game['opp']]['conf'] == conf:
                            if game['result']=='win':
                                win += 1
                            elif game['result']=='loss':
                                loss += 1
                            elif game['result']=='tie':
                                tie += 1
        
        if sum([win, loss, tie]) == 0:
            wp = np.NaN
        else:
            wp = ((win * 1) + (tie * 0.5))/ sum([win, loss, tie])
                
        return win, loss, tie, wp
    
    def __load_wl_records(self):
        
        for team in self.team_records.keys():
            full = self.get_record_team(team)
            div = self.get_record_team(team, 'division')
            conf = self.get_record_team(team, 'conference')
            self.wl_records.update({team: full + div + conf})
            
    def get_records_all(self):
        
        record_df = pd.DataFrame.from_dict(self.wl_records, orient='index')
        record_df.columns = ['win', 'loss', 'tie', 'wp',
                             'div_win', 'div_loss', 'div_tie', 'div_wp',
                             'conf_win', 'conf_loss', 'conf_tie', 'conf_wp']
        
        return record_df
    
    def __load_byes(self):
        
        self.byes = {}
        for week in ['w' + str(i) for i in range(18)]:
            self.byes[week] = []
            if week != 'w0':
                for team, record in self.team_records.items():
                    if record[week]['opp'] is None:
                        self.byes[week].append(team)
                        
        for week in ['WildCard', 'Division', 'ConfChamp', 'SuperBowl']:
            self.byes[week] = []
            if week == 'WildCard':
                for team, record in self.team_records.items():
                    if record['WildCard']['opp'] is None and record['Division']['opp'] is not None:
                        self.byes[week].append(team)

    def load_starting_values(self, 
                             elo_values=None, 
                             qb_vals=None, 
                             qb_vals_team=None,
                             w1_qb_starters=None, 
                             revert=True):
        
        '''
        Load in team ELO rating values and quarterback values to start the season.
        
        elo_values: Dictionary with team acronyms for keys matching 
        FootballSeason.team_names.keys() and ELO rating values
        
        qb_values: Dictionary with quarterback names as keys as they
        appear on profootballreference.com and quarterback value 
        (Total QB yards above replacement)
        
        regress: True or False. Whether to apply regression to the mean
        for values from previous season
        
        qb_starters: Dictionary with keys of teams for week 1 starting QBs
        '''
        
        if elo_values is None:
            for team in self.team_names.keys():
                self.team_records[team]['w1']['elo_pre'] = 1505
                
        else:
            for team, elo in elo_values.items():
                if revert:
                    elo -= (elo - 1505)/3
                team = self.__standardize_team(team)
                self.team_records[team]['w1']['elo_pre'] = elo
                if team in self.byes['w1']:
                    self.team_records[team]['w2']['elo_pre'] = elo
                
        def create_qb_dict():
    
            d = {}
            weeks = [w for w in self.week_numbers.keys() if w != 'w0']
            for week in weeks:
                d[week] = {'val_pre': None, 'val_game': None, 'val_adj': None,
                 'opp': None, 'val_post': None, 'val_change': None}
            
            return d
        
        def create_qbteam_dict():
    
            d = {}
            weeks = [w for w in self.week_numbers.keys() if w != 'w0']
            for week in weeks:
                d[week] = {'valF_pre': None, 'valA_pre': None,
                 'valF_game': None, 'valA_game': None, 
                 'valF_post': None, 'valA_post': None, 'opp': None}
            
            return d
        
        if qb_vals is None:
            self.qb_vals = {}
        else:
            self.qb_vals = {}
            for qb, val in qb_vals.items():
                self.qb_vals[qb] = create_qb_dict()
                self.qb_vals[qb]['w1']['val_pre'] = val
        
        if qb_vals_team is None:
            self.qb_vals_team = {}
            for team in self.team_names.keys():
                self.qb_vals_team[team] = create_qbteam_dict()
                self.qb_vals_team[team]['w1']['valF_pre'] = 50
                self.qb_vals_team[team]['w1']['valA_pre'] = 50
        else:
            self.qb_vals_team = {}
            for team, val in qb_vals_team.items():
                self.qb_vals_team[team] = create_qbteam_dict()
                self.qb_vals_team[team]['w1']['valF_pre'] = val[0]
                self.qb_vals_team[team]['w1']['valA_pre'] = val[1]
        
        if w1_qb_starters:
            self.w1_qb_starters = w1_qb_starters
            
    def get_ending_ELOS(self):
        
        '''Output python dictionary of ending ELOS'''
        
        end_ELOS = {}
        for team, record in self.team_records.items():
            if record['SuperBowl']['elo_post']:
                end_ELOS[team] = record['SuperBowl']['elo_post']
            elif record['ConfChamp']['elo_post']:
                end_ELOS[team] = record['ConfChamp']['elo_post']
            elif record['Division']['elo_post']:
                end_ELOS[team] = record['Division']['elo_post']
            elif record['WildCard']['elo_post']:
                end_ELOS[team] = record['WildCard']['elo_post']
            elif record['w17']['elo_post']:
                end_ELOS[team] = record['w17']['elo_post']
                
        if len(end_ELOS) == 32:
            return end_ELOS
        else:
            return None
            
    def __get_week_from_no(self, number):
        
        '''Return week name from integer number input'''
        
        week = np.NaN
        for w, no in self.week_numbers.items():
            if number == no:
                week = w
        
        return week
        
    def __get_elo_pre_wk(self, team, week):
        
        '''
        Return team's pregame ELO from input week or latest week with pregame ELO
        '''
        
        elos = {w: r['elo_pre'] for w, r in self.team_records[team].items() if
                r['elo_pre'] is not None}
        
        max_elo_week = max(elos, key=self.week_numbers.get)
        
        if self.week_numbers[week] <= self.week_numbers[max_elo_week]:
            elo_pre = self.team_records[team][week]['elo_pre']
            if elo_pre is None: # In case used on team's bye week
                last_week = self.__get_week_from_no(self.week_numbers[week] - 1)
                elo_pre = self.team_records[team][last_week]['elo_pre']
        else:
            elo_pre = self.team_records[team][max_elo_week]['elo_pre']
            
        return elo_pre
    
    def __update_post_elo_team_wk(self, team, week):
        
        '''
        Calculate posterior elo for a team given result
        
        K = 20
        ELO = ELO + (Outcome - Prediction) * K * MOV_factor
        Margin Of Vicotry Factor: 
            ln(AbsolutePointDifferential) * 2.2/(ELODiffOutcomePred * 0.001 + 2.2)
        '''
        
        opp = self.team_records[team][week]['opp']
        elo_pre = self.team_records[team][week]['elo_pre']
        
        prediction = self.predict_game(team, opp, week) #!!!
        result = self.team_records[team][week]['result']
        score = 1 if result=='win' else 0 if result=='loss' else 0.5
        elo_off = score - prediction[0]
        point_diff = self.team_records[team][week]['pf'] - \
            self.team_records[team][week]['pa']
                
        m_o_v = math.log1p(abs(point_diff)) * (2.2/(abs(elo_off)*0.001 + 2.2))
        elo_post = elo_pre + self.k * elo_off * m_o_v
        
        self.team_records[team][week]['win_prob'] = prediction[0]
        self.team_records[team][week]['elo_post'] = elo_post
        self.team_records[team][week]['elo_diff'] = elo_post - elo_pre
        
    def __update_pre_elo_team_next_wk(self, team, week):
        
        '''Enter weeks posterior ELO rating into next week'''
        
        week_no = self.week_numbers[week]
        elo = self.team_records[team][week]['elo_post']
        
        if week == 'SuperBowl':
            pass
            
        else:
            next_week = self.__get_week_from_no(week_no + 1)
            self.team_records[team][next_week]['elo_pre'] = elo
            
            if team in self.byes[next_week]:
                self.team_records[team][next_week]['elo_post'] = elo
                next_next_week = self.__get_week_from_no(week_no + 2)
                self.team_records[team][next_next_week]['elo_pre'] = elo
                
    def update_ELOS(self):
        
        '''Update each team's elo rating week by week'''
        
        start_time = time.time()
        
        this_week = self.this_week
        this_week_no = self.week_numbers[this_week]
        
        weeks_played = [w for w, w_no in self.week_numbers.items() if #!!!
                w_no <= this_week_no and w != 'w0']
        
        for week in weeks_played:
            for team, record in self.team_records.items():
                if record[week]['result'] is not None:
                    self.__update_post_elo_team_wk(team, week)  
                    self.__update_pre_elo_team_next_wk(team, week)
                    
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        
        func_msg = 'Updating ' + str(self.year) + ' ELO rating values completed: '
        sec_msg = str(elapsed_time) + ' sec'
        print(func_msg + sec_msg)
                    
    def __get_home_teams_wk(self, week):
        
        '''Return home teams for a given home week'''
        
        if week == 'SuperBowl':
            return ()
        else:
            return (teams[1] for teams in self.week_matches[week])
    
    def predict_game(self, team1, team2, week):
        
        '''
        Predict regular season game based on ELO ratings
        '''
            
        t1 = self.__standardize_team(team1)
        t2 = self.__standardize_team(team2)
        
        elo1 = self.__get_elo_pre_wk(t1, week)
        elo2 = self.__get_elo_pre_wk(t2, week)
        
        # Bye adjustment
        if week == 'Division':
            if not self.team_records[t1][week]['opp']:
                elo1 += 25
            if not self.team_records[t2][week]['opp']:
                elo2 += 25
                
        else:
            week_no = self.week_numbers[week]
            last_week = self.__get_week_from_no(week_no - 1)
        
            if t1 in self.byes[last_week]:
                elo1 += 25
            if t2 in self.byes[last_week]:
                elo2 += 25
                
        # Distance adjustment
        dist_adj = (self.distances.at[t1, t2]/1000) * 4
            
        # Home adjustment
        if t1 in self.__get_home_teams_wk(week):
            elo1 += 55 + dist_adj
        if t2 in self.__get_home_teams_wk(week):
            elo2 += 55 + dist_adj
            
        # QB adjustment
        qb1 = self.qbs_tostart[week][t1]
        qb2 = self.qbs_tostart[week][t2]
        
        qb_val1 = self.__get_most_recent_qb_val_pre(qb1, week)
        qb_val2 = self.__get_most_recent_qb_val_pre(qb2, week)
        
        elo1 += (qb_val1*3.3)
        elo2 += (qb_val2*3.3)

        p1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        p2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))
        
        return round(p1, 4), round(p2, 4), -round((elo1 - elo2)/25, 1)
    
    def predict_games(self):
        
        self.games_predicted = {}
        for week, games in self.week_matches.items():
            predicted = []
            for game in games:
                prediction = self.predict_game(game[0], game[1], week)
                predicted.append(game[:2] + prediction + game[2:])
            self.games_predicted.update({week: predicted})
            
        self.games_predicted_teams = {}
        for team, games in self.team_matches.items():
            predicted = []
            for game in games:
                prediction = self.predict_game(game[1], game[2], game[0])
                predicted.append(game[:3] + prediction + game[3:])
            self.games_predicted_teams.update({team: predicted})
            
        def game_pick(row):
            wprob = row.away_wprob
            result = row.away_result
            if pd.isnull(result):
                return np.nan
            if result==1:
                if wprob > 0.5:
                    return 1
                else:
                    return 0
            if result==0:
                if wprob > 0.5:
                    return 0
                else:
                    return 1
            if result==0.5:
                if 0.4 < wprob <= 0.6:
                    return 0.5
                else:
                    return 0
        
        PREDICTIONS = pd.DataFrame()
        for week, games in self.games_predicted.items():
            predictions = pd.DataFrame(games)
            predictions = predictions.assign(week = week)
            PREDICTIONS = PREDICTIONS.append(predictions)
            
        PREDICTIONS.columns = ['away', 'home', 'away_wprob', 'home_wprob',
                               'away_spread', 'away_result', 'home_result', 'week']
        
        PREDICTIONS = PREDICTIONS.assign(
                sqfd = (PREDICTIONS.away_wprob - PREDICTIONS.away_result)**2,
                correct = PREDICTIONS.apply(game_pick, axis=1)
                )
        PREDICTIONS.dropna(subset=['away_result'], inplace=True)
        
        brier_score = np.nanmean(PREDICTIONS.sqfd)
        p_correct = np.nanmean(PREDICTIONS.correct)
        
        away = PREDICTIONS[['away', 'sqfd']].copy()
        away.columns = ['team', 'sqfd']
        
        home = PREDICTIONS[['home', 'sqfd']].copy()
        home.columns = ['team', 'sqfd']
        
        team_sqfd = pd.concat([away, home], axis=0, sort=True).dropna()
        team_brier = team_sqfd.groupby('team').mean().sort_values('sqfd', ascending=False)
#        team_brier = team_brier.sqfd.to_dict()
        
        self.brier_score = round(brier_score, 4)
        self.p_correct = round(p_correct, 4) * 100
        self.prediction_results = PREDICTIONS
        self.team_brier_scores = team_brier
        
            
    def __restart_qbs_with_changes(self, week):
        
        if week == 'w1':
            try:
                for team, qb in self.w1_qb_starters.items():
                    self.qbs_tostart[week][team] = qb
            except:
                error = '''
                Unable to load week 1 starting QBs
                from profootballreference.com. 
                Require starter qb dictionary
                '''
                            
                raise ValueError(error)
        
        last_week = self.__get_week_from_no(self.week_numbers[week] - 1)
        for team in self.qbs_tostart[week]:
            if team in self.qb_changes[week]:
                self.qbs_tostart[week][team] = self.qb_changes[week][team]
            else:
                qb = self.qbs_tostart[last_week][team]
                if qb is None:
                    week_before = self.__get_week_from_no(self.week_numbers[last_week] - 1)
                    qb = self.qbs_tostart[week_before][team]
                    self.qbs_tostart[last_week][team] = qb
                    self.qbs_tostart[week][team] = qb
                else:
                    self.qbs_tostart[week][team] = self.qbs_tostart[last_week][team]
    
    def __download_qb_starters(self, week, year=None):
        
        def cleanQB(qb):
            qb = re.sub('\*','', qb)
            qb = re.sub('"','', qb)
            return qb
        
        if year is None:
            year = str(self.year)
        else:
            year = str(year)
        week_no = self.week_numbers[week]
        if week_no <= 17:
            gametype = 'R'
            week_min = str(week_no)
            week_max = str(week_no)
        else:
            gametype = 'P'
            week_min = '0'
            week_max = '99'
        
        head =  'https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1'
        year =  '&match=game&year_min=' + year + '&year_max=' + year
        sea =   '&season_start=1&season_end=-1'
        pos =   '&pos%5B%5D=QB&is_starter=E'
        gt =    '&game_type=' + gametype
        car =   '&career_game_num_min=1&career_game_num_max=400'
        st =    '&qb_start_num_min=1&qb_start_num_max=400'
        gn =    '&game_num_min=0&game_num_max=99'
        wk =    '&week_num_min=' + week_min + '&week_num_max=' + week_max
        qbst =  '&qb_started=Y'
        sort =  '&c5val=1.0&order_by=game_date'
        
        url = ''.join([head, year, sea, pos, gt, car, st, gn, wk, qbst, sort])
        toread = True
        while toread:
            try:
                lines = urllib.request.urlopen(url, timeout = 10).read()
                toread = False
            except:
                time.sleep(5)
                
        starters = pd.read_html(lines)[0]
        starters = starters[starters.Week.astype(str).ne('Week')]
        starters = starters.assign(Player = starters.Player.apply(cleanQB))
        starters.set_index('Tm', inplace=True)
        
        if len(starters) >= 24: # accounts for Byes
            for team, row in starters.iterrows():
                week = self.__get_week_from_no(int(row.Week))
                team = self.__standardize_team(team)
                self.qbs_tostart[week][team] = row.Player
            
            return True
        else:
            return False
            
    def __load_qb_starters(self):
        
        start_time = time.time()
        
        ws = [w for w in self.week_numbers.keys() if w != 'w0']
        self.qbs_tostart = dict.fromkeys(ws)
        for week in self.qbs_tostart.keys():
            self.qbs_tostart[week] = dict.fromkeys(self.team_names.keys())
        
        if self.year < self.hist_stats_until:
            
            STARTERS = pd.read_csv(path + 'qb_starters_thru2018.txt', sep='\t')
            STARTERS = STARTERS[STARTERS.Season.eq(self.year)]
            for i, row in STARTERS.iterrows():
                week = self.__get_week_from_no(int(row.Week))
                team = self.__standardize_team(row.Tm)
                self.qbs_tostart[week][team] = row.Player

        else:

            scrape = True
            for week in self.week_matches.keys():
                
                if scrape:
                    scrape = self.__download_qb_starters(week)
                    if not scrape:
                        self.__restart_qbs_with_changes(week)
                
                else:
                    self.__restart_qbs_with_changes(week)
        
        last_weeks = []         
        for i, (week, teams) in enumerate(self.qbs_tostart.items()):
            last_weeks.append(week)
            if i > 0:
                for team, starter in teams.items():
                    if starter is None:
                        last_week = last_weeks[i-1]
                        self.qbs_tostart[week][team] = self.qbs_tostart[last_week][team]
            
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        
        func_msg = 'Loading ' + str(self.year) + ' starting quarterbacks completed: '
        sec_msg = str(elapsed_time) + ' sec'
        print(func_msg + sec_msg)
            
    def __download_qb_stats_wk(self, week):
        
        '''game type 'E'-regular and playoffs, 'R'-regular, 'P'-playoffs'''
        
        year = str(self.year)
        
        def clean_col(col):
    
            col_text = ' '.join(col)
            col_text = re.sub('Unnamed: [0-9]+_level_[0-9]', '', col_text)
            col_text = col_text.strip()
            
            return col_text
        
        week_no = self.week_numbers[week]
        
        if week_no > 17:
            gametype = 'P'
            week_min = '0'
            week_max = '99'
        else:
            gametype = 'R'
            week_min = str(week_no)
            week_max = str(week_no)
    
        head =  'https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game'
        y1 =    '&year_min=' + str(year)
        y2 =    '&year_max=' + str(year)
        sea =   '&season_start=1&season_end=-1'
        pos1 =  '&pos%5B%5D=QB&pos%5B%5D=WR&pos%5B%5D=RB&pos%5B%5D=TE&pos%5B%5D='
        pos2 =  'OL&pos%5B%5D=DL&pos%5B%5D=LB&pos%5B%5D=DB&is_starter=E'
        gt =    '&game_type=' + gametype
        car =   '&career_game_num_min=1&career_game_num_max=400'
        start = '&qb_start_num_min=1&qb_start_num_max=400'
        game =  '&game_num_min=0&game_num_max=99'
        w1 =    '&week_num_min=' + week_min
        w2 =    '&week_num_max=' + week_max
        
        stats1 = '&c1stat=pass_att&c1comp=gt&c1val=1&c2stat=rush_att&c2comp=gt&c2val=0'
        stats2 = '&c3stat=fumbles&c3comp=gt&c5val=1&c6stat=pass_cmp&order_by=pass_att'
        
        url = ''.join([head,y1,y2,sea,pos1,pos2,gt,car,start,game,w1,w2,stats1,stats2])
        lines = urllib.request.urlopen(url, timeout = 20).read()
        
        stats_df = pd.read_html(lines)[0]
        stats_df.columns = [clean_col(col) for col in stats_df.columns.values]
        stats_df.dropna(subset=['Week'], inplace=True)
        stats_df = stats_df[stats_df.Week.astype(str).ne('Week')]
        stats_df.rename(columns = {'': 'Is_At'}, inplace=True)
        remove_asterisk = lambda x: re.sub('\*', '', x)
        stats_df = stats_df.assign(Player = stats_df.Player.apply(remove_asterisk))
        
        return stats_df
    
    def __load_qb_stats(self):
        
        start_time = time.time()
        
        if self.year < self.hist_stats_until:
            
            QB_STATS = pd.read_csv(path + 'passing_thru2018.txt', 
                                  sep='\t', parse_dates=['Date'])

            min_year = dt.datetime(self.year, 8, 1)
            max_year = dt.datetime(self.year + 1, 3, 1)
            
            QB_STATS = QB_STATS[QB_STATS.Date.gt(min_year)]
            QB_STATS = QB_STATS[QB_STATS.Date.lt(max_year)]
            
        else:
            stat_weeks = [week for week, week_no in self.week_numbers.items() if
                          week_no <= self.week_numbers[self.this_week]]
            
            QB_STATS = pd.DataFrame()
            for week in stat_weeks:
                QB_STATS = QB_STATS.append(self.__download_qb_stats_wk(week))
                
        QB_STATS = QB_STATS.assign(
                Tm = QB_STATS.Tm.apply(self.__standardize_team),
                Opp = QB_STATS.Opp.apply(self.__standardize_team),
                Week = QB_STATS.Week.astype(int)
                )
        
        stat_cols = ['Passing Cmp', 'Passing Att', 'Passing Yds', 'Passing TD', 
                     'Passing Int', 'Passing Sk', 'Rushing Att', 'Rushing Yds', 
                     'Rushing TD', 'Fumbles Fmb']
        
        QB_STATS[stat_cols] = QB_STATS[stat_cols].astype('float64')
        
        self.qb_stats = QB_STATS
        self.active_qbs = list(set(QB_STATS.Player))
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        
        func_msg = 'Loading ' + str(self.year) + ' quarterback stats completed: '
        sec_msg = str(elapsed_time) + ' sec'
        print(func_msg + sec_msg)
        
    def get_qb_stats_raw(self, week, qb=None, for_team=None, against_team=None):
        
        
        '''
        Retreive games stats for quarterback or team-passing stats for given week
        Returns python dictionary of quarterback/passing stats to be used for 
        value calculation implemented through sublcass
        
        '''
        
        stat_cols = ['Passing Cmp', 'Passing Att', 'Passing Yds', 'Passing TD', 
                     'Passing Int', 'Passing Sk', 'Rushing Att', 'Rushing Yds', 
                     'Rushing TD', 'Fumbles Fmb']
        
        week_no = self.week_numbers[week]
        STATS = self.qb_stats.copy()
        STATS = STATS[STATS.Week.eq(week_no)]
        
        if qb:
            STATS = STATS[STATS.Player.eq(qb)]
        elif for_team:
            STATS = STATS[STATS.Tm.eq(for_team)]
        elif against_team:
            STATS = STATS[STATS.Opp.eq(against_team)]
            
        if len(STATS) < 1:
            return None, None
    
        if qb or for_team:
            OPP = list(STATS.Opp)[0]
        else:
            OPP = list(STATS.Tm)[0]
            
        pa = np.nansum(STATS['Passing Att'])
        pc = np.nansum(STATS['Passing Cmp'])
        py = np.nansum(STATS['Passing Yds'])
        pt = np.nansum(STATS['Passing TD'])
        pi = np.nansum(STATS['Passing Int'])
        ps = np.nansum(STATS['Passing Sk'])
        ra = np.nansum(STATS['Rushing Att'])
        ry = np.nansum(STATS['Rushing Yds'])
        rt = np.nansum(STATS['Rushing TD'])
        fm = np.nansum(STATS['Fumbles Fmb'])
        
        stats = {'passing_att': pa, 'passing_cmp': pc, 'passing_yds':py,
                 'passing_td': pt,  'passing_int': pi, 'passing_sk': ps,
                 'rushing_att': ra, 'rushing_yds': ry, 'rushing_td': rt,
                 'fumbles_fmb': fm}
        
        return stats, OPP
    
    def __add_new_qbs(self, week):
        
        def create_qb_dict():
            d = {}
            weeks = [w for w in self.week_numbers.keys() if w != 'w0']
            for week in weeks:
                d[week] = {'val_pre': None, 'val_game': None, 'val_adj': None,
                             'opp': None, 'val_post': None}
            return d
        
        week_no = self.week_numbers[week]
        stats_week = self.qb_stats.copy()
        stats_week = stats_week[stats_week.Week.eq(week_no)]
        stats_week = stats_week[stats_week.Pos.eq('QB')]
        qbs = list(set(stats_week.Player))
        qbs2 = [qb for team, qb in self.qbs_tostart[week].items()]
        qbs = qbs + list(set(qbs2))
        
        for qb in qbs:
            if qb not in self.qb_vals:
                self.qb_vals[qb] = create_qb_dict()
                self.qb_vals[qb][week]['val_pre'] = 30.0
    
    def calculate_qb_val(self, stats):
        '''
        Implement quarterback game value calculation through subclass
        
        Values stored in python dictionary with the following keys:
        
        'passing_att'   'passing_cmp'   'passing_yds'
        'passing_td'    'passing_int'   'passing_sk'
        'rushing_att'   'rushing_yds'   'rushing_td'
        'fumbles_fmb'
        
        Default value returns 50
        '''
        return 50
    
    def __load_raw_qb_vals(self, week):
        
        for qb in self.qb_vals.keys():
            if qb in self.active_qbs:
                stats, opp = self.get_qb_stats_raw(week, qb=qb)
                if stats is not None:
                    val = self.calculate_qb_val(stats)
                    self.qb_vals[qb][week]['val_game'] = val
                    self.qb_vals[qb][week]['opp'] = opp
                
        for team in self.qb_vals_team.keys():
            stats_for, opp = self.get_qb_stats_raw(week, for_team=team)
            stats_against, opp = self.get_qb_stats_raw(week, against_team=team)
            if stats_for is not None:
                val_for = self.calculate_qb_val(stats_for)
                val_against = self.calculate_qb_val(stats_against)
                self.qb_vals_team[team][week]['valF_game'] = val_for
                self.qb_vals_team[team][week]['valA_game'] = val_against
                self.qb_vals_team[team][week]['opp'] = opp
                
    def __calculate_avg_qbteam_val(self, week, F_or_A='for'):
        
        week_no = self.week_numbers[week]
        
        if F_or_A.lower() not in ['for', 'against']:
            raise ValueError('F_or_A must be "for" or "against"')
           
        if F_or_A.lower()=='for':
            valtype = 'valF_pre'
        elif F_or_A.lower()=='against':
            valtype = 'valA_pre'
        
        valsum = 0
        denom = 0
        for team, stats in self.qb_vals_team.items():
            if stats[week][valtype] is not None:
                valsum += stats[week][valtype]
                denom += 1
        
        if denom > 0:
            return valsum/denom
        else:
            return np.nan

    def __update_rolling_qbteam_vals(self, week):
        
        week_no = self.week_numbers[week]
        next_week = self.__get_week_from_no(week_no + 1)
        playoffs = ['Division', 'ConfChamp', 'SuperBowl']
        
        for team, stats in self.qb_vals_team.items():
            
            if stats[week]['valF_game'] is None and week not in playoffs:
                old_val_for = self.__get_most_recent_qbteam_val_pre(team, week, F_or_A='for')
                old_val_ag = self.__get_most_recent_qbteam_val_pre(team, week, F_or_A='against')
                self.qb_vals_team[team][week]['valF_post'] = old_val_for
                self.qb_vals_team[team][week]['valA_post'] = old_val_ag
                self.qb_vals_team[team][next_week]['valF_pre'] = old_val_for
                self.qb_vals_team[team][next_week]['valA_pre'] = old_val_ag
                
            elif stats[week]['valF_game'] is not None:
                old_val_for = self.__get_most_recent_qbteam_val_pre(team, week, F_or_A='for')
                val_for = stats[week]['valF_game']
                new_val_for = (0.9 * old_val_for) + (0.1 * val_for)
                new_val_for = np.clip(new_val_for, 0, 100)
                self.qb_vals_team[team][week]['valF_post'] = new_val_for
                
                old_val_ag = self.__get_most_recent_qbteam_val_pre(team, week, F_or_A='against')
                val_ag = stats[week]['valA_game']
                new_val_ag = (0.9 * old_val_ag) + (0.1 * val_ag)
                new_val_ag = np.clip(new_val_ag, 0, 100)
                self.qb_vals_team[team][week]['valA_post'] = new_val_ag
                
                if week != 'SuperBowl':
                    self.qb_vals_team[team][next_week]['valA_pre'] = new_val_ag
                    self.qb_vals_team[team][next_week]['valF_pre'] = new_val_for
                    
    def __get_most_recent_qb_val_pre(self, qb, week):
        
        stats = self.qb_vals[qb]
        stats_aval = {w: vals for w, vals in stats.items() if 
                      vals['val_pre'] is not None}
        
        most_recent_week = max(stats_aval, key=self.week_numbers.get)
        
        week_no = self.week_numbers[week]
        most_recent_week_no = self.week_numbers[most_recent_week]
        
        if most_recent_week_no < week_no:
            return stats[most_recent_week]['val_pre']
        else:
            return stats[week]['val_pre']
        
    def __get_most_recent_qbteam_val_pre(self, team, week, F_or_A='for'):
        
        if F_or_A not in ['for', 'against']:
            raise ValueError('valtype must be "for" or "against"')
            
        if F_or_A == 'for':
            valtype = 'valF_pre'
        elif F_or_A == 'against':
            valtype = 'valA_pre'
        
        stats = self.qb_vals_team[team]
        stats_aval = {w: vals for w, vals in stats.items() if 
                      vals[valtype] is not None}
        
        most_recent_week = max(stats_aval, key=self.week_numbers.get)
        
        week_no = self.week_numbers[week]
        most_recent_week_no = self.week_numbers[most_recent_week]
        
        if most_recent_week_no < week_no:
            return stats[most_recent_week][valtype]
        else:
            return stats[week][valtype]
                
    def __update_qb_val_post(self, week):
        
        week_no = self.week_numbers[week]
        next_week = self.__get_week_from_no(week_no + 1)  
        playoffs = ['Division', 'ConfChamp', 'SuperBowl']
        
        for qb, stats in self.qb_vals.items():
            if stats[week]['val_game'] is None and week not in playoffs:
                old_val = self.__get_most_recent_qb_val_pre(qb, week)
                self.qb_vals[qb][week]['val_post'] = old_val
                self.qb_vals[qb][week]['val_change'] = 0.0
                if week != 'SuperBowl':
                    self.qb_vals[qb][next_week]['val_pre'] = old_val
            
            elif stats[week]['val_game'] is not None:
                opp = stats[week]['opp']
                opp_valA = self.__get_most_recent_qbteam_val_pre(opp, week, F_or_A='against')
                week_avg = self.__calculate_avg_qbteam_val(week, F_or_A='against')
                above_valA = opp_valA - week_avg
                adj_val = np.clip(stats[week]['val_game'] - above_valA, 0, 100)
                old_val = self.__get_most_recent_qb_val_pre(qb, week)
                new_val = (0.9 * old_val) + (0.1 * adj_val)
                new_val = np.clip(new_val, 0, 100)
                diff = new_val - old_val
                self.qb_vals[qb][week]['val_adj'] = adj_val
                self.qb_vals[qb][week]['val_post'] = new_val
                self.qb_vals[qb][week]['val_change'] = diff
                if week != 'SuperBowl':
                    self.qb_vals[qb][next_week]['val_pre'] = new_val
                
    def update_QB_vals(self):
        
        start_time = time.time()
        
        this_week = self.this_week
        this_week_no = self.week_numbers[this_week]
        
        weeks_played = [w for w, w_no in self.week_numbers.items() if
                        w_no <= this_week_no and w != 'w0']
        
        for week in weeks_played:
            self.__add_new_qbs(week)
            self.__load_raw_qb_vals(week)
            self.__update_rolling_qbteam_vals(week)
            self.__update_qb_val_post(week)
            
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 3)
        
        func_msg = 'Calculating ' + str(self.year) + ' adjusted quarterback values completed: '
        sec_msg = str(elapsed_time) + ' sec'
        print(func_msg + sec_msg)
        
    def get_hth_record(self, teams):
        
        teams = list(teams)
        wins_all = []
        for i, team in enumerate(teams):
            teams[i] = self.__standardize_team(team)
            wins, gp = 0, 0
            for game in self.team_records[team].values():
                if game['opp'] in teams and game['result'] is not None:
                    gp += 1
                    if game['result'] == 'win':
                        wins += 1
                    if game['result'] == 'tie':
                        wins += 0.5
            if gp > 0:
                wins_all.append(wins/gp)
            else:
                wins_all.append(np.NaN)
                    
        return wins_all
    
    def get_hth_winner(self, teams):
        
        teams = list(teams)
        records = self.get_hth_record(teams)
        max_wp  = max(records)
        winners = [team for team, wp in zip(teams, records) if wp >= max_wp]
        if len(winners)==1:
            return winners[0]
        else:
            return np.NaN
    
    def look_up_hth(self, teams):
    
        if tuple(teams) in self.head_to_heads.keys():
            return self.head_to_heads[tuple(teams)]
        else:
            teams = (teams[1], teams[0])
            return self.head_to_heads[teams]
    
    def get_common_record(self, teams):
        
        if len(teams) < 2:
            return [np.nan]
        
        opps_all = []
        for team in teams:
            team = self.__standardize_team(team)
            opps = []
            for game in self.team_records[team].values():
                if game['result']:
                    opps.append(game['opp'])
            opps_all.append(opps)
            
        opps = list(reduce(lambda x, y: set(x) & set(y), opps_all))
        
        wins_all = []
        for team in teams:
            team = self.__standardize_team(team)
            wins, gp = 0, 0
            for game in self.team_records[team].values():
                if game['opp'] in opps and game['result'] is not None:
                    gp +=1
                    if game['result'] == 'win': 
                        wins += 1
                    if game['result'] == 'tie': 
                        wins += 0.5
            if gp > 0:
                wins_all.append(wins/gp)
            else:
                wins_all.append(np.NaN)
            
        return wins_all
    
    def get_strength_victory(self, team):
        
        team = self.__standardize_team(team)
        
        sov, no_opps = 0, 0
        for week, game in self.team_records[team].items():
            if game['result'] == 'win':
                sov += self.get_record_team(game['opp'])[3]
                no_opps += 1
                
        if no_opps < 1:
            return 0.0
        else:
            return sov / no_opps
        
    def get_strength_schedule(self, team):
        
        team = self.__standardize_team(team)
        
        sos, no_opps = 0, 0
        for week, game in self.team_records[team].items():
            if game['result']:
                sos += self.get_record_team(game['opp'])[3]
                no_opps += 1
                
        return sos / no_opps
        
    def __calculate_raw_standings(self):
            
        teams = list(self.team_records.keys())
        standings = pd.DataFrame(teams)
        standings.columns = ['team']
        
        standings = standings.assign(
                wp = standings.team.apply(lambda x: self.get_record_team(x)[3]),
                wp_conf = standings.team.apply(lambda x: self.get_record_team(x, 'conference')[3]),
                wp_div = standings.team.apply(lambda x: self.get_record_team(x, 'division')[3]),
                sov = standings.team.apply(lambda x: self.get_strength_victory(x)),
                sos = standings.team.apply(lambda x: self.get_strength_schedule(x))
                )
        
        standings.set_index('team', inplace=True)
        standings.sort_values(by=['wp', 'wp_div', 'wp_conf', 'sov', 'sos'], 
                              ascending=False, inplace=True)
        
        self.__raw_standings = standings
        self.head_to_heads = {comb: self.get_hth_winner(comb) for comb in
                              itertools.combinations(teams, 2)}
        
    def __load_division_standings(self):
        
        #1) Head-to-head win percent
        #2) Best won-lost-tied percentage in games played within the division.
        #3) Best won-lost-tied percentage in common games.
        #4) Best won-lost-tied percentage in games played within the conference.
        #5) Strength of victory.
        #6) Strength of schedule.
        
        fillna_scalar = lambda x: 0 if pd.isnull(x) else x
        
        all_div_standings = pd.DataFrame()
        for division, teams in self.divisions.items():
            
            div_standings = self.__raw_standings[self.__raw_standings.index.isin(teams)].copy()
            
            div_standings.sort_values('wp', inplace=True, ascending=False)
            for wp, group in itertools.groupby(div_standings.index.tolist(),
                                               key = lambda x: div_standings.at[x, 'wp']):
                current_group = list(group)
                hth_wps = self.get_hth_record(current_group)
                for team, wp_hth in zip(current_group, hth_wps):
                    div_standings.at[team, 'wp_hth'] = wp_hth
            
            div_standings.sort_values(['wp', 'wp_hth', 'wp_div'], inplace=True, ascending=False)
            for wp, group in itertools.groupby(div_standings.index.tolist(), 
                                               key = lambda x: (div_standings.at[x,'wp'],
                                                                fillna_scalar(div_standings.at[x,'wp_hth']),
                                                                div_standings.at[x,'wp_div'])):
                current_group = list(group)
                com_wps = self.get_common_record(current_group)
                for team, wp_com in zip(current_group, com_wps):
                    div_standings.at[team, 'wp_com'] = wp_com
                    
            conf, div = division.split('-')
            div_standings.reset_index(inplace=True)
            div_standings = div_standings.assign(conference = conf, division = div)
            all_div_standings = all_div_standings.append(div_standings, sort=False)
            
            
        all_div_standings.set_index(['conference', 'division'], inplace=True)
        all_div_standings.sort_values(['conference', 'division', 'wp', 'wp_hth', 'wp_div', 
                                       'wp_com', 'wp_conf', 'sov', 'sos'], inplace=True, ascending=False)
            
        self.standings = all_div_standings
        
    def playoff_picture(self):
        
        fillna_scalar = lambda x: 0 if pd.isnull(x) else x
        
        playoffs = {}
        for conf in ['NFC', 'AFC']:
            
            # Division Winners
            winners = []
            for div in ['EAST', 'WEST', 'NORTH', 'SOUTH']:
                df = self.standings.copy()
                df.reset_index(inplace=True)
                df = df[df.conference.eq(conf) & df.division.eq(div)]
                winners.append(df.team.tolist()[0])
                
            df = self.standings.copy()
            df.reset_index(inplace=True)
            df = df[df.conference.eq(conf) & df.team.isin(winners)]
            df.set_index('team', inplace=True)
            
            df.sort_values('wp', inplace=True, ascending=False)
            for wp, group in itertools.groupby(df.index.tolist(), 
                                               key = lambda x: df.at[x, 'wp']):
                current_group = list(group)
                hth_wps = self.get_hth_record(current_group)
                for team, wp_hth in zip(current_group, hth_wps):
                    df.at[team, 'wp_hth'] = wp_hth
                    
            df.sort_values(['wp', 'wp_hth', 'wp_conf'], inplace=True, ascending=False)
            for wp, group in itertools.groupby(df.index.tolist(), 
                                               key = lambda x: (df.at[x, 'wp'],
                                                                fillna_scalar(df.at[x, 'wp_hth']),
                                                                df.at[x, 'wp_conf'])):
                current_group = list(group)
                com_wps = self.get_common_record(current_group)
                for team, wp_com in zip(current_group, com_wps):
                    df.at[team, 'wp_com'] = wp_com     
                    
            df.sort_values(['wp', 'wp_hth', 'wp_conf', 'wp_com', 'sov', 'sos'], 
                           inplace=True, ascending=False)
            
            conf_playoffs = {}
            for place, team in enumerate(df.index.tolist(), start=1):
                conf_playoffs[place] = team
                
            # Wild Cards
            wc = self.standings.copy()
            wc.reset_index(inplace=True)
            wc = wc[wc.conference.eq(conf) & ~wc.team.isin(winners)]
            wc.set_index('team', inplace=True)
            
            wc.sort_values('wp', inplace=True, ascending=False)
            for wp, group in itertools.groupby(wc.index.tolist(), 
                                               key = lambda x: wc.at[x, 'wp']):
                current_group = list(group)
                hth_wps = self.get_hth_record(current_group)
                for team, wp_hth in zip(current_group, hth_wps):
                    wc.at[team, 'wp_hth'] = wp_hth
            
            wc.sort_values(['wp', 'wp_hth', 'wp_conf'], inplace=True, ascending=False)
            for wp, group in itertools.groupby(wc.index.tolist(), 
                                               key = lambda x: (wc.at[x, 'wp'],
                                                                fillna_scalar(wc.at[x, 'wp_hth']),
                                                                wc.at[x, 'wp_conf'])):
                current_group = list(group)
                com_wps = self.get_common_record(current_group)
                for team, wp_com in zip(current_group, com_wps):
                    wc.at[team, 'wp_com_conf'] = wp_com
                    
                    
            wc.sort_values(['wp', 'wp_hth', 'wp_div'], inplace=True, ascending=False)
            for wp, group in itertools.groupby(wc.index.tolist(), 
                                               key = lambda x: (wc.at[x, 'wp'],
                                                                fillna_scalar(wc.at[x, 'wp_hth']),
                                                                wc.at[x, 'wp_div'])):
                current_group = list(group)
                com_wps = self.get_common_record(current_group)
                for team, wp_com in zip(current_group, com_wps):
                    wc.at[team, 'wp_com_div'] = wp_com
                   
                    
            max_wps = wc.wp.nlargest(2)
            winner_1 = wc[wc.wp.eq(max_wps[0])]
            winner_2 = wc[wc.wp.eq(max_wps[1])]
            if len(winner_1) == 1:
                conf_playoffs[5] = winner_1.index[0]
                
                if len(winner_2) == 1:
                    conf_playoffs[6] = winner_2.index[0]
                    
                else:
                    if winner_2.wp_hth.notna().all():
                        max_wp_hth = max(winner_2.wp_hth)
                        winner_2 = winner_2[winner_2.wp_hth.eq(max_wp_hth)]
                        if len(winner_2) == 1:
                            conf_playoffs[6] = winner_2.index.tolist()[0]
                        else:
                            divs = [self.team_divisions[team] for team in winner_2.index.tolist()]
                            if len(set(divs)) == 1:
                                winner_2.sort_values(['wp_div', 'wp_com_div', 'wp_conf', 'sov', 'sos'],
                                                     inplace=True, ascending=False)
                                conf_playoffs[6] = winner_2.index.tolist()[0]
                            else:
                                winner_2.sort_values(['wp_conf', 'wp_com_conf', 'sov', 'sos'],
                                                     inplace=True, ascending=False)
                                conf_playoffs[6] = winner_2.index.tolist()[0]
                    else:
                        divs = [self.team_divisions[team] for team in winner_2.index.tolist()]
                        if len(set(divs)) == 1:
                            winner_2.sort_values(['wp_div', 'wp_com_div', 'wp_conf', 'sov', 'sos'],
                                                 inplace=True, ascending=False)
                            conf_playoffs[6] = winner_2.index.tolist()[0]
                        else:
                            winner_2.sort_values(['wp_conf', 'wp_com_conf', 'sov', 'sos'],
                                                 inplace=True, ascending=False)
                            conf_playoffs[6] = winner_2.index.tolist()[0]
            
            else:
                if winner_1.wp_hth.notna().all():
                    max_wp_hth = max(winner_1.wp_hth)
                    winner_1 = winner_1[winner_1.wp_hth.eq(max_wp_hth)]
                    if len(winner_1) == 1:
                        conf_playoffs[6] = winner_1.index.tolist()[0]
                    else:
                        divs = [self.team_divisions[team] for team in winner_1.index.tolist()]
                        if len(set(divs)) == 1:
                            winner_1.sort_values(['wp_div', 'wp_com_div', 'wp_conf', 'sov', 'sos'],
                                                 inplace=True, ascending=False)
                            conf_playoffs[6] = winner_1.index.tolist()[0]
                        else:
                            winner_1.sort_values(['wp_conf', 'wp_com_conf', 'sov', 'sos'],
                                                 inplace=True, ascending=False)
                            conf_playoffs[6] = winner_1.index.tolist()[0]
                else:
                    divs = [self.team_divisions[team] for team in winner_1.index.tolist()]
                    if len(set(divs)) == 1:
                        winner_1.sort_values(['wp_div', 'wp_com_div', 'wp_conf', 'sov', 'sos'],
                                             inplace=True, ascending=False)
                        conf_playoffs[6] = winner_1.index.tolist()[0]
                    else:
                        winner_1.sort_values(['wp_conf', 'wp_com_conf', 'sov', 'sos'],
                                             inplace=True, ascending=False)
                        conf_playoffs[6] = winner_1.index.tolist()[0]
                        
            playoffs.update({conf: conf_playoffs})
                        
        return playoffs
    
    def __get_most_recent_ELOS(self):
        
        ending_ELOS = {}
        for team, stats in self.team_records.items():
            stats_aval = {week: stat for week, stat in stats.items() if
                          stat['elo_post'] is not None}
            max_week = max(stats_aval, key=self.week_numbers.get)
            elo = stats[max_week]['elo_post']
            ending_ELOS.update({team: elo})
            
        return ending_ELOS
    
    def __get_most_recent_QB_vals_post(self):
        
        ending_QB_vals = {}
        for qb, stats in self.qb_vals.items():
            stats_aval = {w: vals for w, vals in stats.items() if 
                          vals['val_post'] is not None}
            
            max_week = max(stats_aval, key=self.week_numbers.get)
            val = stats[max_week]['val_post']
            ending_QB_vals.update({qb: val})
            
        return ending_QB_vals
    
    def __get_most_recent_QBteam_vals_post(self):
        
        ending_QBteam_vals = {}
        for team, stats in self.qb_vals_team.items():
            stats_aval = {w: vals for w, vals in stats.items() if 
                          vals['valF_post'] is not None}
            
            max_week = max(stats_aval, key=self.week_numbers.get)
            valF = stats[max_week]['valF_post']
            valA = stats[max_week]['valA_post']
            ending_QBteam_vals.update({team: (valF, valA)})
            
        return ending_QBteam_vals
    
    def output_ending_values(self):
        
        '''
        Output python dictionary of ending predictor values
        ELOs, quarterback values, team-passing values
        '''
        
        elos = self.__get_most_recent_ELOS()
        qb_vals = self.__get_most_recent_QB_vals_post()
        qb_vals_team = self.__get_most_recent_QBteam_vals_post()
        
        return elos, qb_vals, qb_vals_team
    
    def ELO_rankings(self):
        
        elos = self.output_ending_values()[0]

        elos = pd.DataFrame.from_dict(elos, orient='index').\
                    sort_values(0, ascending=False,).reset_index()
                    
        elos.index = elos.index + 1
        elos.columns = ['team', 'ELO']
        
        return elos
    
    def qb_val_rankings(self):
        
        qb_vals = self.output_ending_values()[1]
        qb_vals = {qb: val for qb, val in qb_vals.items() if
                   qb in self.active_qbs}
        
        qb_vals = pd.DataFrame.from_dict(qb_vals, orient='index').\
                    sort_values(0, ascending=False,).reset_index()
                    
        qb_vals.index = qb_vals.index + 1
        qb_vals.columns = ['qb', 'value']
        
        return qb_vals
        
    
    def qb_vals_toDataFrame(self):
        
        '''Output pandas dataframe of quarterback values through season'''
        
        QB_VALS_DF = pd.DataFrame()
        year = self.year
        for qb, stats in self.qb_vals.items():
            df = pd.DataFrame.from_dict(stats, orient='index')
            df.dropna(subset=['val_post'], inplace=True)
            df = df.assign(qb = qb, season = year)
            QB_VALS_DF = QB_VALS_DF.append(df)
            
        return QB_VALS_DF
            
    def qbteam_vals_toDataFrame(self):
        
        '''Output pandas dataframe of team-quarterback values through season'''
        
        QBTEAM_VALS_DF = pd.DataFrame()
        year = self.year
        for team, stats in self.qb_vals_team.items():
            df = pd.DataFrame.from_dict(stats, orient='index')
            df.dropna(subset=['valF_post'], inplace=True)
            df = df.assign(team = team, season = year)
            QBTEAM_VALS_DF = QBTEAM_VALS_DF.append(df)
            
        return QBTEAM_VALS_DF
            
    def ELOS_toDataFrame(self):
        
        '''Output pandas dataframe of ELO values through season'''
        
        ELOS_DF = pd.DataFrame()
        year = self.year
        for team, stats in self.team_records.items():
            df = pd.DataFrame.from_dict(stats, orient='index')
            df = df.assign(team = team, season = year)
            ELOS_DF = ELOS_DF.append(df)
            
        return ELOS_DF
    
    def plot_ELOS(self, team):
        
        team = self.__standardize_team(team)
        
        elos = {week: stats['elo_pre'] for week, stats in 
                self.team_records[team].items() if
                stats['elo_pre'] is not None}
        
        elos = pd.DataFrame.from_dict(elos, orient='index').reset_index()
        elos.columns = ['week', 'ELO']
        
#        week_order = [w for w in week_order if w in elos.week.tolist()]
#        week_order.sort(key=self.week_numers.get)
        elos = elos.assign(
#                week = elos.week.astype('category').cat.reorder_categories(week_order),
                week_no = elos.week.apply(self.week_numbers.get)
                )
        
        letters = 'abcdefghijklmnopqrstuv'
        alpha = {i: a for i, a in zip(range(1,23), letters)}
        elos = elos.assign(
                week_alpha = elos.week_no.apply(alpha.get)
                )
        
        if self.prediction_results is not None:
            sqfd = self.prediction_results.copy()
            sqfd = sqfd[sqfd.away.eq(team) | sqfd.home.eq(team)]
            sqfd = sqfd[['week', 'sqfd']]
            elos = elos.merge(sqfd, how='left', on='week')
            
        else:
            elos = elos.assign(sqfd = 0.0)
        
        sns.set_style('whitegrid')
        sns.set_context('talk')
        
#        plot = sns.catplot(x=elos.week, 
#                           y=elos.ELO,
#                           color=self.team_colors.get(team),
#                           kind='point'
#                           )
                           
        plot = sns.lineplot(elos.week_alpha, elos.ELO,
                            color=self.team_colors.get(team)
                            )
        
        plot.scatter(elos.week_alpha, elos.ELO, 
                     color=self.team_colors.get(team),
                     s=(elos.sqfd*50)**2
                     )
        
        labels = elos.week.tolist()
        labels.sort(key=self.week_numbers.get)
        plot.set_title(self.team_names[team][2] + ' ELO rating by week')
        plot.set_xticklabels(labels)
        plot.set_xlabel('')
        
        for item in plot.get_xticklabels():
            item.set_rotation(90)

        
        return plot
    
    def plot_ELOS_all(self, save=None):
        
        DF = pd.DataFrame()
        for team, record in self.team_records.items():
            df = pd.DataFrame.from_dict(record, orient='index')
            df = df.assign(team = team)
            df.dropna(subset=['elo_pre'], inplace=True)
            DF = DF.append(df)
            
        
        DF.reset_index(inplace=True)
        DF.rename(columns ={'index':'week'}, inplace=True)
        wk_order = [w for w in self.week_numbers.keys() if w in set(DF.week)]
        DF = DF.assign(week = DF.week.astype('category').cat.reorder_categories(wk_order))
        
        get_conf = lambda x: self.team_divisions[x].split('-')[0]
        get_div = lambda x: self.team_divisions[x].split('-')[1]
        DF = DF.assign(
                division = DF.team.apply(get_div),
                conference = DF.team.apply(get_conf)
                )
                
        plot = sns.catplot(x='week', y='elo_pre', hue='team',
                           palette=self.team_colors.values(),
                           col='division', row='conference',
                           data=DF, kind='point', markers='.', s=0.5)
        
        plot.set_titles('{row_name} {col_name}')
        plot.set_axis_labels('', 'ELO')
        plot._legend.set_title('Team')
        plt.gcf().subplots_adjust(bottom=0.1) 
              
        for ax in plot.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)
            
        if save is not None:
            plt.savefig('/Users/graemepm/Desktop/' + save + '.jpeg', dpi=900)
            
        return plot
        
    def plot_QB_val(self, qb):
        
        vals = {week: (stats['val_pre'], stats['val_adj']) for week, stats in 
                self.qb_vals[qb].items() if stats['val_adj'] is not None}


        vals = pd.DataFrame.from_dict(vals, orient='index').reset_index()
        vals.columns = ['week', 'Value_Pre', 'Value_Game']
        
        vals = vals.assign(
                week_no = vals.week.apply(self.week_numbers.get)
                )
        
        letters = 'abcdefghijklmnopqrstuv'
        alpha = {i: a for i, a in zip(range(1,23), letters)}
        vals = vals.assign(
                week_alpha = vals.week_no.apply(alpha.get)
                )
        
        sns.set_style('whitegrid')
        sns.set_context('talk')
                           
        plot = sns.lineplot(vals.week_alpha, vals.Value_Pre, color='black')
        
        plot.scatter(vals.week_alpha, vals.Value_Game, 
                     c=vals.Value_Game, cmap='bwr')
        
        
        labels = vals.week.tolist()
        labels.sort(key=self.week_numbers.get)
        plot.set_title(qb + ' QB value by week')
        plot.set_xticklabels(labels)
        plot.set_xlabel('')
        plot.set_ylabel('')
        plot.set_ylim(-5,105)
        
        for item in plot.get_xticklabels():
            item.set_rotation(90)
        
        return plot
