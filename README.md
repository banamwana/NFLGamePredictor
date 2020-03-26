# NFLGamePredictor

ELO rating system and game predictor reverse engineered from FiveThiryEight model https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/

Reads game and player stats from https://www.pro-football-reference.com/

Allows custom algorithm for calculation of quarterback game value.

Sample use:

```
qb_changes = {'w2': {'JAX': 'Gardner Minshew', 'NYJ': 'Trevor Siemian'},
              'w3': {'PIT': 'Mason Rudolph', 'NO': 'Teddy Bridgewater', 
                     'NYJ': 'Luke Falk', 'NYG': 'Daniel Jones', 'MIA': 'Josh Rosen',
                     'CAR': 'Kyle Allen'},
              'w5': {'CHI': 'Chase Daniel'},
              'w6': {'NYJ': 'Sam Darnold'},
              'w8': {'KC': 'Matt Moore'},
              'w12': {'CHI': 'Chase Daniel'}
              }
              
with open(path + 'file.json', 'r') as f: 
    starter_stats = json.loads(f.read())
    
class FootballSeasonQB(FootballSeason):
    
    def calculate_qb_val(self, stats):
        
        '''
        'passing_att'   'passing_cmp'   'passing_yds'
        'passing_td'    'passing_int'   'passing_sk'
        'rushing_att'   'rushing_yds'   'rushing_td'
        'fumbles_fmb'
    
        '''
        
        pa = stats.get('passing_att')
        pc = stats.get('passing_cmp')
        py = stats.get('passing_yds')
        pt = stats.get('passing_td')
        pi = stats.get('passing_int')
        ps = stats.get('passing_sk')
        ra = stats.get('rushing_att')
        ry = stats.get('rushing_yds')
        rt = stats.get('rushing_td')
        fm = stats.get('fumbles_fmb')
        
        value = (-2.2*pa) + (3.7*pc) + (py/5) + (11.3*pt) - (14.1*pi) - \
        (8*ps) - (1.1*ra) + (0.6*ry) + (15.9*rt) - (14.1*fm)
        
        return value
        

    
              
this_time = FootballSeasonQB(2019, qb_changes=qb_changes)

this_time.load_starting_values(
        elo_values=starter_stats['elos'], 
        qb_vals=starter_stats['qb_vals'], 
        qb_vals_team=starter_stats['qb_vals_team'],
        w1_qb_starters=None, 
        revert=True
        )
                               
this_time.update_QB_vals()
this_time.update_ELOS()
this_time.predict_games()

this_time.games_predicted['w16']
this_time.games_predicted_teams['MIN']
this_time.qb_vals['Mitchell Trubisky']

this_time.get_hth_record(['GB', 'DAL'])

this_time.plot_ELOS('MIN')
this_time.plot_ELOS_all(save='image')

this_time.brier_score
this_time.p_correct
this_time.team_brier_scores

this_time.ELO_rankings()
this_time.qb_val_rankings()
```
