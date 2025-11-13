"""
NFL Data Collection Script
Fetches historical NFL play-by-play data and aggregates team statistics
"""

import pandas as pd
import nfl_data_py as nfl
from pathlib import Path


def fetch_nfl_data(years):
    """
    Fetch NFL play-by-play data for specified years and aggregate team statistics.
    
    Parameters:
    -----------
    years : list of int
        List of years to fetch data for (e.g., [2023, 2024])
    
    Returns:
    --------
    pandas.DataFrame
        Aggregated team statistics by season
    """
    
    print(f"Fetching play-by-play data for years: {years}")
    
    # Import play-by-play data
    pbp_data = nfl.import_pbp_data(years)
    
    print(f"Loaded {len(pbp_data)} plays")
    print(f"Columns available: {pbp_data.columns.tolist()}")
    
    # Filter out plays where no team possessed the ball (kickoffs, etc.)
    pbp_data = pbp_data[pbp_data['posteam'].notna()]
    
    print("\nAggregating offensive statistics...")
    
    # Aggregate offensive statistics by team and season
    offensive_stats = pbp_data.groupby(['season', 'posteam']).agg({
        'epa': 'mean',                    # Expected Points Added per play
        'yards_gained': 'sum',            # Total yards gained
        'pass_attempt': 'sum',            # Total pass attempts
        'rush_attempt': 'sum',            # Total rush attempts
        'success': 'mean',                # Success rate (EPA > 0)
        'fumble_lost': 'sum',             # Fumbles lost
        'interception': 'sum',            # Interceptions thrown
    }).reset_index()
    
    # Rename columns for clarity
    offensive_stats.columns = [
        'season', 'team', 'epa_per_play_offense', 'total_yards_offense',
        'pass_attempts', 'rush_attempts', 'success_rate_offense',
        'fumbles_lost_offense', 'interceptions_thrown'
    ]
    
    print("Aggregating defensive statistics...")
    
    # Aggregate defensive statistics (when team is on defense)
    defensive_stats = pbp_data.groupby(['season', 'defteam']).agg({
        'epa': 'mean',                    # EPA allowed per play
        'yards_gained': 'sum',            # Total yards allowed
        'success': 'mean',                # Success rate allowed
        'fumble_lost': 'sum',             # Fumbles forced
        'interception': 'sum',            # Interceptions caught
    }).reset_index()
    
    # Rename columns for clarity
    defensive_stats.columns = [
        'season', 'team', 'epa_per_play_defense', 'total_yards_defense',
        'success_rate_defense', 'fumbles_forced', 'interceptions_caught'
    ]
    
    print("Merging offensive and defensive statistics...")
    
    # Merge offensive and defensive stats
    team_stats = pd.merge(
        offensive_stats,
        defensive_stats,
        on=['season', 'team'],
        how='outer'
    )
    
    print("Fetching schedule data for wins/losses...")
    
    # Get schedule data to calculate wins and losses
    schedule = nfl.import_schedules(years)
    
    # Calculate wins for home teams
    home_wins = schedule[schedule['home_score'] > schedule['away_score']].groupby(['season', 'home_team']).size().reset_index(name='home_wins')
    away_wins = schedule[schedule['away_score'] > schedule['home_score']].groupby(['season', 'away_team']).size().reset_index(name='away_wins')
    
    # Calculate losses for home teams
    home_losses = schedule[schedule['home_score'] < schedule['away_score']].groupby(['season', 'home_team']).size().reset_index(name='home_losses')
    away_losses = schedule[schedule['away_score'] < schedule['home_score']].groupby(['season', 'away_team']).size().reset_index(name='away_losses')
    
    # Rename columns for merging
    home_wins.columns = ['season', 'team', 'home_wins']
    away_wins.columns = ['season', 'team', 'away_wins']
    home_losses.columns = ['season', 'team', 'home_losses']
    away_losses.columns = ['season', 'team', 'away_losses']
    
    # Merge wins and losses
    wins_losses = pd.merge(home_wins, away_wins, on=['season', 'team'], how='outer').fillna(0)
    wins_losses = pd.merge(wins_losses, home_losses, on=['season', 'team'], how='outer').fillna(0)
    wins_losses = pd.merge(wins_losses, away_losses, on=['season', 'team'], how='outer').fillna(0)
    
    # Calculate total wins and losses
    wins_losses['wins'] = wins_losses['home_wins'] + wins_losses['away_wins']
    wins_losses['losses'] = wins_losses['home_losses'] + wins_losses['away_losses']
    wins_losses = wins_losses[['season', 'team', 'wins', 'losses']]
    
    print("Merging wins/losses with team statistics...")
    
    # Merge wins/losses with team stats
    team_stats = pd.merge(
        team_stats,
        wins_losses,
        on=['season', 'team'],
        how='left'
    )
    
    # Calculate games played
    team_stats['games_played'] = team_stats['wins'] + team_stats['losses']
    
    # Calculate per-game statistics
    team_stats['yards_per_game_offense'] = team_stats['total_yards_offense'] / team_stats['games_played']
    team_stats['yards_per_game_defense'] = team_stats['total_yards_defense'] / team_stats['games_played']
    
    # Calculate turnovers
    team_stats['turnovers_lost'] = team_stats['fumbles_lost_offense'] + team_stats['interceptions_thrown']
    team_stats['turnovers_gained'] = team_stats['fumbles_forced'] + team_stats['interceptions_caught']
    team_stats['turnover_differential'] = team_stats['turnovers_gained'] - team_stats['turnovers_lost']
    
    print(f"\nData collection complete! Collected stats for {len(team_stats)} team-seasons")
    
    return team_stats


def save_data(data, filename='nfl_team_stats.csv'):
    """
    Save team statistics to CSV file.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Team statistics dataframe
    filename : str
        Name of output file (default: 'nfl_team_stats.csv')
    """
    
    # Create output directory if it doesn't exist
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / filename
    data.to_csv(output_path, index=False)
    
    print(f"\nData saved to {output_path}")
    print(f"Shape: {data.shape}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nColumn names:")
    print(data.columns.tolist())


def main():
    """
    Main function to fetch and save NFL data.
    """
    
    # Specify years to collect data for
    years_to_collect = [2023, 2024]
    
    print("="*60)
    print("NFL Data Collection Script")
    print("="*60)
    
    # Fetch data
    team_stats = fetch_nfl_data(years_to_collect)
    
    # Save data
    save_data(team_stats)
    
    print("\n" + "="*60)
    print("Data collection complete!")
    print("="*60)


if __name__ == "__main__":
    main()