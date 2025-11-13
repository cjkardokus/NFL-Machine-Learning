# NFL Prediction Project - TODO List

## ✅ Completed
- [x] Set up GitHub repository
- [x] Create project folder structure
- [x] Set up Python virtual environment
- [x] Install required packages (requirements.txt)
- [x] Write data_collection.py script
- [x] Test data collection with 2023-2024 seasons (65 rows generated successfully)

## 🎯 Next Steps (In Order)

### Phase 1: Data Exploration & Feature Engineering (Next Session)
- [ ] Create a Jupyter notebook in `notebooks/` folder to explore the data
  - [ ] Load `data/raw/nfl_team_stats.csv`
  - [ ] Check for missing values
  - [ ] Look at distributions of key features (EPA, yards, turnovers)
  - [ ] Calculate correlations between features and wins
  - [ ] Visualize strongest predictors of wins
  
- [ ] Write `feature_engineering.py` in `src/` folder
  - [ ] Load raw data
  - [ ] Create additional features if needed (e.g., EPA differential = offense - defense)
  - [ ] Normalize features (per-game averages)
  - [ ] Handle any missing values
  - [ ] Save to `data/processed/features.csv`

### Phase 2: Model Training
- [ ] Write `model_training.py` in `src/` folder
  - [ ] Load processed features
  - [ ] Split data (use 2023 for training, 2024 for validation)
  - [ ] Train linear regression model (features → wins)
  - [ ] Evaluate model performance (R², MAE)
  - [ ] Analyze feature importance (coefficients)
  - [ ] Save model to `models/strength_model.pkl`
  - [ ] Save feature importance to `outputs/feature_importance.json`

### Phase 3: Generate Predictions
- [ ] Write `generate_ratings.py` in `src/` folder
  - [ ] Load trained model
  - [ ] Calculate strength rating for each team
  - [ ] Create power rankings (1-32)
  - [ ] Save to `outputs/team_ratings.json`

- [ ] Write `predict_games.py` in `src/` folder
  - [ ] Load team ratings
  - [ ] Fetch upcoming week's schedule
  - [ ] Generate win probabilities for each matchup
  - [ ] Save to `outputs/predictions.json`

### Phase 4: Weekly Update Script
- [ ] Write `weekly_update.py` in `src/` folder
  - [ ] Orchestrate all steps: fetch new data → retrain model → generate predictions
  - [ ] Test full pipeline

### Phase 5: Frontend Development
- [ ] Initialize React app in `frontend/` folder
- [ ] Create components (PowerRankings, GamePredictions, etc.)
- [ ] Set up routing
- [ ] Style with Tailwind CSS
- [ ] Test with sample JSON data

### Phase 6: Deployment
- [ ] Push frontend to GitHub
- [ ] Deploy to Vercel/Netlify
- [ ] Test live site

## 📝 Notes & Ideas

### Data Collection Notes:
- Currently using 2023-2024 data (2 seasons)
- Easy to expand: just change `years_to_collect = [2023, 2024]` in main()
- Consider expanding to 5-10 years once model is working

### Feature Ideas to Explore:
- Point differential (points_scored - points_allowed)
- EPA differential (epa_offense - epa_defense)
- Red zone efficiency (if we can get this data)
- Third down conversion rates

### Questions to Answer in Exploration:
- Which statistics correlate most strongly with wins?
- Is offensive EPA or defensive EPA more predictive?
- Do turnovers matter as much as EPA?
- Any features that are redundant/highly correlated?

## 🐛 Known Issues
- None so far!

## 💡 Future Enhancements (Post-MVP)
- Add more years of historical data
- Include playoff probability predictions
- Track model accuracy over the season
- Add visualizations to blog posts
- Consider ensemble models (Random Forest, XGBoost)
- Add weather data
- Factor in injuries (complex)

---
**Current Status:** Data collection complete ✅  
**Next Session:** Data exploration and feature engineering  
**Last Updated:** November 13, 2024