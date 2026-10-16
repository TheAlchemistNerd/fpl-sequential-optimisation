# Chapter 3: Data Acquisition, Feature Engineering, and Exploratory Data Analysis

## 3.1 The Data Foundation for Sequential Optimisation

The quality of any sequential optimisation system is fundamentally bounded by the quality, richness, and timeliness of its data. In the Fantasy Premier League context, data arrives in multiple modalities: structured tabular statistics from the official FPL API, advanced performance metrics from third-party analytics providers, unstructured textual data from news articles, and relational data encoding fixtures, opponents, and team affiliations. This chapter provides a comprehensive account of the data acquisition pipeline, the feature engineering methodology, and the exploratory data analysis (EDA) techniques that underpin all subsequent modelling.

The data pipeline must satisfy three requirements:

1. **Timeliness**: Data must be fetched and processed within the narrow window between the conclusion of a gameweek's matches and the subsequent gameweek's transfer deadline (typically 18–36 hours).
2. **Completeness**: All relevant features for every active player must be collected, with missing data handled gracefully.
3. **Reproducibility**: All data must be versioned and linked to the specific code and model versions that consumed it, enabling rigorous experimentation and debugging.

## 3.2 FPL API Data Acquisition

The official Fantasy Premier League API is the primary data source, providing ground-truth information on players, teams, fixtures, gameweek results, and managerial histories. The API is RESTful, returning JSON payloads. Each endpoint is documented along with its mathematical role in the system.

### 3.2.1 Bootstrap Static Endpoint

**URL**: `https://fantasy.premierleague.com/api/bootstrap-static/`

This is the master endpoint, returning a single JSON object containing three top-level keys: `elements` (all registered FPL players), `teams` (all 20 Premier League clubs), and `events` (global gameweek summaries).

**Player Data (`elements`)**: For each player $p$, the endpoint returns a feature vector:

$$x^{\text{FPL}}_p = (\text{id}_p, \text{first\_name}_p, \text{second\_name}_p, \text{team}_p, \text{element\_type}_p, \text{now\_cost}_p, \text{total\_points}_p, \text{points\_per\_game}_p, \text{selected\_by\_percent}_p, \text{form}_p, \text{ict\_index}_p, \text{minutes}_p, \text{goals\_scored}_p, \text{assists}_p, \text{clean\_sheets}_p, \text{goals\_conceded}_p, \text{own\_goals}_p, \text{penalties\_saved}_p, \text{penalties\_missed}_p, \text{yellow\_cards}_p, \text{red\_cards}_p, \text{saves}_p, \text{bonus}_p, \text{bps}_p, \text{influence}_p, \text{creativity}_p, \text{threat}_p, \text{starts}_p, \text{expected\_goals}_p, \text{expected\_assists}_p, \text{expected\_goal\_involvements}_p, \text{expected\_goals\_conceded}_p, \text{influence\_rank}_p, \text{creativity\_rank}_p, \text{threat\_rank}_p, \text{corners\_and\_indirect\_freekicks\_order}_p, \text{direct\_freekicks\_order}_p, \text{penalties\_order}_p, \text{status}_p).$$

This vector is the raw material for feature engineering. Key derived quantities include:

- **Per-90 normalisation**: For any cumulative statistic $S_p$ (e.g., goals, xG, shots), the per-90 rate is:

  $$S^{/90}_p = \frac{S_p}{\max(\text{minutes}_p, 1)} \cdot 90.$$

  This controls for playing time, enabling fair comparison between starters and substitutes.

- **Form metric**: The `form` field is a rolling average of points over the most recent gameweeks (typically 3–5). This is recomputed with custom windows:

  $$\text{Form}_p(w) = \frac{1}{w} \sum_{k=0}^{w-1} \text{points}_p(t - k), \quad w \in \{3, 5, 10\}.$$

- **ICT Index decomposition**: The ICT Index is a weighted composite:

  $$\text{ICT}_p = \text{Influence}_p + \text{Creativity}_p + \text{Threat}_p.$$

  Each component is treated as a separate feature, as they capture distinct aspects of performance.

**Team Data (`teams`)**: For each team $c$, the endpoint returns:

$$x^{\text{team}}_c = (\text{id}_c, \text{name}_c, \text{strength}_c, \text{strength\_overall\_home}_c, \text{strength\_overall\_away}_c, \text{strength\_attack\_home}_c, \text{strength\_attack\_away}_c, \text{strength\_defence\_home}_c, \text{strength\_defence\_away}_c, \text{position}_c, \text{played}_c, \text{win}_c, \text{loss}_c, \text{draw}_c, \text{points}_c).$$

The strength ratings (1–5 scale) are used to construct opponent difficulty features.

**Event Data (`events`)**: For each gameweek $t$, the endpoint returns:

$$x^{\text{event}}_t = (\text{id}_t, \text{name}_t, \text{deadline\_time}_t, \text{average\_entry\_score}_t, \text{highest\_score}_t, \text{is\_current}_t, \text{is\_next}_t, \text{is\_previous}_t, \text{chip\_plays}_t),$$

where `chip_plays` records the number of managers who deployed each chip. This provides population-level benchmarks.

### 3.2.2 Fixtures Endpoint

**URL**: `https://fantasy.premierleague.com/api/fixtures/`

Returns all 380 Premier League matches for the season:

$$\text{Fixture}_f = (f, \text{team\_h}_f, \text{team\_a}_f, \text{kickoff\_time}_f, \text{event}_f, \text{team\_h\_score}_f, \text{team\_a\_score}_f, \text{finished}_f).$$

From this, the **fixture difficulty** for player $p$ at gameweek $t$ is derived:

$$\text{FDR}_p(t) = \begin{cases} \text{strength\_defence\_away}(\text{opponent}_p(t)) & \text{if } p \text{ is home}, \\ \text{strength\_defence\_home}(\text{opponent}_p(t)) & \text{if } p \text{ is away}, \end{cases}$$

where $\text{opponent}_p(t)$ is the team that player $p$'s team faces at gameweek $t$.

A **rolling opponent difficulty** feature is also constructed:

$$\text{RollingFDR}_p(t, w) = \frac{1}{w} \sum_{k=0}^{w-1} \text{FDR}_p(t - k).$$

This captures whether a player has faced a run of difficult or favourable fixtures, which may confound form assessment.

### 3.2.3 Element Summary Endpoint

**URL**: `https://fantasy.premierleague.com/api/element-summary/{EID}/`

Returns detailed historical data for player with ID `EID`, including:

- `history`: gameweek-by-gameweek performance (points, minutes, goals, assists, etc.).
- `history_past`: season-by-season career summary.
- `fixtures`: remaining fixtures for the current season.

The gameweek history provides the longitudinal data required for temporal modelling:

$$\text{History}_{p,t} = (\text{element\_score}_p(t), \text{minutes}_p(t), \text{total\_points}_p(t), \text{was\_home}_p(t), \text{opponent\_team}_p(t), \text{round}_p(t), \text{kickoff\_time}_p(t), \text{team\_h\_score}_p(t), \text{team\_a\_score}_p(t), \text{expected\_goals}_p(t), \text{expected\_assists}_p(t), \text{expected\_goal\_involvements}_p(t), \text{expected\_goals\_conceded}_p(t), \text{value}_p(t), \text{transfers\_balance}_p(t), \text{selected}_p(t), \text{transfers\_in}_p(t), \text{transfers\_out}_p(t)).$$

### 3.2.4 Event Live Endpoint

**URL**: `https://fantasy.premierleague.com/api/event/{GW}/live/`

Returns live data for all players during gameweek `GW`. This is used for in-gameweek updates and real-time monitoring:

$$\text{Live}_{p, t} = (\text{stats}_p(t), \text{explain}_p(t), \text{in\_dreamteam}_p(t), \text{special}_p(t), \text{multiplier}_p(t)).$$

### 3.2.5 Entry History Endpoint

**URL**: `https://fantasy.premierleague.com/api/entry/{TID}/history/`

Returns the gameweek-by-gameweek history of a specific FPL manager (team ID `TID`):

$$\text{ManagerHistory}_t = (\text{event}_t, \text{points}_t, \text{totalPoints}_t, \text{rank}_t, \text{rankSort}_t, \text{overallRank}_t, \text{bank}_t, \text{value}_t, \text{eventTransfers}_t, \text{eventTransfersCost}_t, \text{activeChip}_t).$$

This is used for benchmarking the agent's performance against human managers and for studying human decision patterns.

## 3.3 External Data Sources

### 3.3.1 Understat: Advanced Performance Metrics

Understat (https://understat.com/) provides advanced metrics not available through the FPL API, including Expected Goals (xG), Expected Assists (xA), non-Penalty xG (npxG), xGChain, and xGBuildup. These metrics quantify the quality of scoring chances and creative actions, providing a more stable and predictive signal than raw goal/assist counts.

The data is obtained via asynchronous web scraping. The Understat pages embed JSON data within `<script>` tags, which are extracted using regex pattern matching:

```python
UNDERSTAT_PATTERN = r"{}\s+=\s+JSON.parse\(\'(.*?)\'\)"
```

The player data returned by Understat includes:

$$x^{\text{Understat}}_p = (\text{player\_name}_p, \text{games}_p, \text{time}_p, \text{goals}_p, \text{shots}_p, \text{xG}_p, \text{npxG}_p, \text{assists}_p, \text{xA}_p, \text{keys}_p, \text{yellow\_cards}_p, \text{red\_cards}_p).$$

**Name Matching**: Player names in Understat and FPL do not always match exactly. Fuzzy string matching (Levenshtein distance via the `fuzzywuzzy` library) with a threshold of 80 is employed:

$$\text{match}(p_{\text{FPL}}) = \arg\max_{q \in \mathcal{P}_{\text{Understat}}} \text{similarity}(p_{\text{FPL}}, q) \quad \text{if } \max \text{similarity} > 80.$$

### 3.3.2 Sentiment Analysis from News

News sentiment provides a leading indicator of player morale, managerial confidence, and off-pitch factors that may influence performance. Following the methodology of Tamimi and Tran (2025), who demonstrated that integrating news sentiment and injury data with statistical features reduces the Mean Squared Error of player performance prediction from 6.27 to 5.63, news articles are collected from sources such as The Guardian and GDELT, and a pre-trained transformer-based sentiment model is applied.

The **DeBERTa-v3-small-ft-news-sentiment-analysis** model from Hugging Face — the same architecture employed by Tamimi and Tran (2025) in their Method 2, which achieved the best results among Ridge Regression, LightGBM, CNN, and LSTM architectures on a two-season EPL dataset — is used. For each player $p$ at gameweek $t$, all articles mentioning the player in the 7 days preceding the gameweek deadline are collected, and sentiment features are computed:

$$\text{Sentiment}_p(t) = \frac{1}{N_p(t)} \sum_{i=1}^{N_p(t)} s_i, \quad s_i \in [-1, 1],$$

where $s_i$ is the sentiment score of article $i$, and $N_p(t)$ is the number of articles mentioning player $p$ in the relevant window.

A **sentiment volatility** feature is also computed:

$$\text{SentimentVol}_p(t) = \sqrt{\frac{1}{N_p(t) - 1} \sum_{i=1}^{N_p(t)} (s_i - \text{Sentiment}_p(t))^2}.$$

High sentiment volatility may indicate uncertainty surrounding the player (e.g., transfer rumours, injury concerns).

### 3.3.3 Injury Data

Injury status is a critical binary feature. Injury data is sourced from Fantasy Football Scout and the `WorldfootballR` package, which aggregates data from club websites and press conferences. For each player $p$ at gameweek $t$:

$$\text{Injured}_p(t) = \begin{cases} 1 & \text{if player } p \text{ is listed as injured/suspended/doubtful}, \\ 0 & \text{otherwise}. \end{cases}$$

The FPL API's `status` field (e.g., "a" for available, "i" for injured, "u" for unavailable) provides a secondary source, but it may lag behind real-world developments.

## 3.4 Feature Engineering

The raw data from all sources is transformed into a unified feature vector for each player-gameweek observation $(p, t)$. The feature vector is organised into categories:

### 3.4.1 Core Performance Features

These are the standard FPL output metrics, normalised per 90 minutes:

$$x^{\text{core}}_{p,t} = \left( \text{goals}^{/90}_{p,t}, \text{assists}^{/90}_{p,t}, \text{clean\_sheets}^{/90}_{p,t}, \text{saves}^{/90}_{p,t}, \text{bonus}^{/90}_{p,t}, \text{bps}^{/90}_{p,t}, \text{yellow\_cards}^{/90}_{p,t}, \text{red\_cards}^{/90}_{p,t} \right).$$

### 3.4.2 Advanced Performance Features

Derived from Understat data, also per 90:

$$x^{\text{adv}}_{p,t} = \left( \text{xG}^{/90}_{p,t}, \text{xA}^{/90}_{p,t}, \text{npxG}^{/90}_{p,t}, \text{shots}^{/90}_{p,t}, \text{key\_passes}^{/90}_{p,t}, \text{xGChain}^{/90}_{p,t}, \text{xGBuildup}^{/90}_{p,t} \right).$$

### 3.4.3 ICT Features

The ICT components from the FPL API:

$$x^{\text{ICT}}_{p,t} = \left( \text{influence}^{/90}_{p,t}, \text{creativity}^{/90}_{p,t}, \text{threat}^{/90}_{p,t} \right).$$

### 3.4.4 Form Features

Rolling averages over multiple windows capture short-term and medium-term trends:

$$x^{\text{form}}_{p,t} = \left( \text{Form}_p(3), \text{Form}_p(5), \text{Form}_p(10), \text{xGForm}_p(5), \text{xAForm}_p(5) \right),$$

where $\text{xGForm}_p(5) = \frac{1}{5} \sum_{k=1}^5 \text{xG}_p(t-k)$ is the rolling 5-game average of expected goals.

### 3.4.5 Fixture and Opponent Features

$$x^{\text{fixture}}_{p,t} = \left( \text{FDR}_p(t), \text{RollingFDR}_p(t, 3), \text{RollingFDR}_p(t, 5), \text{is\_home}_p(t), \text{rest\_days}_p(t) \right),$$

where $\text{rest\_days}_p(t)$ is the number of days since the player's last match (capturing fixture congestion).

### 3.4.6 Contextual Features

$$x^{\text{ctx}}_{p,t} = \left( \text{Injured}_p(t), \text{Sentiment}_p(t), \text{SentimentVol}_p(t), \text{ownership\_percent}_p(t), \text{price}_p(t), \text{price\_change}_p(t) \right).$$

### 3.4.7 Full Feature Vector

The complete feature vector for player $p$ at gameweek $t$ is the concatenation:

$$x_{p,t} = x^{\text{core}}_{p,t} \oplus x^{\text{adv}}_{p,t} \oplus x^{\text{ICT}}_{p,t} \oplus x^{\text{form}}_{p,t} \oplus x^{\text{fixture}}_{p,t} \oplus x^{\text{ctx}}_{p,t} \in \mathbb{R}^D,$$

where $D$ is the total feature dimensionality (typically 30–40 features after engineering).

## 3.5 Data Preprocessing Pipeline

### 3.5.1 Missing Value Imputation

Missing data arises from several sources: players who did not play (0 minutes), API fields that are null for new players, Understat data that is unavailable for some matches.

The following imputation strategy is applied:

- **Zero minutes**: For players with 0 minutes, all per-90 features are set to 0. This is a structural zero, not missing data.
- **Numerical features**: For players with some minutes but missing specific fields, median imputation is used within the player's position group:

  $$\tilde{x}_{p,t}^{(d)} = \begin{cases} x_{p,t}^{(d)} & \text{if observed}, \\ \text{median}_{p' \in \text{Position}(p)}(x_{p',t'}^{(d)}) & \text{if missing}. \end{cases}$$

- **Sentiment**: If no articles mention a player, $\text{Sentiment}_p(t) = 0$ (neutral) and $\text{SentimentVol}_p(t) = 0$ are set.

### 3.5.2 Standardisation

For the HDP-GMM, which assumes Gaussian emissions, all continuous features are standardised to have zero mean and unit variance:

$$\tilde{x}_{p,t}^{(d)} = \frac{x_{p,t}^{(d)} - \mu_d}{\sigma_d}, \quad \mu_d = \frac{1}{N} \sum_{p,t} x_{p,t}^{(d)}, \quad \sigma_d = \sqrt{\frac{1}{N-1} \sum_{p,t} (x_{p,t}^{(d)} - \mu_d)^2}.$$

The mean $\mu_d$ and standard deviation $\sigma_d$ are computed on the training set only and applied to all subsequent data to prevent data leakage.

### 3.5.3 Position-Encoding

Player position is a categorical variable that is not directly fed into the HDP-GMM (which operates on continuous features). Instead, separate HDP-GMM models are maintained for each position group (GK, DEF, MID, FWD), as the performance distributions differ substantially across positions. This is equivalent to conditioning the model on position:

$$P(x_{p,t} \mid z_{p,t} = k) = P(x_{p,t} \mid z_{p,t} = k, \text{position}(p)).$$

## 3.6 Exploratory Data Analysis (EDA)

EDA is essential for understanding the data's statistical properties, identifying anomalies, and guiding feature selection.

### 3.6.1 Univariate Analysis

**Distribution of FPL Points**: Player-gameweek totals are discrete and right-skewed, with a substantial mass at zero and a tail of high-scoring performances. They are not restricted to nonnegative counts: in the normalized 2023–24 Vaastav archive used in the accompanying study ([merged gameweek data](https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2023-24/gws/merged_gw.csv)), the project loader yields 28,742 player-gameweek rows, of which 162 have negative points (minimum −4) and 18,344 have zero points; the mean is 1.088 and the variance is 5.618. Right skewness and over-dispersion make count models worth investigating, but they do not make an ordinary Negative Binomial valid for raw total points: its support is the nonnegative integers. It is suitable only for a nonnegative count outcome; a model of unmodified FPL totals needs a likelihood that also supports negative values.

**Price Distribution**: Player prices follow a bimodal distribution, with peaks at the lower end (£4.0–5.0m for backups) and the upper end (£10.0–14.0m for elite players). The bimodality reflects the FPL pricing algorithm, which positions players relative to their expected output.

**ICT Index by Position**: The ICT Index distribution varies dramatically by position. Forwards and attacking midfielders dominate the upper tail, while defenders and goalkeepers cluster near the lower end. This confirms that separate models per position are necessary.

### 3.6.2 Bivariate Analysis

**xG vs. Actual Goals**: A scatter plot of xG per 90 vs. goals per 90 reveals the relationship between chance quality and actual output. Players above the diagonal ($\text{goals} > \text{xG}$) are over-performing their expected output (clinical finishers or lucky), while players below are under-performing. The correlation coefficient is typically $r \approx 0.6$–0.7, confirming that xG is a strong but imperfect predictor of goals.

**Correlation Heatmap**: A Pearson correlation matrix of all features reveals multicollinearity patterns:
- xG and shots per 90: $r \approx 0.85$ (highly correlated, as xG is derived from shot locations).
- xA and key passes per 90: $r \approx 0.80$.
- ICT Index and total points: $r \approx 0.75$.
- Influence and Threat: $r \approx 0.65$ (moderately correlated, but each provides unique information).

Features with pairwise correlation $|r| > 0.9$ are candidates for removal to avoid numerical instability in the Gaussian-Wishart model.

### 3.6.3 Time Series Analysis

**Rolling Form Trajectories**: Plotting a player's rolling 5-game average points over time reveals "hot streaks" and "slumps." For example, a forward who scores 3 goals in 2 gameweeks will show a sharp spike in rolling form, followed by a gradual decay as the window slides past those games. The HDP-HMM is designed to capture these transitions as changes in latent state.

**Price Change Dynamics**: Plotting player price changes over time reveals the lag structure of the FPL pricing algorithm. Price changes typically trail performance by 2–4 gameweeks, creating a window of opportunity for managers who can identify emerging form before the market adjusts.

### 3.6.4 Position-Specific Patterns

**Goalkeepers**: Clean sheets and saves dominate the point distribution. The variance is lower than for outfield positions.

**Defenders**: A bimodal distribution exists: attacking full-backs (high creativity/threat) vs. centre-backs (high defensive contribution, low offensive output).

**Midfielders**: The most heterogeneous position group. The HDP-GMM is expected to discover multiple sub-profiles (defensive midfielders, wide creators, attacking #10s).

**Forwards**: Goals dominate. High variance, with occasional massive hauls (15+ points) separated by long sequences of blank returns.

## 3.7 Feature Selection

Not all engineered features are equally predictive. Feature selection reduces dimensionality, removes noise, and improves model interpretability.

### 3.7.1 Filter Methods

**Mutual Information**: The mutual information between each feature $X^{(d)}$ and the target variable $Y$ (total FPL points) is computed:

$$\text{MI}(X^{(d)}; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}.$$

Features with $\text{MI}(X^{(d)}; Y) < \epsilon$ (e.g., $\epsilon = 0.01$) are discarded.

**Correlation Threshold**: Features with $|r| > 0.9$ with another feature are reduced to a single representative. For example, if xG and shots are highly correlated, xG is retained (the more predictive of the two).

### 3.7.2 Embedded Methods

**Lasso (L1) Regularisation**: A linear model with L1 penalty is fit:

$$\min_{\beta} \; \frac{1}{2N} \sum_{i=1}^N (y_i - \beta^\top x_i)^2 + \lambda \|\beta\|_1.$$

Features with $\beta^{(d)} = 0$ are eliminated. The regularisation strength $\lambda$ is tuned via cross-validation.

**Tree-Based Feature Importance**: A Gradient Boosting model (e.g., XGBoost) is trained and the feature importance scores (mean decrease in impurity) are extracted. This captures non-linear feature relevance that linear methods miss.

### 3.7.3 Final Feature Set

After feature selection, the retained features typically include:

- xG per 90, xA per 90, npxG per 90
- Shots per 90, key passes per 90
- Influence per 90, Creativity per 90, Threat per 90
- Rolling form (3, 5 game windows)
- FDR, Rolling FDR (3 game window)
- is_home, rest_days
- Sentiment, ownership_percent
- Injured (binary)

This yields a final dimensionality of $D \approx 15$–20, suitable for the HDP-GMM.

## 3.8 Data Pipeline Implementation

The data pipeline is implemented in Python, structured as follows:

```
data_fetcher.py
├── get_fpl_bootstrap_data()     # Fetches players, teams, events
├── get_fpl_fixture_data()        # Fetches all 380 fixtures
├── get_fpl_gameweek_data(gw)     # Fetches live data for GW
├── get_understat_data(season)    # Async scraping of Understat
└── update_data(latest_gameweek)  # Orchestrates full pipeline
```

The `update_data` function is the entry point, called by a scheduled job (e.g., Airflow DAG) after each gameweek concludes. It fetches all data sources, performs merging (using fuzzy name matching for Understat), engineers per-90 features, and writes the results to versioned CSV files in the `data/` directory.

The data is versioned using **DVC (Data Version Control)**, which tracks data file hashes in Git and stores the actual files in a remote storage backend (e.g., AWS S3, Google Drive). This ensures that every experiment is linked to a specific data version, enabling full reproducibility.

## 3.9 Summary

This chapter has established the data foundation for the FPL sequential optimisation system. The FPL API provides the structural backbone — player statistics, fixtures, and pricing — while external sources (Understat, news sentiment, injury data) enrich the feature set with predictive signals that transcend basic FPL metrics. The feature engineering pipeline transforms raw data into per-90 normalised, rolling-averaged, and contextually enriched features. EDA reveals the statistical properties of the data — skewness, multimodality, multicollinearity — that guide model selection. Feature selection reduces dimensionality while retaining predictive power. The pipeline is automated, versioned, and reproducible, satisfying the requirements of a production-grade MLOps system.

The next chapter develops the theoretical foundations of Bayesian nonparametrics — the Chinese Restaurant Process, the Dirichlet Process, and the Hierarchical Dirichlet Process — which underpin the latent player form modelling approach.
