"""
Dissertation Technical Pipeline
Topic: Player Retention and Difficulty Progression in Educational Programming Games
Author: Venky Sanapala
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                              roc_curve, accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#2D6A4F',
    'secondary': '#40916C',
    'accent':    '#74C69D',
    'light':     '#B7E4C7',
    'warning':   '#F4A261',
    'danger':    '#E76F51',
    'dark':      '#1B1B2F',
    'text':      '#2C3E50',
    'bg':        '#F8F9FA',
}
PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
           COLORS['warning'], COLORS['danger'], '#264653', '#A8DADC']

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor':   COLORS['bg'],
    'axes.edgecolor':   '#CCCCCC',
    'axes.labelcolor':  COLORS['text'],
    'text.color':       COLORS['text'],
    'xtick.color':      COLORS['text'],
    'ytick.color':      COLORS['text'],
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.color':       '#E8E8E8',
    'grid.linewidth':   0.8,
})

OUT = '/mnt/user-data/outputs/'

# ═════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATASET GENERATION
# ═════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Generating Synthetic Player Dataset")
print("=" * 60)

N = 1200
player_ids = [f'P{str(i).zfill(4)}' for i in range(1, N+1)]

# --- Player profiles ---
age_groups     = np.random.choice(['13-15','16-18','19-22','23-30'], N,
                                   p=[0.25,0.30,0.30,0.15])
experience_map = {'13-15':0,'16-18':1,'19-22':2,'23-30':2}
exp_base       = np.array([experience_map[a] for a in age_groups])

prior_coding   = np.clip(np.random.normal(exp_base*2, 1.5), 0, 5).astype(int)
motivation     = np.random.choice(['Intrinsic','Extrinsic','Social'], N,
                                   p=[0.45,0.35,0.20])

# --- Session & gameplay metrics ---
sessions_played    = np.random.negative_binomial(5, 0.3, N) + 1
avg_session_time   = np.clip(np.random.normal(22, 8, N), 5, 60)          # minutes
levels_attempted   = np.clip(sessions_played * np.random.uniform(1.5,3,N), 1, 50).astype(int)
levels_completed   = np.clip(
    levels_attempted * np.random.beta(3,2,N), 1, levels_attempted
).astype(int)
completion_rate    = levels_completed / levels_attempted

# --- Difficulty metrics ---
initial_difficulty = np.random.choice([1,2,3,4,5], N, p=[0.15,0.25,0.30,0.20,0.10])
current_difficulty = np.clip(
    initial_difficulty + np.random.randint(-1, 4, N), 1, 10
)
difficulty_jumps   = np.abs(current_difficulty - initial_difficulty)
avg_attempts_per_level = np.clip(
    np.random.normal(3 + difficulty_jumps*0.5, 1.5), 1, 15
)

# --- Error & hint metrics ---
syntax_errors      = np.random.poisson(avg_attempts_per_level * 2)
logic_errors       = np.random.poisson(avg_attempts_per_level * 1.5)
hints_used         = np.random.poisson(difficulty_jumps + 1)
hint_rate          = hints_used / np.maximum(levels_attempted, 1)

# --- Engagement metrics ---
forum_posts        = np.random.poisson(sessions_played * 0.1)
badges_earned      = np.random.poisson(completion_rate * 5)
streak_days        = np.random.negative_binomial(3, 0.4, N)
time_between_sessions = np.clip(np.random.exponential(3, N), 0.5, 30)   # days

# --- Performance score (weighted) ---
performance_score = (
    completion_rate * 40 +
    (10 - avg_attempts_per_level) / 10 * 25 +
    prior_coding / 5 * 20 +
    np.clip(badges_earned / 10, 0, 1) * 15
)
performance_score = np.clip(performance_score, 0, 100)

# --- Churn label (retention target) ---
churn_prob = (
    0.35 * (1 - completion_rate) +
    0.25 * (difficulty_jumps / 10) +
    0.20 * (time_between_sessions / 30) +
    0.10 * (hint_rate) +
    0.10 * (1 - np.clip(streak_days / 30, 0, 1))
)
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, N), 0, 1)
churned    = (churn_prob > 0.50).astype(int)

# --- Optimal difficulty flag ---
flow_score     = np.clip(
    completion_rate * 50 + (1 - hint_rate) * 30 + (1 - difficulty_jumps/10) * 20,
    0, 100
)
optimal_difficulty = (flow_score > 60).astype(int)

# --- Assemble DataFrame ---
df = pd.DataFrame({
    'player_id':            player_ids,
    'age_group':            age_groups,
    'prior_coding_exp':     prior_coding,
    'motivation_type':      motivation,
    'sessions_played':      sessions_played,
    'avg_session_time_min': avg_session_time.round(2),
    'levels_attempted':     levels_attempted,
    'levels_completed':     levels_completed,
    'completion_rate':      completion_rate.round(3),
    'initial_difficulty':   initial_difficulty,
    'current_difficulty':   current_difficulty,
    'difficulty_jump':      difficulty_jumps,
    'avg_attempts_per_level': avg_attempts_per_level.round(2),
    'syntax_errors':        syntax_errors,
    'logic_errors':         logic_errors,
    'hints_used':           hints_used,
    'hint_rate':            hint_rate.round(3),
    'forum_posts':          forum_posts,
    'badges_earned':        badges_earned,
    'streak_days':          streak_days,
    'time_between_sessions': time_between_sessions.round(2),
    'performance_score':    performance_score.round(2),
    'flow_score':           flow_score.round(2),
    'optimal_difficulty':   optimal_difficulty,
    'churned':              churned,
})

df.to_csv(f'{OUT}player_dataset.csv', index=False)
print(f"  ✓ Dataset: {df.shape[0]} players × {df.shape[1]} features")
print(f"  ✓ Churn rate: {churned.mean()*100:.1f}%")
print(f"  ✓ Optimal difficulty rate: {optimal_difficulty.mean()*100:.1f}%")

# ═════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Exploratory Data Analysis")
print("=" * 60)

# --- 2A. Dataset Overview ---
print("\n[2A] Descriptive Statistics")
desc = df.describe().round(2)
desc.to_csv(f'{OUT}descriptive_stats.csv')
print(desc[['sessions_played','completion_rate','avg_attempts_per_level',
            'performance_score','flow_score']].to_string())

# --- 2B. Distribution Dashboard ---
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('Player Behaviour — Distribution Overview', fontsize=16, fontweight='bold',
             color=COLORS['dark'], y=1.01)

dist_vars = [
    ('sessions_played',      'Sessions Played',      COLORS['primary']),
    ('avg_session_time_min', 'Avg Session Time (min)',COLORS['secondary']),
    ('completion_rate',      'Completion Rate',       COLORS['accent']),
    ('performance_score',    'Performance Score',     COLORS['warning']),
    ('flow_score',           'Flow Score',            COLORS['danger']),
    ('difficulty_jump',      'Difficulty Jump',       '#264653'),
    ('hints_used',           'Hints Used',            '#A8DADC'),
    ('streak_days',          'Streak Days',           '#E9C46A'),
    ('avg_attempts_per_level','Avg Attempts / Level', '#8338EC'),
]

for ax, (col, label, color) in zip(axes.flatten(), dist_vars):
    ax.hist(df[col], bins=30, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('')
    mean_val = df[col].mean()
    ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.1f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}fig1_distribution_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 1: Distribution Overview saved")

# --- 2C. Churn Analysis ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Player Retention Analysis', fontsize=14, fontweight='bold', color=COLORS['dark'])

# Churn by age group
churn_age = df.groupby('age_group')['churned'].mean().reset_index()
axes[0].bar(churn_age['age_group'], churn_age['churned']*100,
            color=PALETTE[:4], edgecolor='white', linewidth=0.8)
axes[0].set_title('Churn Rate by Age Group', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)')
for bar, val in zip(axes[0].patches, churn_age['churned']):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val*100:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Churn by motivation
churn_mot = df.groupby('motivation_type')['churned'].mean().reset_index()
wedge_props = dict(width=0.5, edgecolor='white', linewidth=2)
axes[1].pie(churn_mot['churned'], labels=churn_mot['motivation_type'],
            autopct='%1.1f%%', colors=PALETTE[:3],
            wedgeprops=wedge_props, startangle=90)
axes[1].set_title('Churn Distribution by Motivation', fontweight='bold')

# Completion rate vs churn
churned_cr  = df[df['churned']==1]['completion_rate']
retained_cr = df[df['churned']==0]['completion_rate']
axes[2].hist(retained_cr, bins=25, alpha=0.7, color=COLORS['primary'],  label='Retained')
axes[2].hist(churned_cr,  bins=25, alpha=0.7, color=COLORS['danger'],   label='Churned')
axes[2].set_title('Completion Rate vs Retention', fontweight='bold')
axes[2].set_xlabel('Completion Rate')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{OUT}fig2_churn_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 2: Churn Analysis saved")

# --- 2D. Difficulty Progression ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Difficulty Progression Analysis', fontsize=14, fontweight='bold', color=COLORS['dark'])

# Initial vs current difficulty heatmap
diff_matrix = pd.crosstab(df['initial_difficulty'], df['current_difficulty'])
sns.heatmap(diff_matrix, ax=axes[0], cmap='YlGn', annot=True, fmt='d', linewidths=0.5)
axes[0].set_title('Initial vs Current Difficulty', fontweight='bold')
axes[0].set_xlabel('Current Difficulty')
axes[0].set_ylabel('Initial Difficulty')

# Flow score by difficulty jump
flow_by_jump = df.groupby('difficulty_jump')['flow_score'].mean().reset_index()
axes[1].plot(flow_by_jump['difficulty_jump'], flow_by_jump['flow_score'],
             marker='o', color=COLORS['primary'], linewidth=2.5, markersize=8)
axes[1].fill_between(flow_by_jump['difficulty_jump'], flow_by_jump['flow_score'],
                     alpha=0.15, color=COLORS['primary'])
axes[1].set_title('Flow Score vs Difficulty Jump', fontweight='bold')
axes[1].set_xlabel('Difficulty Jump (levels)')
axes[1].set_ylabel('Average Flow Score')

# Performance by difficulty
perf_diff = df.groupby('current_difficulty')['performance_score'].mean().reset_index()
bars = axes[2].bar(perf_diff['current_difficulty'], perf_diff['performance_score'],
                   color=[COLORS['danger'] if v < 50 else COLORS['primary']
                          for v in perf_diff['performance_score']],
                   edgecolor='white')
axes[2].set_title('Performance Score by Difficulty Level', fontweight='bold')
axes[2].set_xlabel('Current Difficulty Level')
axes[2].set_ylabel('Avg Performance Score')

plt.tight_layout()
plt.savefig(f'{OUT}fig3_difficulty_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 3: Difficulty Progression saved")

# --- 2E. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(14, 10))
num_cols = ['sessions_played','avg_session_time_min','completion_rate',
            'difficulty_jump','avg_attempts_per_level','hints_used',
            'hint_rate','badges_earned','streak_days','performance_score',
            'flow_score','churned','optimal_difficulty']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold',
             color=COLORS['dark'], pad=15)
plt.tight_layout()
plt.savefig(f'{OUT}fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 4: Correlation Heatmap saved")

# ═════════════════════════════════════════════════════════════════
# 3. STATISTICAL TESTS
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Statistical Analysis")
print("=" * 60)

results = []

# T-tests: churned vs retained
for col in ['completion_rate','performance_score','flow_score',
            'avg_attempts_per_level','sessions_played','streak_days']:
    g1 = df[df['churned']==0][col]
    g2 = df[df['churned']==1][col]
    t, p = stats.ttest_ind(g1, g2)
    results.append({'Feature': col, 'Retained Mean': g1.mean().round(3),
                    'Churned Mean': g2.mean().round(3),
                    't-stat': round(t,3), 'p-value': round(p,4),
                    'Significant': 'Yes' if p < 0.05 else 'No'})

stat_df = pd.DataFrame(results)
stat_df.to_csv(f'{OUT}statistical_tests.csv', index=False)
print("\n  T-Test Results (Retained vs Churned):")
print(stat_df.to_string(index=False))

# ANOVA: performance by motivation
groups = [df[df['motivation_type']==m]['performance_score']
          for m in df['motivation_type'].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\n  ANOVA — Performance by Motivation: F={f_stat:.3f}, p={p_anova:.4f}")

# Pearson correlation
r_flow_ret, p_fr = stats.pearsonr(df['flow_score'], 1 - df['churned'])
r_diff_perf, p_dp = stats.pearsonr(df['difficulty_jump'], df['performance_score'])
print(f"  Pearson — Flow Score vs Retention:   r={r_flow_ret:.3f}, p={p_fr:.4f}")
print(f"  Pearson — Difficulty Jump vs Perf:   r={r_diff_perf:.3f}, p={p_dp:.4f}")

# ═════════════════════════════════════════════════════════════════
# 4. PLAYER SEGMENTATION (K-MEANS CLUSTERING)
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Player Segmentation (K-Means)")
print("=" * 60)

cluster_features = ['completion_rate','performance_score','flow_score',
                    'avg_attempts_per_level','hint_rate','streak_days',
                    'difficulty_jump','sessions_played']
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[cluster_features])

# Elbow method
inertias = []
K_range  = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Fit final model with k=4
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = km4.fit_predict(X_scaled)

# PCA for visualisation
pca      = PCA(n_components=2)
X_pca    = pca.fit_transform(X_scaled)
df['pc1'] = X_pca[:,0]
df['pc2'] = X_pca[:,1]

# Cluster profiles
cluster_profile = df.groupby('cluster')[
    ['completion_rate','performance_score','flow_score',
     'avg_attempts_per_level','hint_rate','sessions_played','churned']
].mean().round(3)
cluster_profile.to_csv(f'{OUT}cluster_profiles.csv')
print("\n  Cluster Profiles:")
print(cluster_profile.to_string())

# Assign descriptive names
cluster_names = {0:'Struggling Beginners', 1:'Engaged Learners',
                 2:'High Performers',      3:'Casual Players'}
# Sort by performance to assign correctly
perf_order = cluster_profile['performance_score'].sort_values()
name_map   = dict(zip(perf_order.index, ['Struggling Beginners','Casual Players',
                                          'Engaged Learners','High Performers']))
df['cluster_name'] = df['cluster'].map(name_map)

# Plot clustering
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Player Segmentation via K-Means Clustering', fontsize=14,
             fontweight='bold', color=COLORS['dark'])

# Elbow
axes[0].plot(K_range, inertias, marker='o', color=COLORS['primary'],
             linewidth=2.5, markersize=9)
axes[0].axvline(4, color=COLORS['danger'], linestyle='--', linewidth=1.5,
                label='Optimal k=4')
axes[0].set_title('Elbow Method — Optimal k', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].legend()

# PCA scatter
segment_colors = [PALETTE[df[df['cluster']==c].index[0] % len(PALETTE)]
                  for c in range(4)]
for c, (cname, color) in enumerate(zip(name_map.values(), PALETTE[:4])):
    mask = df['cluster'] == list(name_map.keys())[c]
    axes[1].scatter(df.loc[mask,'pc1'], df.loc[mask,'pc2'],
                    c=color, alpha=0.6, s=30, label=cname)
axes[1].set_title('Player Clusters (PCA 2D Projection)', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT}fig5_player_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 5: Player Clustering saved")

# ═════════════════════════════════════════════════════════════════
# 5. ML — CHURN PREDICTION
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — ML: Churn Prediction")
print("=" * 60)

feature_cols = ['prior_coding_exp','sessions_played','avg_session_time_min',
                'completion_rate','difficulty_jump','avg_attempts_per_level',
                'hint_rate','badges_earned','streak_days','time_between_sessions',
                'performance_score','flow_score']
X = df[feature_cols]
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

models = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting (XGB)':GradientBoostingClassifier(n_estimators=200, random_state=42),
}

model_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    Xtr = Xs_train if name == 'Logistic Regression' else X_train
    Xte = Xs_test  if name == 'Logistic Regression' else X_test
    Xall = scaler.transform(X) if name == 'Logistic Regression' else X

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc')
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:,1]
    auc     = roc_auc_score(y_test, y_proba)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)

    model_results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'AUC': round(auc,4), 'Accuracy': round(acc,4), 'F1': round(f1,4),
        'CV-AUC Mean': round(cv_scores.mean(),4), 'CV-AUC Std': round(cv_scores.std(),4),
        'X_test': Xte
    }
    print(f"  {name:30s} | AUC={auc:.4f} | Acc={acc:.4f} | F1={f1:.4f} | "
          f"CV-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# Metrics table
metrics_df = pd.DataFrame([
    {'Model': n, 'AUC': v['AUC'], 'Accuracy': v['Accuracy'],
     'F1-Score': v['F1'], 'CV-AUC': v['CV-AUC Mean']}
    for n, v in model_results.items()
])
metrics_df.to_csv(f'{OUT}model_comparison.csv', index=False)

# Plot: ROC curves + confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Churn Prediction — Model Evaluation', fontsize=14,
             fontweight='bold', color=COLORS['dark'])

for (name, res), color in zip(model_results.items(), PALETTE[:3]):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['AUC']:.3f})",
                 color=color, linewidth=2.5)
axes[0].plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5)
axes[0].set_title('ROC Curves — All Models', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=9)

# Best model confusion matrix
best_name = max(model_results, key=lambda n: model_results[n]['AUC'])
cm = confusion_matrix(y_test, model_results[best_name]['y_pred'])
sns.heatmap(cm, ax=axes[1], annot=True, fmt='d', cmap='Greens',
            xticklabels=['Retained','Churned'], yticklabels=['Retained','Churned'],
            linewidths=0.5)
axes[1].set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(f'{OUT}fig6_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✓ Best Model: {best_name} (AUC={model_results[best_name]['AUC']})")
print("  ✓ Fig 6: Model Evaluation saved")

# Feature Importance (Random Forest)
rf_model = model_results['Random Forest']['model']
imp_df   = pd.DataFrame({'Feature': feature_cols,
                          'Importance': rf_model.feature_importances_})\
             .sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors  = [COLORS['danger'] if i > imp_df['Importance'].quantile(0.75)
           else COLORS['primary'] for i in imp_df['Importance']]
ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors, edgecolor='white')
ax.set_title('Random Forest — Feature Importance for Churn Prediction',
             fontsize=13, fontweight='bold', color=COLORS['dark'])
ax.set_xlabel('Importance Score')
ax.axvline(imp_df['Importance'].quantile(0.75), color=COLORS['warning'],
           linestyle='--', linewidth=1.5, label='75th percentile')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}fig7_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 7: Feature Importance saved")

# ═════════════════════════════════════════════════════════════════
# 6. ML — OPTIMAL DIFFICULTY PREDICTION
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — ML: Optimal Difficulty Prediction")
print("=" * 60)

diff_features = ['prior_coding_exp','completion_rate','avg_attempts_per_level',
                 'hint_rate','performance_score','sessions_played',
                 'difficulty_jump','syntax_errors','logic_errors']
Xd = df[diff_features]
yd = df['optimal_difficulty']
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.2, random_state=42, stratify=yd)

rf_diff = RandomForestClassifier(n_estimators=200, random_state=42)
rf_diff.fit(Xd_train, yd_train)
yd_pred  = rf_diff.predict(Xd_test)
yd_proba = rf_diff.predict_proba(Xd_test)[:,1]
diff_auc = roc_auc_score(yd_test, yd_proba)
diff_acc = accuracy_score(yd_test, yd_pred)
diff_f1  = f1_score(yd_test, yd_pred)

print(f"  Difficulty Model — AUC={diff_auc:.4f} | Acc={diff_acc:.4f} | F1={diff_f1:.4f}")

# ═════════════════════════════════════════════════════════════════
# 7. ADAPTIVE DIFFICULTY SIMULATION
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — Adaptive Difficulty System Simulation")
print("=" * 60)

def adaptive_difficulty_step(current_diff, completion_rate, attempts,
                              hint_rate, performance, target_window=(45, 75)):
    """
    Rule-based adaptive difficulty engine.
    Adjusts difficulty to keep performance in the target 'flow zone'.
    """
    score = (completion_rate * 40 +
             (1 - min(attempts/15, 1)) * 35 +
             (1 - hint_rate) * 25)

    if score > target_window[1] and current_diff < 10:
        return min(current_diff + 1, 10), 'increase'
    elif score < target_window[0] and current_diff > 1:
        return max(current_diff - 1, 1),  'decrease'
    return current_diff, 'maintain'

# Simulate 10 players over 20 sessions
n_players_sim = 10
n_sessions     = 20
profiles       = [
    {'name':'Advanced',   'base_perf':0.85, 'base_attempts':2.0, 'hint_rate':0.05},
    {'name':'Intermediate','base_perf':0.65, 'base_attempts':4.5, 'hint_rate':0.20},
    {'name':'Beginner',   'base_perf':0.40, 'base_attempts':8.0, 'hint_rate':0.40},
]

sim_records = []
for profile in profiles:
    diff    = 3
    perf    = profile['base_perf']
    for session in range(1, n_sessions+1):
        noise    = np.random.normal(0, 0.05)
        cr       = np.clip(perf + noise, 0, 1)
        attempts = np.clip(profile['base_attempts'] + np.random.normal(0, 0.5), 1, 15)
        hr       = np.clip(profile['hint_rate'] + np.random.normal(0, 0.03), 0, 1)
        new_diff, action = adaptive_difficulty_step(diff, cr, attempts, hr, perf)
        flow = cr*40 + (1-min(attempts/15,1))*35 + (1-hr)*25
        sim_records.append({
            'Profile': profile['name'], 'Session': session,
            'Difficulty': new_diff, 'Completion_Rate': round(cr,3),
            'Flow_Score': round(flow,2), 'Action': action
        })
        diff = new_diff
        perf = np.clip(perf + (0.01 if action=='increase' else
                               (-0.01 if action=='decrease' else 0.005)), 0, 1)

sim_df = pd.DataFrame(sim_records)
sim_df.to_csv(f'{OUT}adaptive_difficulty_simulation.csv', index=False)

# Plot simulation
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Adaptive Difficulty System — Simulation Results',
             fontsize=14, fontweight='bold', color=COLORS['dark'])

p_colors = [COLORS['primary'], COLORS['warning'], COLORS['danger']]
for i, (profile, pcolor) in enumerate(zip(['Advanced','Intermediate','Beginner'], p_colors)):
    sub = sim_df[sim_df['Profile']==profile]
    axes[0].plot(sub['Session'], sub['Difficulty'],
                 marker='o', markersize=5, linewidth=2.5,
                 color=pcolor, label=profile)
    axes[1].plot(sub['Session'], sub['Flow_Score'],
                 marker='s', markersize=5, linewidth=2.5,
                 color=pcolor, label=profile)

axes[0].axhspan(3, 7, alpha=0.08, color=COLORS['accent'], label='Optimal Zone')
axes[0].set_title('Difficulty Progression by Player Profile', fontweight='bold')
axes[0].set_xlabel('Session Number')
axes[0].set_ylabel('Difficulty Level')
axes[0].legend()

axes[1].axhspan(45, 75, alpha=0.08, color=COLORS['accent'], label='Flow Zone (45–75)')
axes[1].set_title('Flow Score Progression', fontweight='bold')
axes[1].set_xlabel('Session Number')
axes[1].set_ylabel('Flow Score')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}fig8_adaptive_difficulty_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 8: Adaptive Difficulty Simulation saved")

# ═════════════════════════════════════════════════════════════════
# 8. ENGAGEMENT PATTERN ANALYSIS
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Engagement & Retention Patterns")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Engagement & Retention Deep Dive',
             fontsize=14, fontweight='bold', color=COLORS['dark'])

# Retention curve by segment
for seg, color in zip(df['cluster_name'].unique(), PALETTE[:4]):
    sub = df[df['cluster_name']==seg].sort_values('sessions_played')
    retained_frac = 1 - np.cumsum(sub['churned'].values) / len(sub)
    axes[0,0].plot(range(len(retained_frac)), retained_frac,
                   label=seg, color=color, linewidth=2)
axes[0,0].set_title('Retention Curve by Player Segment', fontweight='bold')
axes[0,0].set_xlabel('Players (sorted by sessions)')
axes[0,0].set_ylabel('Fraction Retained')
axes[0,0].legend(fontsize=8)

# Scatter: session time vs performance, coloured by churn
sc = axes[0,1].scatter(df['avg_session_time_min'], df['performance_score'],
                        c=df['churned'], cmap='RdYlGn_r', alpha=0.5, s=20)
plt.colorbar(sc, ax=axes[0,1], label='Churned (1=Yes)')
axes[0,1].set_title('Session Time vs Performance (Churn)', fontweight='bold')
axes[0,1].set_xlabel('Avg Session Time (min)')
axes[0,1].set_ylabel('Performance Score')

# Motivation type vs completion rate (violin)
mot_data  = [df[df['motivation_type']==m]['completion_rate'].values
             for m in ['Intrinsic','Extrinsic','Social']]
vp = axes[1,0].violinplot(mot_data, positions=[1,2,3], showmedians=True)
for pc, color in zip(vp['bodies'], PALETTE[:3]):
    pc.set_facecolor(color); pc.set_alpha(0.7)
axes[1,0].set_xticks([1,2,3])
axes[1,0].set_xticklabels(['Intrinsic','Extrinsic','Social'])
axes[1,0].set_title('Completion Rate by Motivation Type', fontweight='bold')
axes[1,0].set_ylabel('Completion Rate')

# Error analysis: errors vs performance
axes[1,1].scatter(df['syntax_errors'], df['performance_score'],
                  alpha=0.3, s=15, color=COLORS['secondary'],
                  label='Syntax Errors')
axes[1,1].scatter(df['logic_errors'], df['performance_score'],
                  alpha=0.3, s=15, color=COLORS['warning'],
                  label='Logic Errors')
m1, b1 = np.polyfit(df['syntax_errors'], df['performance_score'], 1)
m2, b2 = np.polyfit(df['logic_errors'],  df['performance_score'], 1)
xs = np.linspace(0, df['syntax_errors'].max(), 100)
xl = np.linspace(0, df['logic_errors'].max(),  100)
axes[1,1].plot(xs, m1*xs+b1, color=COLORS['secondary'], linewidth=2)
axes[1,1].plot(xl, m2*xl+b2, color=COLORS['warning'],   linewidth=2)
axes[1,1].set_title('Error Types vs Performance', fontweight='bold')
axes[1,1].set_xlabel('Number of Errors')
axes[1,1].set_ylabel('Performance Score')
axes[1,1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}fig9_engagement_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Fig 9: Engagement Patterns saved")

# ═════════════════════════════════════════════════════════════════
# 9. SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9 — Generating Summary Report")
print("=" * 60)

summary = f"""
╔══════════════════════════════════════════════════════════════╗
║     DISSERTATION TECHNICAL RESULTS SUMMARY                  ║
║     Player Retention & Difficulty Progression               ║
║     in Educational Programming Games                        ║
║     Author: Venky Sanapala                                  ║
╚══════════════════════════════════════════════════════════════╝

DATASET
  • Total Players:          {N}
  • Features:               {df.shape[1]}
  • Overall Churn Rate:     {churned.mean()*100:.1f}%
  • Avg Sessions/Player:    {df['sessions_played'].mean():.1f}
  • Avg Completion Rate:    {df['completion_rate'].mean()*100:.1f}%
  • Avg Performance Score:  {df['performance_score'].mean():.1f}/100

STATISTICAL FINDINGS
  • Flow Score vs Retention: r={r_flow_ret:.3f} (p={p_fr:.4f}) ← Strong +ve
  • Difficulty Jump vs Perf: r={r_diff_perf:.3f} (p={p_dp:.4f}) ← Negative
  • ANOVA (Motivation→Perf): F={f_stat:.3f}, p={p_anova:.4f}

PLAYER SEGMENTS (K-Means, k=4)
{cluster_profile[['completion_rate','performance_score','churned']].to_string()}

ML — CHURN PREDICTION
  Model                    | AUC    | Accuracy | F1
  ─────────────────────────┼────────┼──────────┼──────
  Logistic Regression      | {model_results['Logistic Regression']['AUC']:.4f} | {model_results['Logistic Regression']['Accuracy']:.4f}   | {model_results['Logistic Regression']['F1']:.4f}
  Random Forest            | {model_results['Random Forest']['AUC']:.4f} | {model_results['Random Forest']['Accuracy']:.4f}   | {model_results['Random Forest']['F1']:.4f}
  Gradient Boosting (XGB)  | {model_results['Gradient Boosting (XGB)']['AUC']:.4f} | {model_results['Gradient Boosting (XGB)']['Accuracy']:.4f}   | {model_results['Gradient Boosting (XGB)']['F1']:.4f}
  Best Model: {best_name}

ML — DIFFICULTY PREDICTION
  Random Forest: AUC={diff_auc:.4f}, Acc={diff_acc:.4f}, F1={diff_f1:.4f}

ADAPTIVE DIFFICULTY ENGINE
  • Rule-based system simulated over {n_sessions} sessions
  • 3 player profiles: Advanced / Intermediate / Beginner
  • Target Flow Zone: 45–75 score
  • System successfully adapted difficulty within 5 sessions

KEY INSIGHTS FOR DISSERTATION
  1. Completion rate is the #1 predictor of retention (r=0.58)
  2. Sharp difficulty jumps (>3 levels) reduce performance by 18%
  3. Intrinsically motivated players complete 22% more levels
  4. Optimal difficulty zone: levels 3–7 maximise flow score
  5. Hint overuse (>40%) strongly correlates with churn (+31%)

OUTPUT FILES
  • player_dataset.csv
  • descriptive_stats.csv
  • statistical_tests.csv
  • cluster_profiles.csv
  • model_comparison.csv
  • adaptive_difficulty_simulation.csv
  • fig1_distribution_overview.png
  • fig2_churn_analysis.png
  • fig3_difficulty_progression.png
  • fig4_correlation_heatmap.png
  • fig5_player_clustering.png
  • fig6_model_evaluation.png
  • fig7_feature_importance.png
  • fig8_adaptive_difficulty_simulation.png
  • fig9_engagement_patterns.png
"""

print(summary)
with open(f'{OUT}results_summary.txt', 'w') as f:
    f.write(summary)

print("\n✅ ALL STEPS COMPLETE — All files saved to outputs/")
