import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer

data = {
    'User ID': list(range(1, 101)),
    'Age': [25, 34, 29, 41, 22, 37, 30, 45, 27, 32, 50, 23, 36, 48, 28, 31, 26, 39, 33, 42, 29, 35, 40, 47, 32, 44, 27, 38, 31, 49, 24, 45, 37, 46, 28, 40, 35, 50, 26, 41, 30, 44, 34, 48, 33, 47, 28, 39, 32, 46, 23, 38, 27, 44, 29, 36, 30, 45, 25, 50, 28, 47, 26, 39, 31, 43, 24, 38, 28, 48, 32, 41, 29, 50, 26, 44, 30, 47, 25, 49, 31, 42, 27, 45, 34, 50, 28, 39, 30, 47, 23, 36, 29, 50, 27, 43, 33, 46, 25, 41],
    'Gender': ['f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm', 'f', 'm'],
    'Sleep Quality': [8, 7, 9, 5, 8, 6, 8, 4, 9, 7, 5, 9, 8, 4, 9, 6, 8, 5, 9, 7, 8, 5, 9, 6, 8, 4, 9, 7, 8, 5, 9, 6, 8, 4, 9, 7, 8, 5, 9, 6, 8, 4, 9, 7, 8, 5, 9, 6, 8, 4, 9, 6, 8, 5, 9, 7, 8, 4, 9, 6, 8, 5, 9, 7, 8, 4, 9, 6, 8, 5, 9, 7, 8, 4, 9, 6, 8, 5, 9, 7, 8, 4, 9, 6, 8, 5, 9, 7, 8, 4, 9, 6, 8, 5, 9, 7, 8, 4, 9, 6],
    'Bedtime': ['23:00', '00:30', '22:45', '01:00', '23:30', '00:15', '22:30', '01:30', '23:00', '00:45', '01:00', '22:00', '23:30', '01:30', '22:15', '00:30', '23:45', '01:15', '22:30', '00:45', '23:15', '01:00', '22:45', '00:15', '23:00', '01:30', '22:30', '00:45', '23:30', '01:15', '22:00', '00:30', '23:15', '01:00', '22:45', '00:15', '23:30', '01:30', '22:00', '00:45', '23:15', '01:00', '22:45', '00:15', '23:30', '01:30', '22:00', '00:30', '23:15', '01:00', '22:15', '00:30', '23:00', '01:15', '22:30', '00:45', '23:15', '01:30', '22:15', '00:30', '23:00', '01:15', '22:30', '00:45', '23:15', '01:30', '22:15', '00:30', '23:00', '01:15', '22:30', '00:45', '23:15', '01:30', '22:15', '00:30', '23:00', '01:15', '22:30', '00:45', '23:15', '01:30', '22:15', '00:30', '23:00', '01:15', '22:30', '00:45', '23:15', '01:30', '22:15', '00:30', '23:00', '01:15', '22:30', '00:45', '23:15', '01:30', '22:15', '00:30'],
    'Wake-up Time': ['06:30', '07:00', '06:45', '06:30', '07:00', '07:15', '06:00', '07:00', '07:30', '07:15', '07:00', '06:00', '07:30', '06:00', '06:45', '07:30', '06:15', '07:00', '07:00', '07:30', '06:30', '06:30', '06:15', '07:00', '06:30', '07:00', '06:45', '07:15', '07:00', '07:30', '06:00', '07:30', '06:15', '06:30', '07:00', '07:15', '07:00', '06:00', '06:00', '07:30', '06:15', '06:30', '07:00', '07:15', '07:00', '06:00', '06:00', '07:30', '06:15', '06:30', '06:45', '07:00', '07:00', '07:00', '07:30', '07:15', '06:15', '07:00', '06:45', '07:00', '07:00', '07:00', '07:30', '07:15', '06:15', '06:00', '06:45', '07:00', '07:00', '07:00', '07:30', '07:15', '06:15', '07:00', '06:45', '07:00', '07:00', '07:00', '07:30', '07:15', '06:15', '07:00', '06:45', '07:00', '07:00', '07:00', '07:30', '07:15', '06:15', '07:00', '06:45', '07:00', '07:00', '07:00', '07:30', '07:15', '06:15', '07:00', '06:45', '07:00'],
    'Daily Steps': [8000, 5000, 9000, 4000, 10000, 6000, 8500, 3000, 9500, 6500, 3500, 11000, 7000, 3000, 9500, 6000, 8500, 4000, 10000, 5500, 9000, 4000, 9500, 6000, 8500, 3000, 10000, 6500, 9000, 3500, 10500, 5000, 8500, 3000, 10000, 6000, 9000, 3500, 10500, 5000, 8500, 3000, 10000, 6000, 9000, 3500, 10500, 5000, 8500, 3000, 9500, 5000, 8500, 4000, 9000, 6500, 8500, 3000, 9500, 5000, 8500, 4000, 9000, 6500, 8500, 3000, 9500, 5000, 8500, 4000, 9000, 6500, 8500, 3000, 9500, 5000, 8500, 4000, 9000, 6500, 8500, 3000, 9500, 5000, 8500, 4000, 9000, 6500, 8500, 3000, 9500, 5000, 8500, 4000, 9000, 6500, 8500, 3000, 9500, 5000],
    'Calories Burned': [2500, 2200, 2700, 2100, 2800, 2300, 2600, 2000, 2750, 2400, 2100, 2900, 2400, 2000, 2700, 2300, 2500, 2100, 2800, 2400, 2600, 2100, 2750, 2300, 2600, 2000, 2800, 2400, 2500, 2100, 2900, 2200, 2600, 2000, 2750, 2300, 2500, 2100, 2900, 2200, 2600, 2000, 2750, 2300, 2500, 2100, 2900, 2200, 2600, 2000, 2700, 2200, 2600, 2100, 2750, 2400, 2600, 2000, 2700, 2200, 2600, 2100, 2750, 2400, 2600, 2000, 2700, 2200, 2600, 2100, 2750, 2400, 2600, 2000, 2700, 2200, 2600, 2100, 2750, 2400, 2600, 2000, 2700, 2200, 2600, 2100, 2750, 2400, 2600, 2000, 2700, 2200, 2600, 2100, 2750, 2400, 2600, 2000, 2700, 2200],
    'Physical Activity Level': ['medium', 'low', 'high', 'low', 'high', 'medium', 'high', 'low', 'medium', 'medium', 'low', 'high', 'medium', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'medium', 'medium', 'high', 'low', 'medium', 'medium', 'high', 'low', 'high', 'medium', 'high', 'low', 'medium', 'medium', 'high', 'low', 'high', 'medium', 'high', 'low', 'medium', 'medium', 'high', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium', 'medium', 'low', 'high', 'medium', 'high', 'low', 'high', 'medium'],
    'Dietary Habits': ['healthy', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'healthy', 'medium', 'unhealthy', 'healthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'healthy', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'medium', 'unhealthy', 'healthy', 'unhealthy', 'medium', 'unhealthy', 'healthy', 'medium', 'medium', 'unhealthy', 'healthy', 'unhealthy'],
    'Sleep Disorders': ['no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no'],
    'Medication Usage': ['no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no']
}

df = pd.DataFrame(data)

def time_to_minutes(time_str):
    t = datetime.strptime(time_str, '%H:%M')
    return t.hour * 60 + t.minute

def analyze_nutrition(user_data):
    dietary_score = {
        'healthy': 3,
        'moderate': 2,
        'unhealthy': 1
    }
    
    score = dietary_score.get(user_data['Dietary Habits'].lower(), 2)
    
    if score == 3:
        return "Maintain healthy diet, focus on variety", []
    elif score == 2:
        return "Improve diet with more whole foods", ["Increase vegetable intake", "Reduce processed foods"]
    else:
        return "Major dietary overhaul needed", ["Limit sugar and processed foods", "Add lean proteins", "Increase fruits and vegetables"]
    

def analyze_activity(user_data):
    steps = user_data['Daily Steps']
    calories = user_data['Calories Burned']
    activity_level = user_data['Physical Activity Level'].lower()
    
    if activity_level == 'high' or steps > 8000:
        return "Excellent activity level", ["Maintain current activity", "Consider strength training"]
    elif activity_level == 'medium' or steps > 5000:
        return "Good activity level", ["Aim for 10,000 steps daily", "Add 2-3 cardio sessions weekly"]
    else:
        return "Low activity level", ["Start with 30-min daily walks", "Aim for 7000 steps daily"]
    
def analyze_sleep(user_data):
    sleep_quality = user_data['Sleep Quality']
    bedtime = time_to_minutes(user_data['Bedtime'])
    wakeup = time_to_minutes(user_data['Wake-up Time'])
    
    
    sleep_duration = (wakeup - bedtime) % (24 * 60) / 60.0
    
    recommendations = []
    if sleep_quality >= 8 and 7 <= sleep_duration <= 9:
        return "Good sleep quality", ["Maintain consistent sleep schedule"]
    else:
        recommendations.append("Aim for 7-9 hours of sleep")
        if sleep_quality < 7:
            recommendations.append("Create bedtime routine")
        if user_data['Sleep Disorders'] == 'yes':
            recommendations.append("Consult sleep specialist")
        return "Needs sleep improvement", recommendations

def generate_recommendations(user_id, df):
    user_data = df[df['User ID'] == user_id].iloc[0]
    
    nutrition_status, nutrition_recs = analyze_nutrition(user_data)
    activity_status, activity_recs = analyze_activity(user_data)
    sleep_status, sleep_recs = analyze_sleep(user_data)
    
    recommendations = {
        'Nutrition': {
            'Status': nutrition_status,
            'Recommendations': nutrition_recs
        },
        'Activity': {
            'Status': activity_status,
            'Recommendations': activity_recs
        },
        'Sleep': {
            'Status': sleep_status,
            'Recommendations': sleep_recs
        }
    }
    
    return recommendations

#Progress Check
def monitor_progress(user_id, old_data, new_data):
    old_user = old_data[old_data['User ID'] == user_id].iloc[0]
    new_user = new_data[new_data['User ID'] == user_id].iloc[0]
    
    progress = []
  
    steps_diff = new_user['Daily Steps'] - old_user['Daily Steps']
    if steps_diff > 0:
        progress.append(f"Increased daily steps by {steps_diff}")
    
    sleep_diff = new_user['Sleep Quality'] - old_user['Sleep Quality']
    if sleep_diff > 0:
        progress.append(f"Improved sleep quality by {sleep_diff} points")
    
    diet_score_map = {'unhealthy': 1, 'moderate': 2, 'healthy': 3}
    old_diet = diet_score_map.get(old_user['Dietary Habits'].lower(), 2)
    new_diet = diet_score_map.get(new_user['Dietary Habits'].lower(), 2)
    if new_diet > old_diet:
        progress.append("Improved dietary habits")
    
    return progress if progress else ["No significant changes detected"]

from sklearn.preprocessing import StandardScaler
# analyze_activity
def analyze_activity(user_data):
    steps = user_data['Daily Steps']
    calories = user_data['Calories Burned']
    activity_level = user_data['Physical Activity Level'].lower()
    
    if activity_level == 'high' or steps > 8000:
        return "Excellent activity level", ["Maintain current activity", "Consider strength training"]
    elif activity_level == 'medium' or steps > 5000:
        return "Good activity level", ["Aim for 10,000 steps daily", "Add 2-3 cardio sessions weekly"]
    else:
        return "Low activity level", ["Start with 30-min daily walks", "Aim for 7000 steps daily"]

# Sleep 
def analyze_sleep(user_data):
    sleep_quality = user_data['Sleep Quality']
    bedtime = time_to_minutes(user_data['Bedtime'])
    wakeup = time_to_minutes(user_data['Wake-up Time'])
    
    sleep_duration = (wakeup - bedtime) % (24 * 60) / 60.0
    
    recommendations = []
    if sleep_quality >= 8 and 7 <= sleep_duration <= 9:
        return "Good sleep quality", ["Maintain consistent sleep schedule"]
    else:
        recommendations.append("Aim for 7-9 hours of sleep")
        if sleep_quality < 7:
            recommendations.append("Create bedtime routine")
        if user_data['Sleep Disorders'] == 'yes':
            recommendations.append("Consult sleep specialist")
        return "Needs sleep improvement", recommendations

def cluster_users(df):
    features = df[['Age', 'Sleep Quality', 'Daily Steps', 'Calories Burned']].copy()
    
    df['Physical Activity Level'] = df['Physical Activity Level'].str.lower()
    df['Dietary Habits'] = df['Dietary Habits'].str.lower()
    
    activity_map = {'low': 1, 'medium': 2, 'high': 3}
    diet_map = {'unhealthy': 1, 'moderate': 2, 'healthy': 3}
    
    features['Physical Activity Score'] = df['Physical Activity Level'].map(activity_map)
    features['Diet Score'] = df['Dietary Habits'].map(diet_map)
    
    if features.isnull().any().any():
        print("NaN values found in features:")
        print(features.isnull().sum())
        imputer = SimpleImputer(strategy='median')
        features[['Age', 'Sleep Quality', 'Daily Steps', 'Calories Burned']] = imputer.fit_transform(
            features[['Age', 'Sleep Quality', 'Daily Steps', 'Calories Burned']]
        )
        imputer = SimpleImputer(strategy='most_frequent')
        features[['Physical Activity Score', 'Diet Score']] = imputer.fit_transform(
            features[['Physical Activity Score', 'Diet Score']]
        )
    
    # Standardize features for better clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    return df, scaled_features, kmeans


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters_sns(scaled_features, cluster_labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    
    plot_df = pd.DataFrame({
        'PC1': reduced_features[:, 0],
        'PC2': reduced_features[:, 1],
        'Cluster': cluster_labels
    })
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=60, alpha=0.7)
    
    plt.title('KMeans Clusters Visualized with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()
    return plt.gcf()

def main():
    
    clustered_df, scaled_features, kmeans = cluster_users(df)    
    plot_clusters_sns(scaled_features, clustered_df['Cluster'])
    
    for user_id in clustered_df['User ID'][:5]:
        recs = generate_recommendations(user_id, clustered_df)
        print(f"\nRecommendations for User {user_id}:")
        print(f"Cluster: {clustered_df[clustered_df['User ID'] == user_id]['Cluster'].iloc[0]}")
        for category, info in recs.items():
            print(f"\n{category}:")
            print(f"Status: {info['Status']}")
            print("Recommendations:")
            for rec in info['Recommendations']:
                print(f"- {rec}")

if __name__ == "__main__":
    main()
