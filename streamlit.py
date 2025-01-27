import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the cleaned and encoded data
encoded_data = pd.read_csv("encoded_data.csv")  # Load the encoded dataset
cleaned_data = pd.read_csv("cleaned_data.csv")  # Load the cleaned dataset

# Standardize the data (important before PCA)
scaler = StandardScaler()
encoded_data_scaled = scaler.fit_transform(encoded_data)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)  # Adjust n_components based on your needs
encoded_features_pca = pca.fit_transform(encoded_data_scaled)

# Step 3: Define the restaurant recommendation function
def recommend_restaurant(input_cuisine, input_city, encoded_data, cleaned_data, top_n=10, similarity_threshold=0.1):
    # Step 3.1: Find the index of the input cuisine and city
    input_data = cleaned_data[(cleaned_data['cuisine'] == input_cuisine) & (cleaned_data['city'] == input_city)]
    
    if input_data.empty:
        return "No matching input data found."
    
    # Get the input restaurant's encoded vector
    input_index = input_data.index[0]
    input_vector = encoded_data.iloc[input_index].values.reshape(1, -1)
    
    # Standardize the input vector using the same scaler
    input_vector_scaled = scaler.transform(input_vector)
    
    # Transform the input vector using PCA
    input_vector_pca = pca.transform(input_vector_scaled)
    
    # Step 3.2: Calculate cosine similarity between the input restaurant and all others
    similarity_scores = cosine_similarity(input_vector_pca, encoded_features_pca)
    
    # Step 3.3: Get the indices of the top N similar restaurants (above the similarity threshold)
    similar_restaurants = np.argsort(similarity_scores[0])[::-1]  # Sort indices by similarity (descending)
    
    # Filter restaurants that meet the similarity threshold
    recommended_restaurants = []
    for idx in similar_restaurants:
        if similarity_scores[0][idx] >= similarity_threshold:
            recommended_restaurants.append(cleaned_data.iloc[idx])
        if len(recommended_restaurants) >= top_n:
            break
    
    # Step 3.4: Return the recommended restaurants or a message if none meet the threshold
    if not recommended_restaurants:
        return "No restaurants meet the similarity threshold."
    
    return pd.DataFrame(recommended_restaurants), similarity_scores  # Return both recommended restaurants and similarity scores

# Streamlit Interface
st.title("Restaurant Recommendation System")

# User input for cuisine
input_cuisine = st.selectbox("Select Cuisine Type", cleaned_data['cuisine'].unique())

# User input for city - selectbox with all available cities
available_cities = cleaned_data['city'].unique()  # Get the list of unique cities
input_city = st.selectbox("Select City", available_cities)

# Button to get recommendations
if st.button("Get Recommendations"):
    if input_cuisine and input_city:
        recommended_restaurants, similarity_scores = recommend_restaurant(
            input_cuisine, input_city, encoded_data, cleaned_data, top_n=10, similarity_threshold=0.1)
        
        if isinstance(recommended_restaurants, pd.DataFrame):
            st.write("### Recommended Restaurants:")
            st.dataframe(recommended_restaurants[['name', 'rating', 'cost', 'cuisine', 'link']])
            
            st.write("### Similarity Scores:")
            st.write(similarity_scores)
            
            st.write("### PCA Features (Sample):")
            st.write(encoded_features_pca[:5])  # Display first 5 transformed features
        else:
            st.write(recommended_restaurants)
    else:
        st.write("Please provide both cuisine and city to get recommendations.")
