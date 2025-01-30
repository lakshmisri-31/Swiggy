import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Load the preprocessed data
encoded_data = pd.read_csv("encoded_data.csv")
cleaned_data = pd.read_csv("cleaned_data.csv")

# Standardize and apply PCA to the encoded data
scaler = StandardScaler()
encoded_data_scaled = scaler.fit_transform(encoded_data)

# Applying PCA for dimensionality reduction
pca = PCA(n_components=10)  # Adjust n_components based on your needs
encoded_features_pca = pca.fit_transform(encoded_data_scaled)

# Define the recommendation function
def recommend_restaurant(input_cuisine, input_city, encoded_data, cleaned_data, top_n=10, similarity_threshold=0.1):
    input_data = cleaned_data[(cleaned_data['cuisine'] == input_cuisine) & (cleaned_data['city'] == input_city)]
    
    if input_data.empty:
        return "No matching input data found."
    
    input_index = input_data.index[0]
    input_vector = encoded_data.iloc[input_index].values.reshape(1, -1)
    
    input_vector_scaled = scaler.transform(input_vector)
    
    input_vector_pca = pca.transform(input_vector_scaled)
    
    similarity_scores = cosine_similarity(input_vector_pca, encoded_features_pca)
    
    similar_restaurants = np.argsort(similarity_scores[0])[::-1]
    
    recommended_restaurants = []
    for idx in similar_restaurants:
        if similarity_scores[0][idx] >= similarity_threshold:
            recommended_restaurants.append(cleaned_data.iloc[idx])
        if len(recommended_restaurants) >= top_n:
            break
    
    if not recommended_restaurants:
        return "No restaurants meet the similarity threshold."
    
    return pd.DataFrame(recommended_restaurants)

# Streamlit UI
st.title("Restaurant Recommendation System")
st.write("Select your preferences below:")

# Get user input for cuisine and city
input_cuisine = st.selectbox("Choose a Cuisine", cleaned_data['cuisine'].unique())
input_city = st.selectbox("Choose a City", cleaned_data['city'].unique())

# Button to get recommendations
if st.button("Get Recommendations"):
    recommended_restaurants = recommend_restaurant(input_cuisine, input_city, encoded_data, cleaned_data)
    
    if isinstance(recommended_restaurants, str):
        st.write(recommended_restaurants)  # Display message if no data is found
    else:
        st.write("Top 10 Recommended Restaurants:")
        st.dataframe(recommended_restaurants)  # Display the recommendations in a table

