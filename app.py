import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import openai

# Set up OpenAI (safely reference secret)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Title
st.title("AI-Powered Promotion Engine")

#
st.subheader("Upload Customer Data")
uploaded_file = st.file_uploader("Upload a CSV file with columns: customer_id, last_purchase_amount, visits_last_month, category_interest")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Step 1: Segment customers
    features = data[["last_purchase_amount", "visits_last_month"]]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    data["segment"] = kmeans.labels_

    # Step 2: Rule-based promotions
    def select_offer(segment, interest):
        if segment == 1 and interest == "electronics":
            return "20% off on your next electronics purchase!"
        elif segment == 0:
            return "Flat ₹100 cashback on orders above ₹999!"
        return "Free shipping for all orders this week!"

    data["promotion"] = data.apply(lambda row: select_offer(row["segment"], row["category_interest"]), axis=1)

    # Step 3: Generate promo messages
    st.subheader("Generating Promotional Messages...")
    progress = st.progress(0)
    promo_msgs = []
    for i, row in data.iterrows():
        prompt = f"Write a promotional message for customer {row['customer_id']} offering: '{row['promotion']}'"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            message = response.choices[0].message["content"]
        except Exception as e:
            message = f"[Error generating message] {e}"
        promo_msgs.append(message)
        progress.progress((i + 1) / len(data))

    data["promo_message"] = promo_msgs

    # Show output
    st.subheader("Generated Promotions")
    st.dataframe(data[["customer_id", "promotion", "promo_message"]])

    # Download
    st.download_button("Download Results as CSV", data.to_csv(index=False), "promotions.csv")
else:
    st.info("Awaiting file upload...")
