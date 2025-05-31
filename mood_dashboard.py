import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Mood Visualization Dashboard",
    page_icon="📊",
    layout="wide"
)

# Connect to the database
conn = sqlite3.connect('mood_tracker.db')

def get_user_mood_data(user_id=1, days=30):
    """Retrieve mood data for a specific user within the specified timeframe"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = """
    SELECT date, emotion, notes
    FROM mood_entries
    WHERE user_id = ? AND date >= ?
    ORDER BY date
    """
    
    df = pd.read_sql_query(
        query, 
        conn, 
        params=(user_id, start_date.strftime('%Y-%m-%d'))
    )
    
    # Convert date strings to datetime
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def get_emotion_distribution(user_id=1):
    """Get the distribution of emotions for a user"""
    query = """
    SELECT emotion, COUNT(*) as count
    FROM mood_entries
    WHERE user_id = ?
    GROUP BY emotion
    """
    
    return pd.read_sql_query(query, conn, params=(user_id,))

def get_daily_mood_counts(user_id=1, days=30):
    """Get the count of mood entries per day"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = """
    SELECT date, COUNT(*) as entry_count
    FROM mood_entries
    WHERE user_id = ? AND date >= ?
    GROUP BY date
    """
    
    return pd.read_sql_query(
        query, 
        conn, 
        params=(user_id, start_date.strftime('%Y-%m-%d'))
    )

def get_mood_streaks(user_id=1):
    """Calculate longest streak of consecutive days with mood entries"""
    query = """
    SELECT date FROM mood_entries
    WHERE user_id = ?
    GROUP BY date
    ORDER BY date
    """
    
    dates_df = pd.read_sql_query(query, conn, params=(user_id,))
    
    if dates_df.empty:
        return 0, 0
    
    # Convert to datetime
    dates_df['date'] = pd.to_datetime(dates_df['date'])
    
    # Sort dates
    dates = sorted(dates_df['date'].tolist())
    
    if not dates:
        return 0, 0
    
    # Calculate streaks
    streaks = []
    current_streak = 1
    max_streak = 1
    
    for i in range(1, len(dates)):
        # If consecutive days
        if (dates[i] - dates[i-1]).days == 1:
            current_streak += 1
        else:
            streaks.append(current_streak)
            current_streak = 1
    
    # Add the last streak
    streaks.append(current_streak)
    
    # Calculate max streak
    max_streak = max(streaks) if streaks else 0
    
    # Check if current streak is active (includes today)
    current_active_streak = 0
    today = datetime.now().date()
    
    if dates and (today - dates[-1].date()).days == 0:
        # Start from the most recent date
        current_active_streak = 1
        for i in range(len(dates)-1, 0, -1):
            if (dates[i] - dates[i-1]).days == 1:
                current_active_streak += 1
            else:
                break
    
    return max_streak, current_active_streak

def main():
    # Add title
    st.title("📊 Mood Visualization Dashboard")
    
    # Sidebar for filters
    with st.sidebar:
        st.header("Dashboard Settings")
        
        # Time range selector
        time_range = st.radio(
            "Select Time Range:",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
            index=1  # Default to 30 days
        )
        
        # Map selection to days
        days_mapping = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "All Time": 3650  # ~10 years
        }
        
        days = days_mapping[time_range]
        
        # User selector (for admin view - can be hidden for regular users)
        # In a real app, you would restrict this based on login permissions
        st.subheader("User Selection")
        user_id = st.number_input("User ID", min_value=1, value=1)
        
        # Refresh button
        if st.button("Refresh Data"):
            st.success("Data refreshed!")
    
    # Get data based on filters
    mood_data = get_user_mood_data(user_id, days)
    emotion_dist = get_emotion_distribution(user_id)
    daily_counts = get_daily_mood_counts(user_id, days)
    max_streak, current_streak = get_mood_streaks(user_id)
    
    # Display metrics in the top row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_entries = len(mood_data) if not mood_data.empty else 0
        st.metric("Total Mood Entries", total_entries)
    
    with col2:
        if not emotion_dist.empty:
            most_common = emotion_dist.loc[emotion_dist['count'].idxmax()]
            st.metric("Most Common Emotion", most_common['emotion'])
        else:
            st.metric("Most Common Emotion", "No data")
    
    with col3:
        st.metric("Longest Streak", f"{max_streak} days")
    
    with col4:
        st.metric("Current Streak", f"{current_streak} days")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Emotion Timeline", "Emotion Distribution", "Daily Activity"])
    
    with tab1:
        st.subheader("Your Mood Timeline")
        
        if not mood_data.empty:
            # Create a continuous date range
            date_range = pd.date_range(
                start=mood_data['date'].min(),
                end=mood_data['date'].max()
            )
            
            # Create a dataframe with all dates
            all_dates = pd.DataFrame({'date': date_range})
            
            # Merge with mood data
            merged = pd.merge(all_dates, mood_data, on='date', how='left')
            
            # Create emotion mapping for visualization
            emotions = emotion_dist['emotion'].unique() if not emotion_dist.empty else []
            emotion_mapping = {emotion: i for i, emotion in enumerate(emotions)}
            
            # Map emotions to numeric values for the line chart
            if not merged.empty and 'emotion' in merged.columns:
                merged['emotion_code'] = merged['emotion'].map(emotion_mapping)
                
                # Filter out rows with no emotion
                valid_moods = merged.dropna(subset=['emotion'])
                
                if not valid_moods.empty:
                    # Create the line chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=valid_moods['date'],
                        y=valid_moods['emotion_code'],
                        mode='lines+markers',
                        marker=dict(size=10),
                        text=valid_moods['emotion'],
                        hovertemplate='%{text}<extra></extra>',
                        line=dict(width=2)
                    ))
                    
                    # Customize the y-axis to show emotion labels
                    fig.update_layout(
                        yaxis=dict(
                            tickmode='array',
                            tickvals=list(emotion_mapping.values()),
                            ticktext=list(emotion_mapping.keys())
                        ),
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=400,
                        title="Emotion Trends Over Time"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data to show a timeline.")
            else:
                st.info("No mood data available for the selected time period.")
        else:
            st.info("No mood data available. Start logging your emotions to see trends.")
    
    with tab2:
        st.subheader("Emotion Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not emotion_dist.empty:
                # Create a pie chart
                fig = px.pie(
                    emotion_dist,
                    values='count',
                    names='emotion',
                    title="Distribution of Emotions",
                    color='emotion',
                    hole=0.4
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emotion data available for the selected filters.")
        
        with col2:
            if not emotion_dist.empty:
                # Create a bar chart
                fig = px.bar(
                    emotion_dist,
                    x='emotion',
                    y='count',
                    title="Count of Each Emotion",
                    color='emotion'
                )
                
                fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emotion data available for the selected filters.")
    
    with tab3:
        st.subheader("Daily Mood Activity")
        
        if not daily_counts.empty:
            # Convert date strings to datetime if needed
            if not isinstance(daily_counts['date'].iloc[0], datetime):
                daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            # Create a heatmap calendar view
            # Group by month and day for the heatmap
            daily_counts['month'] = daily_counts['date'].dt.month_name()
            daily_counts['day'] = daily_counts['date'].dt.day
            
            fig = px.density_heatmap(
                daily_counts,
                x='day',
                y='month',
                z='entry_count',
                title="Mood Entry Calendar",
                labels={'day': 'Day of Month', 'month': 'Month', 'entry_count': 'Number of Entries'},
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis=dict(tickmode='linear', dtick=1),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Daily entry count line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_counts['date'],
                y=daily_counts['entry_count'],
                mode='lines+markers',
                name='Daily Entries',
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title="Daily Mood Entry Counts",
                xaxis_title="Date",
                yaxis_title="Number of Entries",
                margin=dict(l=20, r=20, t=40, b=20),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily counts data available for the selected time period.")
    
    # Additional section for mood notes/journal entries
    st.subheader("Recent Mood Journal Entries")
    
    if not mood_data.empty:
        # Sort by date descending
        recent_entries = mood_data.sort_values('date', ascending=False).head(5)
        
        for _, entry in recent_entries.iterrows():
            date_str = entry['date'].strftime("%A, %B %d, %Y")
            emotion = entry['emotion']
            
            # Display each entry in an expander
            with st.expander(f"{date_str} - {emotion}"):
                if pd.notna(entry['notes']) and entry['notes']:
                    st.write(entry['notes'])
                else:
                    st.write("No additional notes for this entry.")
    else:
        st.info("No journal entries available. Start logging your emotions with notes to see them here.")
    
    # Close database connection
    conn.close()

if __name__ == "__main__":
    main()
