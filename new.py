import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import io
from PIL import Image
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import xgboost as xgb
from sklearn.cluster import KMeans

df = pd.read_excel("Dataset.xlsx")
df2 =df
buffer = io.StringIO()
df.info(buf=buffer)
summ = df.isnull().sum()
df = df[df['Rating'].notna()]
df = df[df['Country'].notna()]
df.drop(['SolutionCount', 'Name', 'TotalActiveDays'], axis=1, inplace=True)
df['PostViewCount'] = df['PostViewCount'].fillna(0)
df['Reputation'] = df['Reputation'].fillna(0)
df['AttendedContestsCount'] = df['AttendedContestsCount'].fillna(0)
df['Total'] = df['Total'].fillna(0)
df['Hard'] = df['Hard'].fillna(0)
df['Streak'] = df['Streak'].fillna(0)
df['Easy'] = df['Easy'].fillna(0)
df['Medium'] = df['Medium'].fillna(0)
df['Badge'] = df['Badge'].fillna("No Badges")
df['ActiveYears'] = df['ActiveYears'].astype(str)
active_years = df['ActiveYears'].str.split(',')
df['NumActiveYears'] = active_years.apply(len)
df.drop(['ActiveYears'], axis=1, inplace=True)

# Define function to create animation
def animated_loading():
    with st.empty():
        for seconds in range(1):
            st.write("Loading...")
            latest_iteration = st.empty()
            bar = st.progress(0)
            for i in range(100):
                latest_iteration.text(f'Loading {i + 1}%')
                bar.progress(i + 1)
                time.sleep(0.01)
            latest_iteration.text("")
    # st.balloons()


# Define function for the home page
def home():
    col1, col2 = st.columns(2)
    url = requests.get(
        "https://assets4.lottiefiles.com/packages/lf20_49rdyysj.json")
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")

    with col1:
        st_lottie(url_json,
                  # change the direction of our animation
                  reverse=True,
                  # height and width of animation
                  height=400,
                  width=400,
                  # speed of animation
                  speed=1,
                  # means the animation will run forever like a gif, and not as a still image
                  loop=True,
                  # quality of elements used in the animation, other values are "low" and "medium"
                  quality='high',
                  # THis is just to uniquely identify the animation
                  key='Home'
                  )

    with col2:
        st.markdown("<h1 style='text-align:right; color: lightskyblue;'>Leetcode Data Analysis"
                    "</h1>""<h5 style='text-align: right; color: corns-ilk;'>Rohit Khandal -202011064 "
                    "</h5>""<h5 style='text-align: right; color: cornsilk;'>Vishnu Swaroop -202011037"
                    "</h5>""<h5 style='text-align: right; color: cornsilk;'>Ishant Bisen -202011028 "
                    "</h5>""<h5 style='text-align: right; color: cornsilk;'>Vivek Borole -202011018"
                    "</h5>""<h5 style='text-align: right; color: cornsilk;'>Gurupal Singh -202011022"
                    "</h5>", unsafe_allow_html=True)

    st.markdown("***")
    st.markdown("<h1 style='text-align:center; color: lightskyblue;'>Data Science Life Cycle""</h1>",
                unsafe_allow_html=True)
    image = Image.open("data_cycle.png")
    st.write("")
    st.write("")
    st.image(image, width=700)
    st.write("")
    st.write("")
    st.markdown("***")

    st.markdown("<h1 style='text-align:center; color: lightskyblue;'>Team Contribution""</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        url = requests.get(
            "https://assets10.lottiefiles.com/packages/lf20_fclga8fl.json")
        url_json = dict()
        if url.status_code == 200:
            url_json = url.json()
        else:
            print("Error in URL")
        st_lottie(url_json,
                  # change the direction of our animation
                  reverse=True,
                  # height and width of animation
                  height=400,
                  width=350,
                  # speed of animation
                  speed=1,
                  # means the animation will run forever like a gif, and not as a still image
                  loop=True,
                  # quality of elements used in the animation, other values are "low" and "medium"
                  quality='high',
                  # THis is just to uniquely identify the animation
                  key='team'
                  )

    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("- **:cornsilk[Data Collection :- Vishnu Swaroop, Rohit Khandal]**")
        st.markdown("- **:cornsilk[Data Wrangling :- Vivek Borole, Gurupal Singh]**")
        st.markdown("- **:cornsilk[EDA :- Rohit Khandal, Ishant Bisen]**")
        st.markdown("- **:cornsilk[Model Building:- Rohit Khandal]**")
        st.markdown("- **:cornsilk[Model Deployment:- Vivek Borole]**")


def data_collect():
    global df2
    st.markdown("<h2 style='text-align: center; color: lightskyblue;'>Data Collection ""</h2>", unsafe_allow_html=True)
    st.markdown("***")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("Kaggle leetcode username Dataset [link](https://www.kaggle.com/datasets/nidhaypancholi/leetcode-indian-user-ratings)")
    st.markdown("- Obtained Username Dataset from above link")
    st.markdown("- Found Leetcode's GraphqlAPI")
    st.markdown("- Generated Our Own Datased Using API")
    st.markdown("- Concated the data of every person and converted into XL")

    st.code('''for (let i = 1; i <= 14000; i++) {
  const cell = worksheet['B' + i]
  if (cell && cell.v) {
    console.log(cell.v)

    const query = {
      query: `
        query languageStats($username: String!) {
          matchedUser(username: $username) {
            languageProblemCount {
              languageName
              problemsSolved
            }
          }
        }
      `,
      variables: {
        // Here cell.v gives every username in the dataset of users
        username: `${cell.v}`,
      },
      operationName: 'languageStats',
    }

    const response = await axios.post('https://leetcode.com/graphql/', query)

    const Data = {
      sno: val,
      username: cell.v,
      response,
    }

    const newWorkbook = XLSX.utils.book_new()
    const newSheet = XLSX.utils.json_to_sheet(Data)
    XLSX.utils.book_append_sheet(newWorkbook, newSheet, Data)

    XLSX.writeFile(
      newWorkbook,
      'D://Clg/SEM6/CS312/Project/Output/Dataset.xlsx',
    )
  }
}''',language='javascript')
    st.markdown("***")
    st.markdown("<h1 style='text-align: center; color: lightskyblue;'>Data ""</h1>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.dataframe(df2, use_container_width=True)


def data_wrangle1():
    global df
    global buffer
    global summ
    # df = pd.read_excel("Dataset.xlsx")
    col1, col2 = st.columns(2)
    with col1:
        url = requests.get(
            "https://assets1.lottiefiles.com/packages/lf20_EKUtIRMvz5.json")
        url_json = dict()
        if url.status_code == 200:
            url_json = url.json()
        else:
            print("Error in URL")
        st_lottie(url_json,
                  # change the direction of our animation
                  reverse=True,
                  # height and width of animation
                  height=250,
                  width=250,
                  # speed of animation
                  speed=1,
                  # means the animation will run forever like a gif, and not as a still image
                  loop=True,
                  # quality of elements used in the animation, other values are "low" and "medium"
                  quality='high',
                  # THis is just to uniquely identify the animation
                  key='clean'
                  )
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("<h1 style='text-align: left; color: lightskyblue;'>Data Cleaning "
                    "</h1>", unsafe_allow_html=True)
    st.markdown("***")
    st.write("")
    st.write("")
    st.write("")

    st.markdown("<h3 style='text-align: center; color: mediumseagreen;'>Data Description "
                "</h3>", unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)
    st.markdown("***")
    s = buffer.getvalue()
    st.markdown("<h3 style='text-align: center; color: mediumseagreen;'>Checking for Null values "
                "</h3>", unsafe_allow_html=True)
    st.text(s)
    st.markdown("***")
    st.markdown("<h3 style='text-align: center; color: mediumseagreen;'>Total Null Values "
                "</h3>", unsafe_allow_html=True)
    st.text(summ)
    st.markdown("***")
    st.markdown("<h1 style='text-align: center; color: mediumseagreen;'>Data Cleaning "
                "</h1>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center; color: wheat;'>Fill Null Values "
                "</h5>", unsafe_allow_html=True)
    st.code(
        '''df = df[df['Rating'].notna()]
df = df[df['Country'].notna()]
df['PostViewCount'] = df['PostViewCount'].fillna(0)
df['Reputation'] = df['Reputation'].fillna(0)
df['AttendedContestsCount'] = df['AttendedContestsCount'].fillna(0)
df['Total'] = df['Total'].fillna(0)
df['Hard'] = df['Hard'].fillna(0)
df['Streak'] = df['Streak'].fillna(0)
df['Easy'] = df['Easy'].fillna(0)
df['Medium'] = df['Medium'].fillna(0)''', language='python')

    st.markdown("<h5 style='text-align: center; color: wheat;'>Feature Engineering "
                "</h5>", unsafe_allow_html=True)
    st.code(
        '''df['Badge'] = df['Badge'].fillna("No Badges") 
df['ActiveYears'] = df['ActiveYears'].astype(str)
active_years = df['ActiveYears'].str.split(',')
df['NumActiveYears'] = active_years.apply(len)
df.drop(['ActiveYears'], axis=1, inplace=True)''', language='python')

    st.markdown("***")
    st.markdown("<h5 style='text-align: center; color: wheat;'>Cleaned Data Info"
                "</h5>", unsafe_allow_html=True)
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.markdown("***")

# Define function for the correlation analysis page

# here





def correlation_analysis():
    global df
    st.markdown("<h1 style='text-align: left; color: lightskyblue;'>Correlation Matrix "
                "</h1>", unsafe_allow_html=True)
    # fig, ax = plt.subplots()
    plt.figure(figsize=(10, 6))
    corr_df = df.drop(['Username', 'Country', 'TotalParticipants', 'Badge'], axis=1).corr(method='pearson')
    fig = px.imshow(corr_df, color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect="auto")
    fig.update_layout(
        height=800,
        width=800
    )
    st.write(fig)

    st.markdown("***")

    # fig, ax = plt.subplots()
    sample_df = df.sample(int(0.04 * len(df)))
    fig = plt.figure(figsize=(10, 6))
    fig = px.scatter(
        sample_df, x='Total', y='Medium', opacity=0.65,
        trendline='ols', trendline_color_override='darkblue'
        , title="Total vs Medium Regression")
    st.write(fig)
    st.markdown("***")
    fig = plt.figure(figsize=(10, 6))
    fig = px.scatter(
        df, x='PostViewCount', y='Reputation', opacity=0.65,
        trendline='ols', trendline_color_override='darkblue'
        , title="PostViewCount vs Reputation Regression")
    st.write(fig)

    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown(
        "- Skills development The high correlation suggests that users who attempt a larger number of medium type"
        " questions on LeetCode are likely working to develop their programming skills beyond the basics."
        " As users become more proficient in coding, they may seek out more challenging problems to solve, which could explain the high correlation.")
    st.markdown(
        "- Career preparation: The high correlation may indicate that users who are preparing for coding interviews or seeking to advance their careers in software development may focus more heavily on solving medium difficulty problems,"
        " which are more representative of the types of problems encountered in industry interviews.")
    st.markdown(
        "- Quality of solutions: The high correlation suggests that users who have a higher reputation on LeetCode may be more likely to have higher quality solutions, as indicated by the number of upvotes. This may in turn attract more views to their user profile, as other users seek out high quality solutions to difficult problems.")
    st.markdown(
        "- Networking and career opportunities: Finally, the high correlation may suggest that users with a high reputation and large number of views on their user profile may have more networking and career opportunities within the software development industry. ")


def distribution():
    global df
    st.markdown("***")
    st.markdown("<h2 style='text-align: center; color: lightskyblue;'>Distribution PLots "
                "</h2>", unsafe_allow_html=True)
    st.markdown("***")
    total = [df['Total'].to_numpy()]
    easy = [df['Easy'].to_numpy()]
    medium = [df["Medium"].to_numpy()]
    hard = [df["Hard"].to_numpy()]

    # group_labels = ["Total"]
    st.write("")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Distribution of Total Question "
                "</h4>", unsafe_allow_html=True)
    fig = px.histogram(df, x="Total", nbins=80, histnorm="density", marginal="violin",
                       color_discrete_sequence=["Crimson"])
    # fig = ff.create_distplot(total, ["Total"], bin_size=1.1, colors=['Crimson'])
    fig.update_layout(height=600, width=800)
    st.write(fig)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Distribution of Easy Question "
                "</h4>", unsafe_allow_html=True)
    fig = px.histogram(df, x="Easy", nbins=80, histnorm="density", marginal="violin",
                       color_discrete_sequence=["peru"])
    # fig = ff.create_distplot(easy, ["Easy"], bin_size=1.1, colors=['olive'])
    fig.update_layout(height=600, width=800)
    st.write(fig)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Distribution of Medium Question "
                "</h4>", unsafe_allow_html=True)
    fig = px.histogram(df, x="Medium", nbins=80, histnorm="density", marginal="violin",
                       color_discrete_sequence=["lightblue"])
    # fig = ff.create_distplot(medium, ["Medium"], bin_size=1.1, colors=['azure'])
    fig.update_layout(height=600, width=800)
    st.write(fig)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Difficulty levels: The fact that the distributions for medium and easy type questions are normally distributed with a slight right skew suggests that users are generally attempting questions across different levels of difficulty on LeetCode. The distribution of hard type questions, on the other hand, being exponential, suggests that users are not attempting as many hard type questions, which may be more challenging or time-consuming to solve.")
    st.markdown("- Learning strategies: The distribution of question attempts across difficulty levels could indicate that users are strategically choosing questions to solve based on their skill level and learning goals. For example, users may be attempting more easy and medium type questions to build foundational skills before tackling more challenging problems.")
    st.write("")
    st.write("")
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Distribution of Hard Question "
                "</h4>", unsafe_allow_html=True)
    fig = px.histogram(df, x="Hard", nbins=80, histnorm="density",
                       marginal="violin",
                       color_discrete_sequence=["darkkhaki"])
    # fig = ff.create_distplot(hard, ["Hard"], bin_size=1.1, colors=['darkkhaki'],curve_type='kde')
    fig.update_layout(height=600, width=800)
    st.write(fig)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown(
        "- Community focus: The fact that users are attempting questions across different levels of difficulty suggests that LeetCode's community is not focused solely on difficult challenges. Users may be motivated by a range of factors, such as learning, competition, or simply the satisfaction of solving coding challenges.")
    st.markdown(
        "- Specialization: The exponential distribution of hard type question attempts could suggest that some users on LeetCode may specialize in particularly difficult or niche areas of programming. These users may be more interested in solving challenging problems and may have developed expertise in these areas.")

    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Density Plot of Rating "
                "</h4>", unsafe_allow_html=True)
    fig = px.histogram(df, x="Rating", nbins=80, histnorm="density",
                       marginal="violin",
                       color_discrete_sequence=["olive"])
    st.write(fig)
    st.markdown("***")

    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Competition and rankings: The density plot suggests that there may be a significant amount of competition and emphasis on rankings among LeetCode users. Users may be motivated to improve their ratings and achieve higher rankings on the platform, which may be an important factor for them.")    # st.markdown("- Learning strategies: The distribution of question attempts across difficulty levels could indicate that users are strategically choosing questions to solve based on their skill level and learning goals. For example, users may be attempting more easy and medium type questions to build foundational skills before tackling more challenging problems.")
    st.markdown("- Learning and improvement: The high density of users around the 1500-1550 rating range could also suggest that this is a common rating level for users who are actively learning and improving their programming skills on LeetCode. Users at this rating range may be more focused on improving their skills than achieving high rankings, which could explain why they are not as spread out across different rating levels.")    # st.markdown("- Specialization: The exponential distribution of hard type question attempts could suggest that some users on LeetCode may specialize in particularly difficult or niche areas of programming. These users may be more interested in solving challenging problems and may have developed expertise in these areas.")


def question_badges():
    global df
    group = df.groupby("Badge", as_index=False).mean()
    # st.write("This page shows the geographic distribution of LeetCode users.")
    st.markdown("<h2 style='text-align: center; color: mediumseagreen;'>Number of Badges "
                "</h2>", unsafe_allow_html=True)
    host = df['Badge'].value_counts(ascending=False)
    fig = px.bar(x=host, y=host.index, orientation='h', color=host.values,
                 color_continuous_scale=px.colors.sequential.Viridis)
    st.write(fig)
    st.write("")
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Distribution of users: The fact that the majority of users (13389 out of 16000) do not have any badges suggests that the badge system may not be a significant factor for most users on LeetCode. This could be because users are more focused on improving their skills or achieving specific goals on the platform rather than earning badges. However, the fact that there are still a significant number of users with badges (401 with guardian badge and 1885 with knight badge) suggests that the badge system is still relevant for some users and may be an important factor for certain use cases, such as hiring or recruiting.")
    st.markdown("***")
    # Data
    # group = df.groupby("Badge", as_index=False).mean()
    class_1 = group["Total"].tolist()
    class_2 = group["Easy"].tolist()
    class_3 = group["Medium"].tolist()
    class_4 = group["Hard"].tolist()

    categories = ['Guardian', 'Knight', 'No Badges']

    st.markdown("<h2 style='text-align: center; color: mediumseagreen;'>Categorize number of Question "
                "</h2>", unsafe_allow_html=True)
    trace1 = go.Bar(x=categories, y=class_1, name='Total')
    trace2 = go.Bar(x=categories, y=class_2, name='Easy')
    trace3 = go.Bar(x=categories, y=class_3, name='Medium')
    trace4 = go.Bar(x=categories, y=class_4, name='Hard')
    layout = go.Layout(xaxis=dict(title='Category'), yaxis=dict(title='Value'))

    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)

    # Show the figure
    st.write(fig)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Skill level: The fact that users with guardian badges have the highest average number of total questions suggests that these users may be more advanced or experienced than users with knight badges or no badges. Similarly, users with guardian badges also have the highest average number of hard type questions, which further supports this hypothesis.")
    st.markdown("- Motivation and dedication: The fact that users with guardian badges have higher average numbers of questions across all categories (total, easy, medium, and hard) could suggest that these users are highly motivated and dedicated to improving their skills on LeetCode. They may be more likely to spend more time and effort on the platform, which could explain their higher question counts.")
    st.markdown("- Importance of badges: The fact that users with guardian badges have significantly higher average question counts than users with knight badges or no badges suggests that the badge system on LeetCode may be relevant for some users. Users may be motivated to earn badges as a way to showcase their skills and accomplishments, which could lead to better opportunities or recognition in the industry. However, it's important to note that not all users may be equally motivated by badges, as evidenced by the fact that there are still a significant number of users with no badges who have high question counts.")

# Define function for the badge analysis page
def three_d():
    # st.write("This page shows the distribution of different types of badges.")
    global df
    st.markdown("<h2 style='text-align: center; color: lightskyblue;'>Clustering "
                "</h2>", unsafe_allow_html=True)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Using Streak & Total Contest Attended "
                "</h4>", unsafe_allow_html=True)
    # # Select columns for clustering
    X = df[['Rating', 'AttendedContestsCount', 'Streak']]
    # # Perform clustering with K-Means algorithm
    #
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    #
    # # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Plot the clusters
    fig = px.scatter_3d(df, x='Rating', y='AttendedContestsCount', z='Streak', color='Cluster')
    st.plotly_chart(fig, use_container_width=True)
    # Insert code for creating the chart here
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Dependence of rating on streak and contest attendance: By clustering users based on streak, rating, and contest attendance, you can identify groups of users with similar characteristics. For example, you may find that users with higher contest attendance and longer streaks tend to have higher ratings. By examining the centroid (or mean) values for each cluster, you can quantify the average streak, rating, and contest attendance for each group and identify any patterns or trends.")
    st.markdown("- Importance of contest attendance: The fact that contest attendance is one of the variables used for clustering suggests that it may be an important factor for determining a user's streak and rating on LeetCode. Users who participate in more contests may be more likely to have longer streaks and higher ratings, which could indicate a higher level of dedication or skill.")
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Using Total & number of active years"
                "</h4>", unsafe_allow_html=True)
    X = df[['Rating', 'Total', 'NumActiveYears']]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    df['Cluster'] = kmeans.labels_

    # Plot the clusters
    fig = px.scatter_3d(df, x='Rating', y='Total', z='NumActiveYears', color='Cluster')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Quality of solutions: It's possible that a user's rating is not strongly correlated with the number of questions answered or the length of active participation, but is instead influenced more by the quality of their solutions. Users who consistently provide high-quality solutions may receive more upvotes and have a higher rating, regardless of how many questions they have answered or how long they have been active.")
    st.markdown("***")

def pair_plot():
    st.markdown("<h2 style='text-align: center; color: lightskyblue;'>Pair Plot "
                "</h2>", unsafe_allow_html=True)
    st.markdown("***")
    sns.set_style("darkgrid")
    fig = plt.figure()
    new_df = df.drop(
        ["Sno", "AttendedContestsCount", "PostViewCount", "GlobalRanking", "Country", "PostViewCount",
         "Reputation", "Streak", "TotalParticipants"], axis=1)
    pairplot_figure = sns.pairplot(new_df, hue="Badge")
    # pairplot_figure.fig.set_size_inches(9, 6.5)
    st.pyplot(pairplot_figure.fig)
    st.markdown("***")
    st.markdown("<h4 style='text-align: center; color: mediumseagreen;'>Insight"
                "</h4>", unsafe_allow_html=True)
    st.markdown("- Differences between easy, medium, and hard questions: The pair plot shows that there are differences in the distribution of easy, medium, and hard questions answered, with some users answering more of one type of question than the others. This suggests that users may have different strengths and weaknesses when it comes to different types of questions on LeetCode.")
    st.markdown("- Active years and rating: The pair plot shows a weak positive correlation between the number of active years and a user's rating. This suggests that users who have been active on LeetCode for a longer period of time may have higher ratings, although other factors such as the quality of their solutions may play a more important role.")


def predict_rating():
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')

    # Caching the model for faster loading
    @st.cache_data
    def predict(total, easy, medium, hard, reputation, streak, attended):
        prediction = model.predict(pd.DataFrame([[total, easy, medium, hard, reputation, streak, attended]],
                                                columns=['Total', 'Easy', 'Medium', 'Hard', 'Reputation', 'Streak',
                                                         'AttendedContestsCount']))
        return prediction

    st.markdown("<h1 style='text-align: center; color: mediumseagreen;'>🔮 Predict Rating 🔮"
                "</h1>", unsafe_allow_html=True)
    st.markdown("***")

    with st.form("my_form"):
        # st.write("Inside the form")
        total = st.slider("Total Question",min_value=0,max_value=2617)
        easy = st.slider("Total Easy Question",min_value=0,max_value=640)
        medium = st.slider("Total Medium Question",min_value=0,max_value=1392)
        hard = st.slider("Total Hard Question",min_value=0,max_value=585)
        reputation = st.slider("Total Reputation",min_value=0,max_value=26798)
        streak = st.slider("Maximum Streak",min_value=0,max_value=365)
        attended  = st.slider("Total Contest Ateended",min_value=0,max_value=309)



        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            rate = predict(total, easy, medium, hard, reputation, streak, attended)
            st.success(f'The predicted value of the Rating is {rate[0]:.2f}')

def deploy():
    st.markdown("<h4 style='text-align: center; color: lightskyblue;'>you can visit our web app from the below link "
                "</h4>", unsafe_allow_html=True)
    # st.write("[link](https://vivek-borole-leetcode-data-analysis-new-hn2yrb.streamlit.app)")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    col1, col2,col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        st.write("Deployed Link on Streamlit Server : - [link](https://vivek-borole-leetcode-data-analysis-new-hn2yrb.streamlit.app)")
    with col3:
        st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("<h1 style='text-align: center; color: lightskyblue;'> !! Thank You !!"
                "</h1>", unsafe_allow_html=True)
# Create navigation bar
nav = st.sidebar.radio("Leetcode ", ["Introduction", "Data Collection", "Data Wrangling", "EDA", "Predict Rating","Deploy"])
# "Distribution Analysis", "Correlation Analysis", "Trend Analysis", "Geographic Analysis", "Badge Analysis"

if nav == "Introduction":
    animated_loading()
    home()
elif nav == "Data Collection":
    # global df
    animated_loading()
    data_collect()
elif nav == "Data Wrangling":
    # global df
    animated_loading()
    data_wrangle1()

elif nav == "EDA":
    animated_loading()
    nav = st.sidebar.radio("EDA", ["Correlation", "Distribution", "Questions vs Badges", "3_D","Pair_plot"])
    if nav == "Correlation":
        animated_loading()
        correlation_analysis()
    elif nav == "Distribution":
        animated_loading()
        # st.dataframe(df.head())
        distribution()
    elif nav == "Questions vs Badges":
        animated_loading()
        question_badges()
    elif nav == "3_D":
        animated_loading()
        three_d()
    elif nav =="Pair_plot":
        animated_loading()
        pair_plot()
elif nav == "Predict Rating":
    animated_loading()
    predict_rating()
elif nav == "Deploy":
    animated_loading()
    deploy()
