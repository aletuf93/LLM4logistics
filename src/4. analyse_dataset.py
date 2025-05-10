# %%
import pandas as pd

import matplotlib.pyplot as plt


# %%
#Import cleaned dataset
df_analysis = pd.read_excel("../data/dataset_raw/_master_analysis.xlsx", sheet_name="Final dataset")
# Create a new column "QueryGrouped" based on the values in the "Query" column
def group_queries(query):
    query = query.lower()
    if any(keyword in query for keyword in ["analytics production facility",
                                             "production facility control", 
                                             "production facility design"]):
        return "Production Facility"
    elif any(keyword in query for keyword in ["analytics supply chain distribution network",
                                               "distribution network control",
                                                 "distribution network design",
                                                 "supply chain distribution network control",
                                                 "supply chain distribution network design"]):
        return "Distribution Network"
    elif any(keyword in query for keyword in ["analytics supply chain inventory storage system",
                                               "storage system design",
                                               "supply chain storage inventory system control",
                                               "supply chain storage inventory system design"]):
        return "Warehousing System"
    else:
        return "Cross SCS"

df_analysis['QueryGrouped'] = df_analysis['Query'].apply(group_queries)

# %%
# Create a figure with two subplots aligned in a row
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# Define a consistent color palette for QueryGrouped
color_palette = {
    'Production Facility': 'blue',
    'Distribution Network': 'green',
    'Warehousing System': 'orange',
    'Cross SCS': 'gray'
}

# First subplot: Stacked histogram
df_analysis['Year'] = pd.to_numeric(df_analysis['Year'], errors='coerce')  # Convert to numeric, setting invalid values to NaN
df_analysis['Year'] = df_analysis['Year'].fillna(0).astype(int)  # Replace NaN with 0 and convert to integer
df_analysis.groupby(['Year', 'QueryGrouped']).size().unstack().plot(
    kind='bar', stacked=True, ax=axes[0], color=[color_palette[key] for key in ["Cross SCS", "Production Facility", "Warehousing System", "Distribution Network"]]
)
axes[0].set_title('Number of Research Papers by Year and SCS')
axes[0].set_xlabel('Year')
axes[0].get_legend().remove()  # Remove the legend from the first plot
axes[0].legend(
    title='SCS Type',
    labels=["Cross SCS", "Production Facility", "Warehousing System", "Distribution Network"],
    handles=[plt.Line2D([0], [0], color=color_palette[label], lw=4) for label in ["Cross SCS", "Production Facility", "Warehousing System", "Distribution Network"]],
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
axes[0].set_ylabel('Number of Research Papers')
#axes[0].legend(title='QueryGrouped', bbox_to_anchor=(1.05, 1), loc='upper left')

# Second subplot: Horizontal bar chart
query_counts = df_analysis.groupby(['Query', 'QueryGrouped']).size().reset_index(name='Count')
# Sort by QueryGrouped in the specified order and then by Count
query_counts['QueryGrouped'] = pd.Categorical(
    query_counts['QueryGrouped'],
    categories=["Cross SCS", "Production Facility", "Warehousing System", "Distribution Network"],
    ordered=True
)
query_counts = query_counts.sort_values(['QueryGrouped', 'Count'], ascending=[True, False])

# Update the color palette to follow the specified order
ordered_color_palette = ["gray", "blue", "orange", "green"]

# Apply the ordered colors to the second plot
colors = query_counts['QueryGrouped'].cat.codes.map(lambda x: ordered_color_palette[x])
colors = query_counts['QueryGrouped'].map(color_palette)
axes[1].barh(query_counts['Query'], query_counts['Count'], color=colors)
axes[1].set_title('Number of Resear Papers by Search Term')
axes[1].set_xlabel('Number of Research Papers')
axes[1].set_ylabel('Search Terms')

# Adjust layout
plt.tight_layout()
plt.show()

# %%
