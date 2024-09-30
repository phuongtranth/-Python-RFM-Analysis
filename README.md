# [Python] RFM Analysis and Customer Segmentation Visualization
### I. Introduction
In this project, I conducted an **RFM** (Recency, Frequency, Monetary) analysis for a **global retail company** -  SuperStore, utilizing Python to segment customers and deliver actionable insights **for the Marketing and Sales teams**. Through exploratory **data analysis**, **segmentation modeling**, and **visualizations**, I supported the teams in optimizing customer engagement and enhancing strategic decision-making by identifying key customer groups for targeted campaigns.

#### Dataset
This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
| Field       | Explaintion                                                                                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| InvoiceNo   | Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. |
| StockCode   | Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.                                                         |
| Description | Product (item) name. Nominal.                                                                                                                               |
| Quantity    | The quantities of each product (item) per transaction. Numeric.                                                                                             |
| InvoiceDate | Invoice Date and time. Numeric, the day and time when each transaction was generated.                                                                       |
| UnitPrice   | Unit price. Numeric, Product price per unit in sterling.                                                                                                    |
| CustomerID  | Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.                                                                     |
| Country     | Country name. Nominal, the name of the country where each customer resides.                                                                                 |

### II. Data Preparation
#### Cleaning Data
This part involved checking for missing values, duplicates, and incorrect data types. Appropriate actions were taken, such as imputing missing values, removing duplicates, and correcting data types to ensure data integrity. Additionally, any incorrect or outlier values were identified and handled based on the dataset's context to maintain accuracy and consistency.
```python
#Check for datatype
transactions.dtypes

#Check for missing data in each column
transactions.isna().sum()

#Check for duplicates
print(transactions.duplicated().su

#Check for incorrect values
transactions.describe()
```

**Actions**
```python
#Remove rows had null CustomerID
transactions = transactions.dropna(subset=['CustomerID'])

#Convert CustomerID to datatype 'object' and remove '.0'
transactions['InvoiceDate'] = pd.to_datetime(transactions['InvoiceDate'])
transactions['CustomerID'] = transactions['CustomerID'].astype('object')
transactions['CustomerID'] = transactions['CustomerID'].astype(str).str.replace('.0', '', regex=False)

#Filter data that >0 in Quantity and Unitprice and not cancelled transactions (InvoiceNo not star with C)
transactions = transactions[
    (~transactions['InvoiceNo'].astype(str).str.startswith('C')) &
    (transactions['Quantity'] > 0) &
    (transactions['UnitPrice'] > 0)]
```
#### RFM Calculation and Segmentation
Following data cleaning, the Recency, Frequency, and Monetary values were calculated for each customer. Recency was determined based on the number of days since the last purchase, Frequency measured the total number of transactions, and Monetary value represented the total spending of each customer.
```python
# Calculate Recency
last_purchase_date = transactions.groupby('CustomerID')['InvoiceDate'].max().reset_index(name='LastPurchaseDate')
today = pd.to_datetime('2011-12-31')
last_purchase_date['Recency'] = (today - last_purchase_date['LastPurchaseDate']).dt.days

# Calculate Frequency
frequency = transactions.groupby('CustomerID')['InvoiceDate'].nunique().reset_index(name='Frequency')
frequency['Rank'] = frequency['Frequency'].rank(method='first').astype(int)

# Calculate Monetary
monetary = transactions.assign(Revenue=transactions['Quantity'] * transactions['UnitPrice']).groupby('CustomerID')['Revenue'].sum().reset_index(name='Monetary')

# Merge in 1 df
rfm = last_purchase_date.merge(frequency, on='CustomerID').merge(monetary, on='CustomerID')
rfm.shape
```
Quintiles were used to assign scores to each RFM component, categorizing customers into segments based on their relative RFM values. These scores served as the foundation for customer segmentation and further analysis.
```python
# Quintile
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Rank'], q=5, labels= [1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels= [1,2,3,4,5])

# Create RFM score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
```
#### Segmentation
```python
def segment_customers(row):
    score = row['RFM_Score']
    if score in ['555', '554', '544', '545', '454', '455', '445']:
        return 'Champions'
    elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
        return 'Loyal'
    elif score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451', '442', '441', '431', '453', '433', '432', '423', '353', '352', '351', '342', '341', '333', '323']:
        return 'Potential Loyalist'
    elif score in ['512', '511', '422', '421', '412', '411', '311']:
        return 'New Customers'
    elif score in ['525', '524', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313']:
        return 'Promising'
    elif score in ['535', '534', '443', '434', '343', '334', '325', '324']:
        return 'Need Attention'
    elif score in ['331', '321', '312', '221', '213', '231', '241', '251']:
        return 'About To Sleep'
    elif score in ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '133', '125', '124']:
        return 'At Risk'
    elif score in ['155', '154', '144', '214', '215', '115', '114', '113']:
        return 'Cannot Lose Them'
    elif score in ['332', '322', '233', '232', '223', '222', '132', '123', '122', '212', '211']:
        return 'Hibernating customers'
    elif score in ['111', '112', '121', '131', '141', '151']:
        return 'Lost customers'
    else:
        return 'Unknown'

# Apply the segmentation function to create a new 'Segment' column
rfm['Segment'] = rfm.apply(segment_customers, axis=1)
```
### III. Data Visualization and Insight
#### RFM Distribution Analysis
```python
#Distribution of Recency
fig, ax = plt.subplots(figsize=(12, 3))
sns.histplot(data=rfm, x='Recency', bins =10, ax=ax)
ax.set_title('Distribution of Recency')
ax.set_xlim(left=0)
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=2)
plt.show()
```
**Recency Distribution:** The graph shows the distribution of customers based on how recently they made a purchase.
![a](https://github.com/user-attachments/assets/e41de750-18d4-441b-808a-da0281f3b65a)

**Key observations:**
The two highest bars are in the 0-100 range with over 2600 customers (over 60% of total customers), indicating that the company has a strong base of recent customers, which is positive.

```python
#Distribution of Frequency
binsF = [0, 2, 5, 20, np.inf]
labelsF = ['1-2', '2-5', '5-20', '20+']
rfm['FrequencyGroup'] = pd.cut(rfm['Frequency'], bins=binsF, labels=labelsF)
fig, ax = plt.subplots(figsize=(8, 3))
sns.countplot(x='FrequencyGroup', data=rfm, ax=ax)
ax.set_title('Distribution of Frequency')
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=2)
plt.show()

```
**Frequency Distribution:** This graph shows how often customers make purchases.
![b](https://github.com/user-attachments/assets/5e4e5f7f-66d3-48a1-80f4-0021f1f03c34)

**Key observations:**
- The majority of customers fall in the 1-2 frequency group. Hence, there's potential to convert one-time buyers into repeat customers.
- There's a smaller but valuable segment of frequent purchasers (777 in the 5-20 frequency group)

```python
#Distribution of Monetary
binsM = [0, 100, 1000, 10000, np.inf]
labelsM = ['0-100', '100-1k', '1k-10k', '10k+']
rfm['MonetaryGroup'] = pd.cut(rfm['Monetary'], bins=binsM, labels=labelsM)
fig, ax = plt.subplots(figsize=(8, 3))
sns.countplot(x='MonetaryGroup', data=rfm, ax=ax)
ax.set_title('Distribution of Monetary')
ax.yaxis.set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for container in ax.containers:
    ax.bar_label(container, label_type='edge', padding=2)
plt.show()
```
**Monetary Distribution:** This graph shows the distribution of customer spending.
![c](https://github.com/user-attachments/assets/2f52707b-6c0b-49e7-b31b-ca87398b79c5)

**Key observations:**
- The company has a solid base of mid-range spenders with 2516 in the 100-1k and 1560 in the 1k-10k.
- There's a valuable segment of high-spending customers to nurture (104 in the 10k+)

#### Customer base
```python
#Assign color
segment_colors = {
    'Champions': '#FF0000',
    'Loyal': '#00FFFF',
    'Potential Loyalist': '#00FF00',
    'At Risk': '#FFFF00',
    'Hibernating customers': '#800080',
    'Lost customers': '#FFA500',
    'Need Attention': '#A52A2A',
    'About To Sleep': '#808000',
    'New Customers': '#FFC0CB',
    'Promising': '#FF00FF',
    'Cannot Lose Them': '#736F6E'
}
```
```python
# Sort data by the number of customers in descending order
Number_of_customer = Number_of_customer.sort_values('Cust_count', ascending=False)

# Sort data by the number of customers in descending order
Number_of_customer = Number_of_customer.sort_values('Cust_count', ascending=False)

# Create a bar chart
fig, ax = plt.subplots(figsize=(14, 8))

# Bar plot
bars = ax.bar(Number_of_customer['Segment'],
              Number_of_customer['Cust_count'],
              color=[segment_colors[segment] for segment in Number_of_customer['Segment']],
              edgecolor="black")

# Adding labels on top of bars
for bar, count, share in zip(bars, Number_of_customer['Cust_count'], Number_of_customer['Count_share']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{count:,}\n{int(round(share))}%",
            ha='center', va='bottom', fontsize=12)

# Title and labels
plt.title('Number of Customers by RFM Segment', fontsize=16)
plt.ylabel('Number of Customers', fontsize=14)

# Display the plot
plt.xticks(rotation=45, ha='right')
plt.show()
```
![d](https://github.com/user-attachments/assets/c2c08f9f-6ae3-4010-a4c1-7dfdd7474b22)
```python
# % and Monetary values by Segment
segment_monetary = rfm.groupby('Segment')['Monetary'].sum().reset_index()
total_monetary = segment_monetary['Monetary'].sum()
segment_monetary['Percentage'] = segment_monetary['Monetary'] / total_monetary * 100

segment_monetary = segment_monetary.sort_values('Monetary', ascending=False)

# Create the treemap

fig, ax = plt.subplots(1, figsize=(20,8))
squarify.plot(sizes=segment_monetary['Monetary'],
              label=[f"{s}\n${int(m):,}\n{int(p)}%"
                     for s, m, p in zip(segment_monetary['Segment'],
                                        segment_monetary['Monetary'],
                                        segment_monetary['Percentage'])],
              color=[segment_colors[segment] for segment in segment_monetary['Segment']],
              alpha=0.8,
              bar_kwargs=dict(linewidth=1.5, edgecolor="white"))

plt.title('Total Monetary Value by RFM Segment', fontsize=16)
plt.axis('off')

plt.show()
```
![e](https://github.com/user-attachments/assets/7f84ad3e-448c-401e-ab76-e38c6ebb2976)
**Key observations:**
- SuperStore Company's customer base primarily consists of "Champions" (19%), "Hibernating customers" (15%), "At Risk" (9%), "Potential Loyalist" (9%), and "Loyal" (9%) segments.
- Despite comprising only 9% of the customer base, the "Loyal" segment generates a substantial 62% of the company's total monetary value, followed by "Champions" at 11%.
- The presence of "At Risk", "About To Sleep", and "Cannot Lose Them" segments suggests customer retention challenges.
- The small "New Customers" (0%) and "Promising" (1%) segments indicate a need for customer acquisition and growth strategies.

--> Overall, SuperStore Company has a mix of highly loyal, potentially loyal, and at-risk customers, presenting opportunities for targeted retention, acquisition, and cultivation efforts to optimize the value of its customer base.

### IV. Segment Characteristics and Recommendations
| Segment                                               | Characteristics                                                                                                                             | Recommendations                                                                                                                                                                                                                                                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Champions (19% of customers, 62% of value)            | • Highest monetary value, frequent recent purchases<br>• Likely long-term customers with strong brand loyalty<br>• High average order value | • Send personalized "Year in Review" thank you cards highlighting their top purchases<br>• Offer exclusive early access to holiday sales with additional discount<br>• Provide complimentary gift wrapping and priority shipping<br>• Invite to a virtual VIP holiday event with special product previews |
| Loyal (9% of customers, 11% of value)                 | • High value, consistent engagement<br>• Slightly lower recency or frequency than Champions                                                 | • Send personalized holiday greeting with a thank you gift (e.g., branded calendar)<br>• Offer a loyalty point multiplier for holiday purchases<br>• Provide a surprise upgrade or add-on with their next purchase                                                                                        |
| Potential Loyalists (9% of customers, 2% of value)    | • Recent purchases, moderate frequency<br>• Lower monetary value than Loyal                                                                 | • Offer a special discount on a product category they haven't tried yet<br>• Provide a free consultation or product demo as a holiday bonus                                                                                                                                                               |
| Cannot Lose Them (2% of customers, 2% of value)       | • High-value customers<br>• Recent drop in engagement                                                                                       | • Send a heartfelt "We Miss You" holiday message with a significant comeback offer<br>• Offer an exclusive "loyal customer" discount on their favorite products                                                                                                                                           |
| At Risk (9% of customers, 8% of value)                | • Decreasing engagement<br>• Previously valuable customers<br>• High urgency for re-engagement                                              | • Provide a dedicated customer service line for any issues or questions<br>• Send a survey with a gift card reward to understand their needs and preferences                                                                                                                                              |
| Need Attention (6% of customers, 5% of value)         | • Moderate value<br>• Declining engagement<br>• May be price-sensitive or have changing needs                                               | • Create a personalized product recommendation list for holiday shopping<br>• Offer incentives for feedback                                                                                                                                                                                               |
| About To Sleep (6% of customers, 0% of value)         | • Low recent activity<br>• Previously active customers                                                                                      | • Send a holiday-themed product update highlighting new features or improvements<br>• Time-limited offers to encourage action                                                                                                                                                                             |
| Hibernating customers (15% of customers, 3% of value) | • No recent activity<br>• Previously engaged customers                                                                                      | • Send a year-end catalog featuring best-sellers and new products<br>• Consider retargeting ads                                                                                                                                                                                                           |
| Lost customers (11% of customers, 1% of value)        | • Longest period of inactivity<br>• Lowest engagement                                                                                       | • Send a feedback request with a holiday gift incentive for responses<br>• Analyze reasons for loss<br>• Use insights to prevent future customer loss                                                                                                                                                     |
| New Customers (6% of customers, 0% of value)          | • Recent first purchase<br>• Unknown long-term value<br>• Limited data on preferences and behavior                                          | • Offer a "Holiday Newcomer" discount on their second purchase<br>• Provide a guided tour of product ranges and services via email series                                                                                                                                                                 |
| Promising (3% of customers, 1% of value)              | • Recent engagement<br>• Moderate purchasing behavior                                                                                       | • Nurture with targeted content and offers<br>• Create a personalized "Holiday Must-Haves" list based on browsing history<br>• Offer a progressive discount: save more on each subsequent holiday purchase                                                                                                |
