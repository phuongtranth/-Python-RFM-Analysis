# [Python] RFM Analysis
### I. Introduction
This project leverages Python to develop an RFM (Recency, Frequency, Monetary) segmentation model for SuperStore, a global retail company. The objective is to support the Marketing team in classifying a large customer dataset, enabling personalized and effective marketing strategies for the upcoming festive season. Comprehensive data preparation and segmentation were conducted to create customer groups based on RFM metrics, followed by visualizations to provide insights into customer behavior. Strategic recommendations were also provided to prioritize specific RFM components for optimizing SuperStore's marketing and sales efforts.
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

#### RFM Calculation
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
### III. Data Visualization and Insight
#### RFM Distribution Analysis
**Recency Distribution:** The graph shows the distribution of customers based on how recently they made a purchase.
![a](https://github.com/user-attachments/assets/e41de750-18d4-441b-808a-da0281f3b65a)
**Key observations:**
The two highest bars are in the 0-100 range with over 2600 customers (over 60% of total customers), indicating that the company has a strong base of recent customers, which is positive.

**Frequency Distribution:** This graph shows how often customers make purchases.
![b](https://github.com/user-attachments/assets/5e4e5f7f-66d3-48a1-80f4-0021f1f03c34)
**Key observations:**
- The majority of customers fall in the 1-2 frequency group. Hence, there's potential to convert one-time buyers into repeat customers.
- There's a smaller but valuable segment of frequent purchasers (777 in the 5-20 frequency group)

**Monetary Distribution:** This graph shows the distribution of customer spending.
![c](https://github.com/user-attachments/assets/2f52707b-6c0b-49e7-b31b-ca87398b79c5)
**Key observations:**
- The company has a solid base of mid-range spenders with 2516 in the 100-1k and 1560 in the 1k-10k.
- There's a valuable segment of high-spending customers to nurture (104 in the 10k+)

#### Customer base
![d](https://github.com/user-attachments/assets/c2c08f9f-6ae3-4010-a4c1-7dfdd7474b22)

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
