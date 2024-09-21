# [Python] RFMAnalysis
### Introduction
### EDA 
### Data Visualization and Insight
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

### Segment Characteristics and Recommendations
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
