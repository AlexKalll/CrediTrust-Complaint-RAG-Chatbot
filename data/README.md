# Data Directory

This directory contains complaint data for the CreditTrust AI Complaint Analyst.

## Files

- **`sample_complaints.csv`** - Sample data for demonstration purposes
- **`filtered_complaints.csv`** - Your actual complaint data (not tracked in git)

## Adding Your Own Data

1. **Place your CSV file** in this directory
2. **Name it** `filtered_complaints.csv`
3. **Ensure it has these columns:**
   - `Product` - Product categories (e.g., Credit Card, Personal Loan)
   - `Consumer complaint narrative` - Complaint text/description
   - `Issue` - Issue types (e.g., Billing Error, Customer Service)
   - `Complaint ID` - Unique identifier for each complaint

## Data Format Example

```csv
Product,Consumer complaint narrative,Issue,Complaint ID
Credit Card,"I was charged an annual fee without notification",Billing Error,CC001
Personal Loan,"Loan terms were not clearly explained",Unclear Terms,PL001
```

## Notes

- The `sample_complaints.csv` file is included for immediate demo functionality
- Your actual data files are excluded from version control for privacy
- The app will automatically detect and use whichever data file is available
- For production use, replace the sample data with your real complaint data
