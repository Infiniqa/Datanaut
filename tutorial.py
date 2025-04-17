from datanaut.sql_connectors.datanaut_sql_client import DatanautSQLClient, DatabaseDialect
from datanaut.sql_connectors.schema_retriever import describe_schema_for_agent
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()


# use the sql client
client = DatanautSQLClient(
    dialect="mysql",
    host="localhost",
    port=3306,
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
    
)

# Query data
results = client.execute_query("SELECT * FROM Amazon_Sale_Report LIMIT 10")

# Get data as DataFrame
df = client.query_to_dataframe("SELECT * FROM Financial_Statements LIMIT 100")

# Get table information
tables = client.get_tables()
# print(tables)
schema = client.get_table_schema("Sale_Report")
# print(schema)

# Context manager usage
with DatanautSQLClient(
    dialect="mysql",
    host="localhost",
    port=3306,
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
) as client:
    data = client.execute_query("SELECT * FROM Sales_Product_Ecommerce LIMIT 10")
    # print(data)

# Use the schema retirever
if __name__ == "__main__":
    # Example of how to use the function
    schema_info = describe_schema_for_agent(
        dialect="mysql",
        host="localhost",
        port=3306,
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        format="json" # or "markdown/json"
    )
    print(schema_info)


# How to use python executor tool
from datanaut.python_executor.datanaut_python_executor import DatanautPythonExecutor, DatanautAnalysisTool

# Initialize SQL client
sql_client = DatanautSQLClient(
    dialect="mysql",
    host="localhost",
    port=3306,
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

# Initialize Python executor
python_executor = DatanautPythonExecutor(
    venv_path="venvs",
    default_packages=["numpy", "pandas", "matplotlib", "seaborn"]
)

# Example 1: Execute Python code asynchronously
code = """
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data
data = {'Category': ['A', 'B', 'C', 'D'], 'Values': [10, 25, 15, 30]}
df = pd.DataFrame(data)

# Generate a bar plot
plt.figure(figsize=(10, 6))
plt.bar(df['Category'], df['Values'])
plt.title('Sample Bar Chart')
plt.savefig('sample_plot.png')

print("Analysis complete!")
"""

# Execute asynchronously
import asyncio
result = asyncio.run(python_executor.execute_code_async(code))
print(result.stdout)


# Example 2: Use the integrated analysis tool
analysis_tool = DatanautAnalysisTool(
    sql_client=sql_client,
    python_executor=python_executor
)

# SQL query to get data
query = """SELECT 
  SKU AS product_sku,
  SUM(Amount) AS total_sales
FROM Amazon_Sale_Report
WHERE Status NOT LIKE '%Cancelled%'
GROUP BY SKU
ORDER BY total_sales DESC
LIMIT 10;
"""

# Python analysis code
analysis_code = """
# df is automatically provided with query results
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style='whitegrid')

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='product_sku', y='total_sales', data=df, palette='Blues_d')
plt.title('Top 10 Products by Total Sales (Excluding Cancelled Orders)')
plt.xlabel('Product SKU')
plt.ylabel('Total Sales (INR)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_10_products_sales.png')

# Calculate statistics
total_sales = df['total_sales'].sum()
average_sales = df['total_sales'].mean()

print(f"Total sales (top 10 products): ₹{total_sales:,.2f}")
print(f"Average sales per top product: ₹{average_sales:,.2f}")

"""

# Run analysis (combines SQL and Python)
results = asyncio.run(analysis_tool.query_and_analyze(query, analysis_code))
print(results['analysis_output'])