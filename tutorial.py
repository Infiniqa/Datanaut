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