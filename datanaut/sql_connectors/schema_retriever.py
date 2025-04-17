from typing import Dict, List, Any, Optional
import logging
import json
from pydantic import BaseModel

from .datanaut_sql_client import DatanautSQLClient, DatabaseDialect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SchemaRetriever")

class TableSchema(BaseModel):
    """Model representing the schema of a database table"""
    table_name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    row_count: Optional[int] = None
    sample_data: Optional[List[Dict[str, Any]]] = None

class DatabaseSchema(BaseModel):
    """Model representing the overall database schema"""
    dialect: str
    database_name: str
    tables: List[TableSchema]
    relationships: Optional[List[Dict[str, Any]]] = None

def get_database_schema(
    dialect: str,
    database: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    include_sample_data: bool = False,
    sample_rows: int = 3,
    include_row_counts: bool = True,
    detect_relationships: bool = True
) -> Dict[str, Any]:
    """
    Retrieve complete schema information for all tables in a database.
    
    Args:
        dialect: Database dialect (postgresql, mysql, sqlite, mssql)
        database: Database name or path
        host: Database host (not required for SQLite)
        port: Database port (optional)
        user: Database username (not required for SQLite)
        password: Database password (not required for SQLite)
        include_sample_data: Whether to include sample data from each table
        sample_rows: Number of sample rows to retrieve if include_sample_data is True
        include_row_counts: Whether to include row counts for each table
        detect_relationships: Whether to detect and include foreign key relationships
        
    Returns:
        Dictionary containing comprehensive database schema information
    """
    logger.info(f"Retrieving schema for {dialect} database: {database}")
    
    try:
        # Initialize the DatanautClient
        client = DatanautSQLClient(
            dialect=dialect,
            database=database,
            host=host,
            port=port,
            user=user,
            password=password
        )
        
        # Get all tables
        tables = client.get_tables()
        logger.info(f"Found {len(tables)} tables in database")
        
        table_schemas = []
        relationships = []
        
        # Process each table
        for table_name in tables:
            logger.info(f"Processing schema for table: {table_name}")
            
            # Get column information
            columns = client.get_table_schema(table_name)
            
            # Extract primary keys
            primary_keys = [col["name"] for col in columns if col.get("primary_key", False)]
            
            table_schema = {
                "table_name": table_name,
                "columns": columns,
                "primary_keys": primary_keys
            }
            
            # Get row count if requested
            if include_row_counts:
                try:
                    count_result = client.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
                    row_count = count_result[0]["count"] if count_result else 0
                    table_schema["row_count"] = row_count
                    logger.info(f"Table {table_name} has {row_count} rows")
                except Exception as e:
                    logger.warning(f"Could not get row count for {table_name}: {str(e)}")
                    table_schema["row_count"] = None
            
            # Get sample data if requested
            if include_sample_data:
                try:
                    sample_data = client.execute_query(f"SELECT * FROM {table_name} LIMIT {sample_rows}")
                    table_schema["sample_data"] = sample_data
                    logger.info(f"Retrieved {len(sample_data)} sample rows from {table_name}")
                except Exception as e:
                    logger.warning(f"Could not get sample data for {table_name}: {str(e)}")
                    table_schema["sample_data"] = []
            
            table_schemas.append(table_schema)
        
        # Detect relationships if requested and if dialect supports it
        if detect_relationships and dialect != DatabaseDialect.SQLITE:
            relationships = _detect_relationships(client, dialect)
        
        # Create the final schema object
        database_schema = {
            "dialect": dialect,
            "database_name": database,
            "tables": table_schemas,
            "relationships": relationships if detect_relationships else None
        }
        
        # Close the connection
        client.close()
        
        logger.info(f"Successfully retrieved schema for {len(tables)} tables")
        return database_schema
        
    except Exception as e:
        logger.error(f"Error retrieving database schema: {str(e)}")
        raise RuntimeError(f"Failed to retrieve database schema: {str(e)}")

def _detect_relationships(client: DatanautSQLClient, dialect: str) -> List[Dict[str, Any]]:
    """
    Detect foreign key relationships in the database.
    Different dialects require different queries to extract this information.
    
    Args:
        client: DatanautClient instance
        dialect: Database dialect
        
    Returns:
        List of dictionaries containing relationship information
    """
    relationships = []
    
    try:
        if dialect == DatabaseDialect.POSTGRESQL:
            query = """
            SELECT
                tc.table_schema as schema_name,
                tc.constraint_name as constraint_name,
                tc.table_name as source_table,
                kcu.column_name as source_column,
                ccu.table_schema as target_schema,
                ccu.table_name as target_table,
                ccu.column_name as target_column
            FROM
                information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
            WHERE
                tc.constraint_type = 'FOREIGN KEY';
            """
            
        elif dialect == DatabaseDialect.MYSQL:
            query = """
            SELECT
                TABLE_NAME as source_table,
                COLUMN_NAME as source_column,
                REFERENCED_TABLE_NAME as target_table,
                REFERENCED_COLUMN_NAME as target_column,
                CONSTRAINT_NAME as constraint_name
            FROM
                information_schema.KEY_COLUMN_USAGE
            WHERE
                REFERENCED_TABLE_SCHEMA = DATABASE()
                AND REFERENCED_TABLE_NAME IS NOT NULL;
            """
            
        elif dialect == DatabaseDialect.MSSQL:
            query = """
            SELECT
                fk.name as constraint_name,
                OBJECT_NAME(fk.parent_object_id) as source_table,
                COL_NAME(fkc.parent_object_id, fkc.parent_column_id) as source_column,
                OBJECT_NAME(fk.referenced_object_id) as target_table,
                COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) as target_column
            FROM
                sys.foreign_keys fk
                INNER JOIN sys.foreign_key_columns fkc
                    ON fk.object_id = fkc.constraint_object_id;
            """
        else:
            logger.info(f"Relationship detection not implemented for {dialect}")
            return []
        
        # Execute the appropriate query
        relationships_data = client.execute_query(query)
        logger.info(f"Found {len(relationships_data)} foreign key relationships")
        
        # Format the relationships in a consistent way
        for rel in relationships_data:
            relationship = {
                "source_table": rel.get("source_table"),
                "source_column": rel.get("source_column"),
                "target_table": rel.get("target_table"),
                "target_column": rel.get("target_column"),
                "constraint_name": rel.get("constraint_name")
            }
            relationships.append(relationship)
        
        return relationships
        
    except Exception as e:
        logger.warning(f"Error detecting relationships: {str(e)}")
        return []

def get_schema_as_markdown(schema: Dict[str, Any]) -> str:
    """
    Convert the database schema to a formatted markdown string.
    This can be useful for displaying the schema in a readable format.
    
    Args:
        schema: The database schema dictionary returned by get_database_schema
        
    Returns:
        Markdown string representation of the schema
    """
    markdown = f"# Database Schema: {schema['database_name']} ({schema['dialect']})\n\n"
    
    # Add table information
    for table in schema['tables']:
        markdown += f"## Table: {table['table_name']}\n\n"
        
        if 'row_count' in table and table['row_count'] is not None:
            markdown += f"Rows: {table['row_count']}\n\n"
        
        # Add column information
        markdown += "| Column | Type | Nullable | Primary Key | Default |\n"
        markdown += "|--------|------|----------|-------------|--------|\n"
        
        for column in table['columns']:
            is_pk = "✓" if column.get("primary_key", False) else ""
            is_nullable = "✓" if column.get("nullable", True) else "✗"
            default = column.get("default") if column.get("default") is not None else ""
            
            markdown += f"| {column['name']} | {column['type']} | {is_nullable} | {is_pk} | {default} |\n"
        
        markdown += "\n"
        
        # Add sample data if available
        if 'sample_data' in table and table['sample_data']:
            markdown += "### Sample Data\n\n"
            
            # Get all keys from the sample data
            keys = set()
            for row in table['sample_data']:
                keys.update(row.keys())
            
            # Create the markdown table header
            markdown += "| " + " | ".join(keys) + " |\n"
            markdown += "| " + " | ".join(["---" for _ in keys]) + " |\n"
            
            # Add the sample data rows
            for row in table['sample_data']:
                values = [str(row.get(key, "")) for key in keys]
                markdown += "| " + " | ".join(values) + " |\n"
            
            markdown += "\n"
    
    # Add relationship information if available
    if schema.get('relationships'):
        markdown += "## Foreign Key Relationships\n\n"
        markdown += "| Source Table | Source Column | Target Table | Target Column | Constraint Name |\n"
        markdown += "|-------------|--------------|-------------|--------------|----------------|\n"
        
        for rel in schema['relationships']:
            markdown += f"| {rel['source_table']} | {rel['source_column']} | {rel['target_table']} | {rel['target_column']} | {rel['constraint_name']} |\n"
    
    return markdown

def describe_schema_for_agent(
    dialect: str,
    database: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    format: str = 'json'
) -> str:
    """
    Generate a schema description suitable for an AI agent to understand database structure.
    
    Args:
        dialect: Database dialect (postgresql, mysql, sqlite, mssql)
        database: Database name or path
        host: Database host (not required for SQLite)
        port: Database port (optional)
        user: Database username (not required for SQLite)
        password: Database password (not required for SQLite)
        format: Output format ('json' or 'markdown')
        
    Returns:
        Database schema in the specified format
    """
    logger.info(f"Generating {format} schema description for agent")
    
    # Get the schema with sample data and relationship detection
    schema = get_database_schema(
        dialect=dialect,
        database=database,
        host=host,
        port=port,
        user=user,
        password=password,
        include_sample_data=True,
        sample_rows=2,
        include_row_counts=True,
        detect_relationships=True
    )
    
    # Return in the requested format
    if format.lower() == 'markdown':
        return get_schema_as_markdown(schema)
    elif format.lower() == 'json':
        return json.dumps(schema, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'markdown'.")

# Example usage
if __name__ == "__main__":
    # Example of how to use the function
    schema_info = describe_schema_for_agent(
        dialect="postgresql",
        host="localhost",
        port=5432,
        database="example_db",
        user="postgres",
        password="password",
        format="markdown"
    )
    
    print(schema_info)