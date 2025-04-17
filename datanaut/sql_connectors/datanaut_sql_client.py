import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, SecretStr, validator
from sqlalchemy import create_engine, exc, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatanautClient")


class DatabaseDialect(str, Enum):
    """Supported database dialects"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"


class DatabaseConfig(BaseModel):
    """Database configuration with validation"""
    dialect: DatabaseDialect
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    user: Optional[str] = None
    password: Optional[SecretStr] = None
    
    # Additional connection parameters
    connect_args: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('host')
    def validate_host(cls, v, values):
        """Validate host is present for remote databases"""
        if values.get('dialect') != DatabaseDialect.SQLITE and not v:
            raise ValueError(f"Host is required for {values.get('dialect')} connections")
        return v
    
    @validator('port')
    def validate_port(cls, v, values):
        """Set default port based on dialect if not provided"""
        if not v:
            dialect = values.get('dialect')
            if dialect == DatabaseDialect.POSTGRESQL:
                return 5432
            elif dialect == DatabaseDialect.MYSQL:
                return 3306
            elif dialect == DatabaseDialect.MSSQL:
                return 1433
            elif dialect == DatabaseDialect.SQLITE:
                return None  # SQLite doesn't use port
        return v
    
    @validator('user')
    def validate_user(cls, v, values):
        """Validate user is present for remote databases"""
        if values.get('dialect') != DatabaseDialect.SQLITE and not v:
            raise ValueError(f"User is required for {values.get('dialect')} connections")
        return v
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True


class DatanautSQLClient:
    """
    A database client that supports multiple database dialects with read-only access.
    
    Supported databases:
    - PostgreSQL
    - MySQL
    - SQLite
    - Microsoft SQL Server
    """
    
    # Mapping of dialects to SQLAlchemy driver names
    DIALECT_DRIVERS = {
        DatabaseDialect.POSTGRESQL: "psycopg2",
        DatabaseDialect.MYSQL: "pymysql",
        DatabaseDialect.SQLITE: "",  # SQLite doesn't need a separate driver specification
        DatabaseDialect.MSSQL: "pyodbc",
    }
    
    # Mapping of dialects to additional connection arguments
    DIALECT_CONNECT_ARGS = {
        DatabaseDialect.MSSQL: {"driver": "ODBC Driver 17 for SQL Server"},
        # Add other dialect-specific connect args as needed
    }
    
    def __init__(
        self,
        dialect: str,
        database: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        connect_args: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the DatanautClient with database connection parameters.
        
        Args:
            dialect: Database dialect (postgresql, mysql, sqlite, mssql)
            database: Database name (or file path for SQLite)
            host: Database host (not required for SQLite)
            port: Database port (optional, defaults to standard port for the dialect)
            user: Database username (not required for SQLite)
            password: Database password (not required for SQLite)
            connect_args: Additional connection arguments to pass to SQLAlchemy
            log_level: Logging level (default: INFO)
        """
        # Set logging level
        logger.setLevel(log_level)
        logger.info(f"Initializing DatanautClient for {dialect}")
        
        # Initialize config with validation
        try:
            # Create masked password for logging
            masked_password = "********" if password else None
            logger.info(f"Connecting to {dialect} database '{database}' on {host}:{port} with user '{user}'")
            
            self.config = DatabaseConfig(
                dialect=dialect,
                host=host,
                port=port,
                database=database,
                user=user,
                password=SecretStr(password) if password else None,
                connect_args=connect_args or {}
            )
            
            # Merge dialect-specific connect args with user-provided ones
            if dialect in self.DIALECT_CONNECT_ARGS:
                for k, v in self.DIALECT_CONNECT_ARGS[dialect].items():
                    if k not in self.config.connect_args:
                        self.config.connect_args[k] = v
            
            # Create the engine with read-only guarantees
            self.engine = self._create_engine()
            logger.info(f"Successfully created engine for {dialect}")
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Error initializing DatanautClient: {str(e)}")
            raise
    
    def _create_engine(self) -> Engine:
        """
        Create a SQLAlchemy engine with read-only guarantees.
        
        Returns:
            SQLAlchemy Engine instance
        """
        try:
            if self.config.dialect == DatabaseDialect.SQLITE:
                # SQLite connection string
                connection_string = f"sqlite:///{self.config.database}"
                # Add read-only mode for SQLite
                if "mode" not in self.config.connect_args:
                    self.config.connect_args["mode"] = "ro"
            else:
                # Construct driver string
                driver = self.DIALECT_DRIVERS[self.config.dialect]
                driver_str = f"+{driver}" if driver else ""
                
                # Get password safely
                password = self.config.password.get_secret_value() if self.config.password else ""
                
                # Connection string for other databases
                connection_string = (
                    f"{self.config.dialect}{driver_str}://"
                    f"{self.config.user}:{password}@"
                    f"{self.config.host}:{self.config.port}/"
                    f"{self.config.database}"
                )
            
            # Create engine with connection pooling disabled for better isolation
            engine = create_engine(
                connection_string,
                connect_args=self.config.connect_args,
                poolclass=NullPool,  # Disable connection pooling for better isolation
                execution_options={"readonly": True}  # Set read-only mode
            )
            
            return engine
            
        except Exception as e:
            logger.error(f"Error creating database engine: {str(e)}")
            raise ValueError(f"Failed to create database engine: {str(e)}")
    
    def _test_connection(self) -> None:
        """Test the database connection to ensure it works."""
        try:
            with self.engine.connect() as conn:
                # Use a simple test query
                result = conn.execute(text("SELECT 1")).fetchone()
                if result[0] == 1:
                    logger.info("Database connection test successful")
                else:
                    logger.warning("Database connection test returned unexpected result")
        except exc.SQLAlchemyError as e:
            logger.error(f"Database connection test failed: {str(e)}")
            raise ConnectionError(f"Could not connect to database: {str(e)}")
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query and return results as a list of dictionaries.
        
        Args:
            query: SQL query string
            params: Query parameters for parameterized queries
            
        Returns:
            List of dictionaries representing query results
        """
        if params is None:
            params = {}
            
        # Basic SQL injection check
        self._validate_read_only_query(query)
        
        try:
            logger.info(f"Executing query: {query}")
            logger.debug(f"Query parameters: {params}")
            
            with self.engine.connect() as conn:
                # Execute with parameters
                result = conn.execute(text(query), params)
                
                # Convert result to list of dictionaries
                columns = result.keys()
                rows = result.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except exc.SQLAlchemyError as e:
            logger.error(f"Database query error: {str(e)}")
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def query_to_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a read-only SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters for parameterized queries
            
        Returns:
            pandas DataFrame with query results
        """
        if params is None:
            params = {}
            
        # Basic SQL injection check
        self._validate_read_only_query(query)
        
        try:
            logger.info(f"Executing query to dataframe: {query}")
            logger.debug(f"Query parameters: {params}")
            
            with self.engine.connect() as conn:
                # Read directly into dataframe
                df = pd.read_sql(text(query), conn, params=params)
                logger.info(f"Query returned {len(df)} rows with {len(df.columns)} columns")
                return df
                
        except Exception as e:
            logger.error(f"Query to dataframe error: {str(e)}")
            raise RuntimeError(f"Query to dataframe failed: {str(e)}")
    
    def get_tables(self) -> List[str]:
        """
        Get a list of all tables in the database.
        
        Returns:
            List of table names
        """
        try:
            logger.info("Fetching table list")
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"Found {len(tables)} tables")
            return tables
        except exc.SQLAlchemyError as e:
            logger.error(f"Error fetching tables: {str(e)}")
            raise RuntimeError(f"Failed to get table list: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a specified table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of dictionaries with column information
        """
        try:
            logger.info(f"Fetching schema for table '{table_name}'")
            inspector = inspect(self.engine)
            
            if table_name not in inspector.get_table_names():
                logger.warning(f"Table '{table_name}' not found in database")
                raise ValueError(f"Table '{table_name}' not found in database")
                
            columns = inspector.get_columns(table_name)
            schema_info = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": col.get("default", None),
                    "primary_key": col.get("primary_key", False)
                }
                for col in columns
            ]
            
            logger.info(f"Found {len(schema_info)} columns for table '{table_name}'")
            return schema_info
            
        except exc.SQLAlchemyError as e:
            logger.error(f"Error fetching schema for table '{table_name}': {str(e)}")
            raise RuntimeError(f"Failed to get schema for table '{table_name}': {str(e)}")
    
    def _validate_read_only_query(self, query: str) -> None:
        """
        Validate that the query is read-only.
        
        Args:
            query: SQL query to validate
            
        Raises:
            ValueError: If the query is not read-only
        """
        # Remove comments and normalize whitespace
        normalized_query = " ".join(
            line for line in query.split("\n") 
            if not line.strip().startswith("--")
        ).strip().lower()
        
        # Check for write operations
        write_operations = [
            "insert into", "update ", "delete from", "drop ", "alter ", "create ",
            "truncate ", "grant ", "revoke ", "merge ", "upsert ", "replace into",
            "call ", "exec ", "execute "
        ]
        
        for operation in write_operations:
            if operation in normalized_query:
                logger.error(f"Write operation detected: '{operation}' in query")
                raise ValueError(f"Write operation '{operation}' is not allowed. This client is read-only.")
        
        logger.debug("Query validated as read-only")
    
    def close(self) -> None:
        """Close all database connections."""
        if hasattr(self, 'engine'):
            try:
                logger.info("Closing database connections")
                self.engine.dispose()
                logger.info("Database connections closed successfully")
            except Exception as e:
                logger.error(f"Error closing database connections: {str(e)}")
    
    def __enter__(self):
        """Context manager entry method."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method that ensures connections are closed."""
        self.close()



# Example usage
# # Basic usage
# client = DatanautSQLClient(
#     dialect="postgresql",
#     host="localhost",
#     port=5432,
#     database="your_db",
#     user="your_user",
#     password="your_password"
# )

# # Query data
# results = client.execute_query("SELECT * FROM users WHERE status = :status", {"status": "active"})

# # Get data as DataFrame
# df = client.query_to_dataframe("SELECT * FROM products LIMIT 100")

# # Get table information
# tables = client.get_tables()
# schema = client.get_table_schema("customers")

# # Context manager usage
# with DatanautSQLClient(...) as client:
#     data = client.execute_query("SELECT * FROM orders")