import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nest_asyncio
from pydantic import BaseModel, Field, validator

# Apply nest_asyncio to allow nested event loops (useful in notebook environments)
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatanautPythonExecutor")


class ExecutionMode(str, Enum):
    """Supported execution modes"""
    SYNC = "sync"      # Synchronous execution (blocking)
    ASYNC = "async"    # Asynchronous execution (non-blocking)


class ExecutionLevel(str, Enum):
    """Security levels for code execution"""
    ISOLATED = "isolated"      # Fully isolated environment with restricted imports
    RESTRICTED = "restricted"  # Limited access to system resources
    STANDARD = "standard"      # Standard Python execution environment


class ExecutionConfig(BaseModel):
    """Configuration for Python code execution"""
    # Main configuration options
    venv_path: str = Field(default="./venvs")
    execution_mode: ExecutionMode = Field(default=ExecutionMode.ASYNC)
    execution_level: ExecutionLevel = Field(default=ExecutionLevel.STANDARD)
    timeout: int = Field(default=60)  # Timeout in seconds
    venv_name: Optional[str] = Field(default=None)  # Name for the virtual environment
    
    # Package management
    default_packages: List[str] = Field(default_factory=list)
    allowed_packages: List[str] = Field(default_factory=list)
    
    # Resource limits
    memory_limit_mb: Optional[int] = Field(default=None)
    
    # Output handling
    capture_stdout: bool = Field(default=True)
    capture_stderr: bool = Field(default=True)
    max_output_size: int = Field(default=1024 * 1024)  # 1MB
    
    # Execution settings
    working_directory: Optional[str] = Field(default=None)
    env_variables: Dict[str, str] = Field(default_factory=dict)
    
    @validator('venv_path')
    def validate_venv_path(cls, v):
        """Ensure the path is absolute and exists"""
        path = Path(v).resolve()
        # Create the directory if it doesn't exist
        if not path.exists():
            logger.info(f"Creating virtual environment directory at {path}")
            path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    @validator('default_packages')
    def validate_default_packages(cls, v):
        """Ensure default packages list contains valid package names"""
        if not v:
            # Default set of useful packages for data analysis
            return [
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "scikit-learn",
                "plotly",
                "statsmodels",
                "pydantic",
                "python-dotenv",
                "ipython",
                "nest_asyncio"
            ]
        return v
    
    @validator('allowed_packages')
    def validate_allowed_packages(cls, v, values):
        """If allowed_packages is empty, use default_packages"""
        if not v:
            return values.get('default_packages', []) + [
                "scipy",
                "openpyxl",
                "xlrd",
                "tabulate",
                "beautifulsoup4",
                "requests",
                "jupyter",
                "pillow",
                "networkx",
                "sympy",
                "tqdm"
            ]
        return v


class ExecutionResult(BaseModel):
    """Result of a Python code execution"""
    success: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exception: Optional[str] = None
    execution_time: float = 0.0
    return_value: Any = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    execution_id: str
    code_snippet: str


class DatanautPythonExecutor:
    """
    A sandboxed Python code execution environment for AI agents.
    
    This class creates an isolated virtual environment and provides
    methods to execute Python code safely with various security levels.
    """
    
    def __init__(
        self,
        venv_path: Optional[str] = None,
        venv_name: Optional[str] = None,
        default_packages: Optional[List[str]] = None,
        execution_level: str = "standard",
        create_venv: bool = True,
        working_directory: Optional[str] = None,
        env_variables: Optional[Dict[str, str]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the Python executor with specified configuration.
        
        Args:
            venv_path: Path to store virtual environments (default: './venvs')
            venv_name: Specific name for the virtual environment (default: auto-generated)
            default_packages: List of packages to install in the virtual environment
            execution_level: Security level for code execution ('isolated', 'restricted', 'standard')
            create_venv: Whether to create/setup the virtual environment during initialization
            working_directory: Directory where code will be executed
            env_variables: Environment variables to set for code execution
            log_level: Logging level
        """
        # Set logging level
        logger.setLevel(log_level)
        logger.info("Initializing DatanautPythonExecutor")
        
        # Initialize configuration
        try:
            self.config = ExecutionConfig(
                venv_path=venv_path,
                venv_name=venv_name,
                default_packages=default_packages or [],
                execution_level=execution_level,
                working_directory=working_directory,
                env_variables=env_variables or {}
            )
            
            # Set up virtual environment path with specific or unique identifier
            if self.config.venv_name:
                self.venv_name = self.config.venv_name
            else:
                self.venv_name = "datanaut_py"
            
            self.venv_path = os.path.join(self.config.venv_path, self.venv_name)
            
            # Create the working directory for code execution
            if self.config.working_directory:
                os.makedirs(self.config.working_directory, exist_ok=True)
            else:
                self.config.working_directory = tempfile.mkdtemp(prefix="datanaut_work_")
            
            logger.info(f"Working directory set to {self.config.working_directory}")
            
            # Check if the virtual environment already exists
            venv_exists = self._check_venv_exists()
            
            # Create and set up the virtual environment if requested and doesn't exist
            if create_venv and not venv_exists:
                logger.info(f"Creating new virtual environment at {self.venv_path}")
                self._create_virtual_environment()
                self._install_default_packages()
            elif venv_exists:
                logger.info(f"Using existing virtual environment at {self.venv_path}")
            
            # Initialize event loop for async operations
            self.loop = asyncio.get_event_loop()
            
            # Keep track of running executions
            self.active_executions = {}
            
        except Exception as e:
            logger.error(f"Error initializing DatanautPythonExecutor: {str(e)}")
            raise
    
    def _check_venv_exists(self) -> bool:
        """Check if the virtual environment already exists and is valid."""
        if not os.path.exists(self.venv_path):
            return False
            
        # Check for Python executable in the venv
        python_path = self._get_python_executable()
        if not os.path.exists(python_path):
            return False
            
        # Check for pip in the venv
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(self.venv_path, 'Scripts', 'pip')
            if not os.path.exists(pip_path + '.exe'):
                return False
        else:  # Unix/Linux/Mac
            pip_path = os.path.join(self.venv_path, 'bin', 'pip')
            if not os.path.exists(pip_path):
                return False
                
        return True
    
    def _create_virtual_environment(self) -> None:
        """Create a new virtual environment for code execution."""
        try:
            logger.info(f"Creating virtual environment at {self.venv_path}")
            
            # Check if Python's venv module is available
            try:
                import venv
                venv_available = True
            except ImportError:
                venv_available = False
            
            # Create virtual environment
            if venv_available:
                venv.create(self.venv_path, with_pip=True)
            else:
                # Fall back to using the virtualenv command
                result = subprocess.run(
                    [sys.executable, "-m", "virtualenv", self.venv_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.debug(f"virtualenv output: {result.stdout}")
            
            logger.info(f"Virtual environment created successfully at {self.venv_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e.stderr}")
            raise RuntimeError(f"Failed to create virtual environment: {e.stderr}")
        except Exception as e:
            logger.error(f"Error creating virtual environment: {str(e)}")
            raise
    
    def _install_default_packages(self) -> None:
        """Install default packages in the virtual environment."""
        if not self.config.default_packages:
            logger.info("No default packages to install")
            return
        
        try:
            logger.info(f"Installing {len(self.config.default_packages)} default packages")
            
            # Get path to pip in virtual environment
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(self.venv_path, 'Scripts', 'pip')
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(self.venv_path, 'bin', 'pip')
            
            # Install packages
            packages_str = " ".join(self.config.default_packages)
            logger.info(f"Installing packages: {packages_str}")
            
            cmd = [pip_path, "install", "--upgrade", "pip"]
            self._run_command(cmd)
            
            cmd = [pip_path, "install"] + self.config.default_packages
            self._run_command(cmd)
            
            logger.info("Default packages installed successfully")
            
        except Exception as e:
            logger.error(f"Error installing default packages: {str(e)}")
            raise
    
    def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    async def install_package(self, package_name: str) -> bool:
        """
        Install a Python package in the virtual environment.
        
        Args:
            package_name: Name of the package to install
            
        Returns:
            True if installation successful, False otherwise
        """
        if package_name not in self.config.allowed_packages:
            logger.warning(f"Package '{package_name}' not in allowed list")
            return False
        
        try:
            # Get path to pip in virtual environment
            if os.name == 'nt':  # Windows
                pip_path = os.path.join(self.venv_path, 'Scripts', 'pip')
            else:  # Unix/Linux/Mac
                pip_path = os.path.join(self.venv_path, 'bin', 'pip')
            
            logger.info(f"Installing package: {package_name}")
            
            # Run pip install asynchronously
            proc = await asyncio.create_subprocess_exec(
                pip_path, "install", package_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                logger.info(f"Successfully installed {package_name}")
                return True
            else:
                logger.error(f"Failed to install {package_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing package {package_name}: {str(e)}")
            return False
    
    def _prepare_code_file(self, code: str, execution_id: str) -> str:
        """
        Save the code to a temporary file for execution.
        
        Args:
            code: Python code to execute
            execution_id: Unique identifier for this execution
            
        Returns:
            Path to temporary code file
        """
        # Create file path in working directory
        file_path = os.path.join(
            self.config.working_directory,
            f"datanaut_code_{execution_id}.py"
        )
        
        # Write code to file
        with open(file_path, 'w') as f:
            f.write(code)
        
        return file_path
    
    def _get_python_executable(self) -> str:
        """Get the path to the Python executable in the virtual environment."""
        if os.name == 'nt':  # Windows
            python_path = os.path.join(self.venv_path, 'Scripts', 'python.exe')
        else:  # Unix/Linux/Mac
            python_path = os.path.join(self.venv_path, 'bin', 'python')
        
        return python_path
    
    async def execute_code_async(self, code: str) -> ExecutionResult:
        """
        Execute Python code asynchronously in the virtual environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult object containing execution results
        """
        execution_id = uuid.uuid4().hex
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Starting async execution {execution_id}")
        logger.debug(f"Code to execute: {code[:100]}...")
        
        # Save active execution
        self.active_executions[execution_id] = {
            "status": "running",
            "start_time": start_time
        }
        
        try:
            # Save code to temporary file
            code_file = self._prepare_code_file(code, execution_id)
            
            # Get Python executable path
            python_executable = self._get_python_executable()
            
            # Prepare environment variables
            env = os.environ.copy()
            env.update(self.config.env_variables)
            
            # Add execution ID as environment variable
            env["DATANAUT_EXECUTION_ID"] = execution_id
            
            # Set up sandbox constraints based on execution level
            if self.config.execution_level == ExecutionLevel.ISOLATED:
                # Add code to restrict imports and system access
                with open(code_file, 'r') as f:
                    original_code = f.read()
                
                sandboxed_code = f"""
# Sandbox security constraints
import sys
import builtins

# List of allowed modules in isolated mode
ALLOWED_MODULES = {{'os', 'sys', 'math', 'datetime', 'random', 'json', 
                   'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
                   'plotly', 'statsmodels', 'io', 're', 'collections',
                   'itertools', 'functools', 'operator', 'pathlib'}}

# Store original import
original_import = builtins.__import__

def secure_import(name, *args, **kwargs):
    if name not in ALLOWED_MODULES:
        if '.' in name:
            base_module = name.split('.')[0]
            if base_module not in ALLOWED_MODULES:
                raise ImportError(f"Import of {{name}} not allowed in isolated mode")
        else:
            raise ImportError(f"Import of {{name}} not allowed in isolated mode")
    return original_import(name, *args, **kwargs)

# Replace built-in __import__ with our secure version
builtins.__import__ = secure_import

# Sandbox output directory
import os
os.environ['MPLCONFIGDIR'] = '{self.config.working_directory}'

# Original code starts below
{original_code}
"""
                # Save the sandboxed code
                with open(code_file, 'w') as f:
                    f.write(sandboxed_code)
            
            # Execute the code
            proc = await asyncio.create_subprocess_exec(
                python_executable, code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.working_directory
            )
            
            # Set up timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    self.config.timeout
                )
            except asyncio.TimeoutError:
                # Kill the process if it times out
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                
                end_time = asyncio.get_event_loop().time()
                execution_time = end_time - start_time
                
                result = ExecutionResult(
                    success=False,
                    stdout=None,
                    stderr=f"Execution timed out after {self.config.timeout} seconds",
                    exception=f"TimeoutError: Execution exceeded {self.config.timeout} seconds",
                    execution_time=execution_time,
                    execution_id=execution_id,
                    code_snippet=code[:100] + "..." if len(code) > 100 else code
                )
                return result
            
            # Process completed
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            
            # Collect artifacts (generated files)
            artifacts = {}
            try:
                for f in os.listdir(self.config.working_directory):
                    if f.startswith(f"datanaut_output_{execution_id}"):
                        file_path = os.path.join(self.config.working_directory, f)
                        with open(file_path, 'r') as file:
                            artifacts[f] = file.read()
            except Exception as e:
                logger.warning(f"Error collecting artifacts: {str(e)}")
            
            # Create result object
            result = ExecutionResult(
                success=proc.returncode == 0,
                stdout=stdout_str if self.config.capture_stdout else None,
                stderr=stderr_str if self.config.capture_stderr else None,
                exception=None if proc.returncode == 0 else "Execution failed with non-zero exit code",
                execution_time=execution_time,
                execution_id=execution_id,
                artifacts=artifacts,
                code_snippet=code[:100] + "..." if len(code) > 100 else code
            )
            
            # Update execution status
            self.active_executions[execution_id] = {
                "status": "completed" if proc.returncode == 0 else "failed",
                "start_time": start_time,
                "end_time": end_time
            }
            
            logger.info(f"Code execution {execution_id} completed in {execution_time:.2f}s with status: {result.success}")
            return result
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            logger.error(f"Error during code execution {execution_id}: {str(e)}")
            
            # Update execution status
            self.active_executions[execution_id] = {
                "status": "error",
                "start_time": start_time,
                "end_time": end_time,
                "error": str(e)
            }
            
            # Create failure result
            result = ExecutionResult(
                success=False,
                stdout=None,
                stderr=None,
                exception=str(e),
                execution_time=execution_time,
                execution_id=execution_id,
                code_snippet=code[:100] + "..." if len(code) > 100 else code
            )
            return result
    
    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute Python code synchronously in the virtual environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            ExecutionResult object containing execution results
        """
        try:
            # Run the async version using the event loop
            return self.loop.run_until_complete(self.execute_code_async(code))
        except Exception as e:
            logger.error(f"Error in synchronous code execution: {str(e)}")
            
            execution_id = uuid.uuid4().hex
            result = ExecutionResult(
                success=False,
                exception=str(e),
                execution_id=execution_id,
                code_snippet=code[:100] + "..." if len(code) > 100 else code
            )
            return result
    
    async def execute_data_analysis_async(
        self, 
        code: str,
        dataframe_var: str = "df",
        data: Optional[Any] = None
    ) -> ExecutionResult:
        """
        Execute data analysis code with a provided dataframe asynchronously.
        
        This is a convenience method for executing code that operates on a dataframe.
        The dataframe is serialized and loaded in the execution environment.
        
        Args:
            code: Python code for data analysis
            dataframe_var: Variable name for the dataframe in the code
            data: Pandas DataFrame or data that can be converted to a DataFrame
            
        Returns:
            ExecutionResult object containing execution results
        """
        try:
            import pandas as pd
            
            execution_id = uuid.uuid4().hex
            
            # Convert data to DataFrame if needed
            if data is not None and not isinstance(data, pd.DataFrame):
                try:
                    data = pd.DataFrame(data)
                except Exception as e:
                    logger.error(f"Could not convert data to DataFrame: {str(e)}")
                    raise ValueError(f"Could not convert data to DataFrame: {str(e)}")
            
            # Create temporary CSV file for the dataframe
            if data is not None:
                csv_path = os.path.join(
                    self.config.working_directory,
                    f"datanaut_data_{execution_id}.csv"
                )
                
                logger.info(f"Saving dataframe with {len(data)} rows to {csv_path}")
                data.to_csv(csv_path, index=False)
                
                # Wrap user code with dataframe loading code
                complete_code = f"""
                    import pandas as pd

                    # Load the dataframe
                    {dataframe_var} = pd.read_csv(r"{csv_path}")

                    # User code starts here
                    {code}
                    """
            else:
                complete_code = code
            
            # Execute the complete code
            return await self.execute_code_async(complete_code)
            
        except Exception as e:
            logger.error(f"Error in data analysis execution: {str(e)}")
            
            result = ExecutionResult(
                success=False,
                exception=str(e),
                execution_id=execution_id,
                code_snippet=code[:100] + "..." if len(code) > 100 else code
            )
            return result
    
    def execute_data_analysis(
        self, 
        code: str,
        dataframe_var: str = "df",
        data: Optional[Any] = None
    ) -> ExecutionResult:
        """
        Execute data analysis code with a provided dataframe synchronously.
        
        Args:
            code: Python code for data analysis
            dataframe_var: Variable name for the dataframe in the code
            data: Pandas DataFrame or data that can be converted to a DataFrame
            
        Returns:
            ExecutionResult object containing execution results
        """
        try:
            # Run the async version using the event loop
            return self.loop.run_until_complete(
                self.execute_data_analysis_async(code, dataframe_var, data)
            )
        except Exception as e:
            logger.error(f"Error in synchronous data analysis execution: {str(e)}")
            
            execution_id = uuid.uuid4().hex
            result = ExecutionResult(
                success=False,
                exception=str(e),
                execution_id=execution_id,
                code_snippet=code[:100] + "..." if len(code) > 100 else code
            )
            return result
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the status of an execution by ID.
        
        Args:
            execution_id: ID of the execution to check
            
        Returns:
            Dictionary with execution status information
        """
        if execution_id not in self.active_executions:
            return {"status": "unknown", "error": "Execution ID not found"}
        
        return self.active_executions[execution_id]
    
    def cleanup(self) -> None:
        """Clean up resources, including the virtual environment."""
        try:
            logger.info("Cleaning up resources")
            
            # We don't remove the virtual environment automatically anymore
            # since it may be reused by future instances
            
            # Remove the working directory if it's a temporary one
            if (self.config.working_directory and 
                os.path.exists(self.config.working_directory) and
                "datanaut_work_" in self.config.working_directory):
                logger.info(f"Removing working directory at {self.config.working_directory}")
                shutil.rmtree(self.config.working_directory, ignore_errors=True)
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def __enter__(self):
        """Context manager entry method."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method that ensures resources are cleaned up."""
        self.cleanup()


class DatanautAnalysisTool:
    """
    A high-level tool for database analysis combining SQL and Python execution.
    
    This class integrates DatanautSQLClient and DatanautPythonExecutor to provide
    a comprehensive tool for data analysis workflows.
    """
    
    def __init__(
        self,
        sql_client=None,
        python_executor=None,
        create_python_env: bool = True,
        default_packages: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the analysis tool with SQL client and Python executor.
        
        Args:
            sql_client: Existing DatanautSQLClient instance (optional)
            python_executor: Existing DatanautPythonExecutor instance (optional)
            create_python_env: Whether to create a Python environment if not provided
            default_packages: Default packages to install in Python environment
            working_directory: Working directory for code execution
            log_level: Logging level
        """
        self.logger = logging.getLogger("DatanautAnalysisTool")
        self.logger.setLevel(log_level)
        
        # Store SQL client
        self.sql_client = sql_client
        
        # Create or store Python executor
        if python_executor:
            self.python_executor = python_executor
        elif create_python_env:
            self.logger.info("Creating Python execution environment")
            self.python_executor = DatanautPythonExecutor(
                default_packages=default_packages,
                working_directory=working_directory,
                log_level=log_level
            )
        else:
            self.python_executor = None
            
        self.logger.info("DatanautAnalysisTool initialized successfully")
    
    async def query_and_analyze(
        self, 
        query: str,
        analysis_code: str,
        query_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run an SQL query and analyze the results with Python code.
        
        Args:
            query: SQL query to execute
            analysis_code: Python code to analyze the query results
            query_params: Parameters for SQL query
            
        Returns:
            Dictionary with query results and analysis results
        """
        if not self.sql_client:
            raise ValueError("SQL client not initialized")
        if not self.python_executor:
            raise ValueError("Python executor not initialized")
        
        try:
            self.logger.info("Executing SQL query")
            
            # Run the SQL query and get results as DataFrame
            df = self.sql_client.query_to_dataframe(query, query_params)
            
            self.logger.info(f"Query returned {len(df)} rows, now running analysis")
            
            # Run the analysis code on the DataFrame - use synchronous version to avoid nested awaits
            result = self.python_executor.execute_data_analysis(analysis_code, data=df)
            
            return {
                "query_success": True,
                "query_rows": len(df),
                "analysis_success": result.success,
                "analysis_output": result.stdout,
                "analysis_error": result.stderr,
                "execution_time": result.execution_time,
                "execution_id": result.execution_id
            }
            
        except Exception as e:
            self.logger.error(f"Error in query_and_analyze: {str(e)}")
            return {
                "query_success": False,
                "analysis_success": False,
                "error": str(e)
            }
    
    def cleanup(self) -> None:
        """Clean up resources used by the tool."""
        if self.python_executor:
            self.python_executor.cleanup()
        
        if hasattr(self.sql_client, 'close'):
            self.sql_client.close()