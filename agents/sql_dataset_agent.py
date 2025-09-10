"""
SQL Dataset Generation Agent
Integrates SQL query generation and dataset creation with the AutoML pipeline
"""

import chromadb
import datetime
import json
import os
import pandas as pd
import re
import sqlite3
import tempfile
import uuid
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from sqlalchemy import text
from typing import Dict, Any, List, Optional

from .base_agent import create_model_client

# Load environment variables
load_dotenv()

# Constants
SQL_EXAMPLES_DIR = "sql"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "sql_examples"


class SQLDatabaseManager:
    """Manages SQL database connections and operations"""

    def __init__(self):
        self.db = None
        self.list_tables_tool = None
        self.get_schema_tool = None
        self.chroma_client = None
        self.collection = None
        self.llm = ChatOpenAI(model="gpt-4o", request_timeout=30)

    def initialize_database(self, pg_uri: str):
        """Initialize database connection and tools"""
        try:
            self.db = SQLDatabase.from_uri(pg_uri)
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            tools = toolkit.get_tools()

            self.list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
            self.get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

            print("✅ Database connection initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
            return False

    def initialize_chromadb(self):
        """Initialize ChromaDB for SQL examples"""
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(name=COLLECTION_NAME)
            print("✅ ChromaDB initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            return False


# Global database manager
db_manager = SQLDatabaseManager()


def ensure_sql_examples_dir():
    """Ensure SQL examples directory exists"""
    if not os.path.exists(SQL_EXAMPLES_DIR):
        print(f"Creating directory '{SQL_EXAMPLES_DIR}' for SQL examples.")
        os.makedirs(SQL_EXAMPLES_DIR)


def index_sql_files():
    """
    Check SQL_EXAMPLES_DIR and add any new .sql files to ChromaDB
    that haven't been indexed yet.
    """
    global db_manager

    ensure_sql_examples_dir()

    if not db_manager.initialize_chromadb():
        return False

    print("Starting SQL files indexing process...")

    try:
        # Get list of already indexed files
        indexed_files = db_manager.collection.get(include=[]).get('ids', [])
    except Exception as e:
        print(f"Error getting indexed files from ChromaDB: {e}")
        indexed_files = []

    files_in_dir = [f for f in os.listdir(SQL_EXAMPLES_DIR) if f.endswith(".sql")]
    new_files_found = False

    for filename in files_in_dir:
        if filename not in indexed_files:
            new_files_found = True
            filepath = os.path.join(SQL_EXAMPLES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sql_content = f.read()

                db_manager.collection.add(
                    documents=[sql_content],
                    ids=[filename]
                )
                print(f"✅ File indexed: {filename}")
            except Exception as e:
                print(f"❌ Error indexing file {filename}: {e}")

    if not new_files_found and files_in_dir:
        print("No new files found to index. Database is up to date.")
    elif not files_in_dir:
        print(f"No .sql files in directory '{SQL_EXAMPLES_DIR}' to index.")

    return True


def generate_prompt_template(system_prompt: str, user_input: str):
    """Generate a prompt template for the model."""
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    return (
            {
                "context": RunnablePassthrough(),
                "input": RunnableLambda(lambda x: x.get("input", "")),
            }
            | prompt_template
            | ChatOpenAI(model="gpt-4o")
            | StrOutputParser()
    )


def list_tables() -> List[str]:
    """Get available tables in the database."""
    global db_manager
    if not db_manager.list_tables_tool:
        raise Exception("Database not initialized. Call initialize_database first.")

    tables = db_manager.list_tables_tool.invoke({})
    print("Available tables:", tables)
    return tables


def get_schema(table_names: str) -> str:
    """
    Get schema for the provided tables.
    Args:
        table_names: A string with table names separated by commas.
    """
    global db_manager
    if not db_manager.get_schema_tool:
        raise Exception("Database not initialized. Call initialize_database first.")

    schema = db_manager.get_schema_tool.invoke({"table_names": table_names})
    print("Schema retrieved:", schema)
    return schema


def retrieve_sql_examples(user_question: str) -> List[str]:
    """
    Search the vector database (ChromaDB) for relevant SQL examples.
    """
    global db_manager
    n_results = 5
    print(f"\nSearching for {n_results} SQL examples for question: '{user_question}'")

    if not db_manager.collection:
        print("⚠️ ChromaDB not initialized. No examples retrieved.")
        return []

    try:
        results = db_manager.collection.query(
            query_texts=[user_question],
            n_results=n_results
        )

        retrieved_docs = results.get('documents', [[]])[0]

        if retrieved_docs:
            print(f"✅ RAG: Found {len(retrieved_docs)} relevant SQL examples.")
        else:
            print("⚠️ RAG: No SQL examples found for this query.")

        return retrieved_docs
    except Exception as e:
        print(f"❌ Error during ChromaDB search: {e}")
        return []


def run_query(sql_query: str, add_limit: bool = True) -> Optional[Dict[str, Any]]:
    """Execute the final SQL query on the database and return the result.
    
    Args:
        sql_query: The SQL query to execute
        add_limit: Whether to add a LIMIT clause if one doesn't exist (default: True)
    """
    global db_manager
    if not db_manager.db:
        raise Exception("Database not initialized. Call initialize_database first.")

    print("Executing query:", sql_query)

    try:
        # Clean up the query generated by LLM
        query = sql_query.replace("```sql", "").replace("```", "").strip()

        # Skip EXPLAIN queries if they somehow got through
        if query.upper().startswith("EXPLAIN"):
            print("Skipping EXPLAIN query - removing EXPLAIN prefix")
            query = query[7:].strip()  # Remove "EXPLAIN" prefix

        # Only add limit if requested and query doesn't already have one
        if add_limit and "limit" not in query.lower():
            query += " LIMIT 100"
            print("Added LIMIT 100 to query for safety")
        elif not add_limit:
            print("Executing query without automatic LIMIT (full dataset generation)")

        # Use the database to run the query and get structured results
        result = db_manager.db.run_no_throw(query, include_columns=True)

        print("Query result:", result)

        if "psycopg2.errors" in str(result) or "error" in str(result).lower():
            print("Error in query:", result)
            return {"result": None, "query": query, "error": str(result)}

        # Try to get more structured data by executing directly with cursor
        try:
            # Get the underlying connection and execute with cursor for better control
            engine = db_manager.db._engine
            with engine.connect() as connection:
                # Use text() from sqlalchemy to create a proper text clause
                from sqlalchemy import text
                cursor_result = connection.execute(text(query))

                # Get column names
                columns = list(cursor_result.keys())

                # Get all rows as dictionaries
                rows = []
                for row in cursor_result:
                    row_dict = {}
                    for i, col_name in enumerate(columns):
                        value = row[i]
                        # Convert special types to JSON-serializable formats
                        if hasattr(value, 'isoformat'):  # datetime objects
                            row_dict[col_name] = value.isoformat()
                        elif str(type(value)) == "<class 'decimal.Decimal'>":  # Decimal objects
                            row_dict[col_name] = float(value)
                        else:
                            row_dict[col_name] = value
                    rows.append(row_dict)

                return {"result": rows, "query": query, "columns": columns}

        except Exception as structured_error:
            print(f"Structured query failed, using fallback: {structured_error}")
            # Fallback to original result
            return {"result": result, "query": query, "error": str(structured_error)}

    except Exception as e:
        print(f"Error executing query: {e}")
        return None


def select_tables(context: str, input_question: str) -> str:
    """Generate relevant tables based on user question."""
    prompt_text = """You are an expert in selecting all database tables involved in the user's query.
    Based on the user's query, provide a comma-separated list of all database tables that relate to the query.
    Only respond with the table names in the format: table1, table2, table3. 
    Here are the tables in the database: {context}"""

    chain = generate_prompt_template(
        system_prompt=prompt_text,
        user_input=input_question,
    )
    return chain.invoke({"context": context, "input": input_question})


def generate_sql_query(context: str, input_question: str) -> str:
    """Generate SQL query based on user question and schema."""
    query_system_prompt = """You are a SQL expert with strong attention to detail.

    When generating the query:
    - This is an Odoo PostgreSQL database
    - Output the SQL query that answers the input question without a tool call
    - Given an input question, output a syntactically correct PostgreSQL query to run
    - Never query for all columns from a specific table, only ask for relevant columns
    - NEVER make stuff up if you don't have enough information to answer the query
    - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database
    - Use the database schema to create the query
    - Just give me the SQL text to execute without explanation or comments

    
    Context: {context}"""

    chain = generate_prompt_template(
        system_prompt=query_system_prompt,
        user_input=input_question,
    )
    return chain.invoke({"context": context, "input": input_question})


def check_sql_query(sql_query: str, engine) -> str:
    """
    Check SQL query with basic local fixes + validation via EXPLAIN in PostgreSQL.
    
    Args:
        sql_query: The SQL query string.
        engine: SQLAlchemy engine already conectado a PostgreSQL.

    Returns:
        str: The validated/corrected SQL query.

    Raises:
        ValueError: If the query has critical issues.
    """
    query = sql_query.strip()

    # --- Reglas locales rápidas ---
    # 1. active = 1/0 → boolean
    query = re.sub(r"\bactive\s*=\s*1\b", "active = true", query, flags=re.IGNORECASE)
    query = re.sub(r"\bactive\s*=\s*0\b", "active = false", query, flags=re.IGNORECASE)

    # 2. UNION → UNION ALL (si no es UNION ALL)
    query = re.sub(r"\bUNION\b(?!\s+ALL)", "UNION ALL", query, flags=re.IGNORECASE)

    # 3. Detectar NOT IN con NULL (no corregimos, pero avisamos)
    if re.search(r"NOT\s+IN\s*\(.*NULL.*\)", query, flags=re.IGNORECASE):
        raise ValueError("Query usa NOT IN con NULL, esto puede dar resultados inesperados.")

    # --- Validación en PostgreSQL ---
    try:
        explain_query = f"EXPLAIN {query}"
        with engine.connect() as conn:
            conn.execute(text(explain_query))
    except Exception as e:
        error_msg = str(e)
        # Check for specific SQL errors that we can provide better messages for
        if "must appear in the GROUP BY clause" in error_msg:
            raise ValueError(
                f"SQL Error: Columns in SELECT must appear in GROUP BY clause. Query needs to be fixed: {error_msg}")
        elif "column" in error_msg and "does not exist" in error_msg:
            raise ValueError(f"SQL Error: Referenced column does not exist: {error_msg}")
        elif "syntax error" in error_msg.lower():
            raise ValueError(f"SQL Syntax Error: {error_msg}")
        else:
            # Log the error but don't fail completely - some EXPLAIN errors might not affect actual execution
            print(f"Warning: EXPLAIN validation failed: {error_msg}")
            print("Proceeding with query execution despite EXPLAIN failure...")
            # Don't raise an exception, just log and continue

    return query


def summarize_result(user_question: str, sql_query: str, query_result: Dict) -> str:
    """
    Generate a natural language summary of the query result,
    considering the user's original question.
    """
    return (
        f"Based on your question '{user_question}', I executed the following query:\n"
        f"```sql\n{sql_query}\n```\n"
        f"Result: {query_result.get('result', 'No results')}"
    )


# Tool functions for the agent
def select_tables_node(question: str, previous_error: str = None) -> Dict[str, Any]:
    """Node for selecting relevant tables."""
    try:
        context = list_tables()
        if previous_error:
            context += f"\nPrevious error: {previous_error}"
        selected_tables = select_tables(context=str(context), input_question=question)
        return {"tables": selected_tables, "error": None}
    except Exception as e:
        return {"tables": "", "error": str(e)}


def get_schema_node(question: str, tables: str) -> Dict[str, Any]:
    """Node for getting table schema."""
    try:
        schema = get_schema(tables)
        return {"schema": schema, "tables": tables, "question": question}
    except Exception as e:
        return {"schema": "", "tables": tables, "question": question, "error": str(e)}


def generate_sql_query_node(question: str, schema: str, retrieve_docs: List[str]) -> Dict[str, Any]:
    """Node for generating SQL query."""
    try:
        sql_examples = "\n\n".join(retrieve_docs) if retrieve_docs else ""
        context = f"Schema: {schema}\n\nSQL query examples:\n{sql_examples}"
        sql_query = generate_sql_query(context=context, input_question=question)
        return {"sql_query": sql_query}
    except Exception as e:
        return {"sql_query": "", "error": str(e)}


def check_sql_query_node(sql_query: str) -> Dict[str, Any]:
    """Node for checking SQL query."""
    try:
        global db_manager
        checked_query = check_sql_query(sql_query, db_manager.db._engine)
        return {"sql_query": checked_query}
    except Exception as e:
        return {"sql_query": sql_query, "error": str(e)}


def execute_query_node(sql_query: str) -> Dict[str, Any]:
    """Node for executing SQL query."""
    try:
        query_result = run_query(sql_query)
        if query_result is None:
            return {"error": "Error executing query", "sql_query": sql_query}
        return {"query_result": query_result, "sql_query": sql_query}
    except Exception as e:
        return {"error": str(e), "sql_query": sql_query}


def display_result_node(query_result: Dict[str, Any]) -> Dict[str, Any]:
    """Node for displaying query results."""
    try:
        result_data = query_result.get("result", "")
        if isinstance(result_data, str):
            # Try to parse JSON-like string
            result_data = result_data.replace("'", '"').replace("None", "null")
            try:
                data = json.loads(result_data)
                df = pd.DataFrame(data)
                print(df)
                return {"query_result": df.to_dict(orient="records")}
            except:
                print("Raw result:", result_data)
                return {"query_result": result_data}
        else:
            print("Result:", result_data)
            return {"query_result": result_data}
    except Exception as e:
        print(f"Error formatting query result: {e}")
        return {"query_result": query_result}


def summary_node(sql_query: str, question: str, query_result: Dict[str, Any]) -> str:
    """Node for generating summary."""
    try:
        summary = summarize_result(question, sql_query, query_result)
        print("Summary:", summary)
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"


class SQLDatasetAgent:
    """Agent for generating datasets from SQL queries for ML training"""

    def __init__(self):
        self.model_client = create_model_client()

        # Create the assistant agent
        self.agent = AssistantAgent(
            name="SQLDatasetAgent",
            model_client=self.model_client,
            system_message="""You are an expert Text-to-SQL agent.
                You must execute EXACTLY the following steps, in order,
                and call ONLY the appropriate tool for each step.

                1. select_tables_node: call list_available_tables and decide relevant tables.
                2. get_schema_node: call get_table_schema with those tables.
                3. retrieve_sql_examples: call retrieve_sql_examples(original question).
                4. generate_sql_query_node: call generate_sql_query with schemas+examples.
                5. check_sql_query_node: call check_sql_query and approve. If errors, correct and retry.
                6. execute_query_node: call execute_sql_query.
                7. summary_node: call summarize_result(question, query, rows).

                Don't skip steps. Call one tool at a time until step 7.
                For the final response, respond only with the natural answer in Spanish.
                Include the SQL query result in a table and also include the executed SQL query.
                """,
            tools=[
                select_tables_node,
                get_schema_node,
                retrieve_sql_examples,
                generate_sql_query_node,
                check_sql_query_node,
                execute_query_node,
                display_result_node,
                summary_node,
            ],
        )

    async def generate_sql_dataset(self, question: str, connection_info: Dict[str, Any],
                                   output_dir: str = "./datasets") -> Dict[str, Any]:
        """Generate dataset from SQL query based on natural language question."""

        try:
            # Validate connection info
            if not connection_info or not connection_info.get("pg_uri"):
                return {
                    "success": False,
                    "error": "Missing database connection information",
                    "sql_query": None,
                    "csv_path": None,
                    "row_count": 0,
                    "column_count": 0,
                    "file_size_mb": 0.0,
                    "columns": []
                }

            # Initialize database connection
            global db_manager
            if not db_manager.initialize_database(connection_info["pg_uri"]):
                return {
                    "success": False,
                    "error": "Failed to initialize database connection",
                    "sql_query": None,
                    "csv_path": None,
                    "row_count": 0,
                    "column_count": 0,
                    "file_size_mb": 0.0,
                    "columns": []
                }

            # Index SQL files
            if not index_sql_files():
                print("⚠️ Warning: Failed to index SQL files, continuing without examples")

            # Create termination condition and team
            termination = TextMessageTermination("SQL_Agent")
            team = RoundRobinGroupChat(
                [self.agent],
                termination_condition=termination,
                max_turns=20
            )

            # Run the agent team
            result = await team.run(task=question)
            final_response = result.messages[-1].content
            print("+++++++++++++++++++++++++++++++++++ Final response:", final_response)

            # Try to extract SQL query and data from the agent's execution
            sql_query = None
            query_data = None

            # Look through the messages to find the SQL query and results
            for message in result.messages:
                content = message.content
                if isinstance(content, str):
                    # Try to extract SQL query from the content
                    if "```sql" in content:
                        import re
                        sql_match = re.search(r'```sql\n(.*?)\n```', content, re.DOTALL)
                        if sql_match:
                            sql_query = sql_match.group(1).strip()
                            if sql_query:
                                try:
                                    # For dataset generation, don't add automatic LIMIT to get full data
                                    query_result = run_query(sql_query, add_limit=False)
                                    if query_result and query_result.get("result") and not query_result.get("error"):
                                        query_data = query_result["result"]
                                        # If we got structured data with columns, use it directly
                                        if isinstance(query_data, list) and query_data and isinstance(query_data[0],
                                                                                                      dict):
                                            print("✅ Got structured data from query execution")
                                        print(f"Query data type: {type(query_data)}")
                                        print(f"Query data sample: {str(query_data)[:300]}...")

                                        # Also check if we got columns info
                                        if query_result.get("columns"):
                                            print(f"Available columns: {query_result['columns']}")
                                    elif query_result and query_result.get("error"):
                                        print(f"Query execution returned error: {query_result.get('error')}")
                                        # Don't set query_data, let it remain None
                                except Exception as e:
                                    print(f"Error re-executing query: {e}")
                                    # Don't set query_data, let it remain None

            # If we have data, save it to CSV
            csv_path = None
            columns = []
            if query_data and sql_query:
                try:
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)

                    # Generate unique filename
                    filename = f"sql_dataset_{str(uuid.uuid4())[:8]}.csv"
                    csv_path = os.path.join(output_dir, filename)

                    # Convert query result to DataFrame and save as CSV
                    if isinstance(query_data, str):
                        # Try to parse string result that looks like a list of dictionaries
                        try:
                            # Handle the specific format from database queries
                            import ast
                            import re
                            from decimal import Decimal

                            # Clean up the string and try to evaluate it as Python literal
                            # First, let's try to parse it as a list of dictionaries
                            clean_data = query_data.strip()

                            # If it's a string representation of a list, evaluate it safely
                            if clean_data.startswith('[') and clean_data.endswith(']'):
                                try:
                                    # Use ast.literal_eval for safe evaluation of basic Python literals
                                    # But we need to handle special types like Decimal and datetime

                                    # First, extract and store datetime values
                                    datetime_pattern = r"datetime\.datetime\((\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+),\s*(\d+)(?:,\s*(\d+))?)?\)"
                                    datetime_matches = []

                                    def datetime_replacer(match):
                                        year, month, day = match.group(1), match.group(2), match.group(3)
                                        hour = match.group(4) or '0'
                                        minute = match.group(5) or '0'
                                        second = match.group(6) or '0'

                                        # Create ISO format datetime string
                                        datetime_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute.zfill(2)}:{second.zfill(2)}"
                                        datetime_matches.append(datetime_str)
                                        return f"'datetime_placeholder_{len(datetime_matches) - 1}'"

                                    # Replace datetime objects with placeholders
                                    clean_data = re.sub(datetime_pattern, datetime_replacer, clean_data)

                                    # Replace Decimal objects
                                    clean_data = re.sub(r"Decimal\('([^']+)'\)", r"'\1'", clean_data)

                                    # Now try to evaluate
                                    data = ast.literal_eval(clean_data)
                                    df = pd.DataFrame(data)

                                    # Replace datetime placeholders with actual datetime strings
                                    for col in df.columns:
                                        if df[col].dtype == 'object':
                                            # Replace datetime placeholders
                                            for i, datetime_str in enumerate(datetime_matches):
                                                placeholder = f'datetime_placeholder_{i}'
                                                df[col] = df[col].astype(str).replace(placeholder, datetime_str)

                                            # Also handle the generic placeholder case
                                            if 'datetime_placeholder' in df[col].astype(str).values:
                                                # If we still have generic placeholders, try to replace with a default
                                                df[col] = df[col].astype(str).replace('datetime_placeholder',
                                                                                      '1970-01-01T00:00:00')

                                            # Try to convert string numbers to numeric
                                            if df[col].dtype == 'object':
                                                try:
                                                    # Check if all non-null values are numeric strings
                                                    numeric_mask = df[col].astype(str).str.match(r'^\d+\.?\d*$',
                                                                                                 na=False)
                                                    if numeric_mask.sum() > 0:
                                                        df[col] = pd.to_numeric(df[col], errors='ignore')
                                                except:
                                                    pass

                                except (ValueError, SyntaxError):
                                    # If literal_eval fails, create a simple DataFrame
                                    df = pd.DataFrame({"result": [query_data]})
                            else:
                                # Try JSON parsing as fallback
                                clean_data = query_data.replace("'", '"').replace("None", "null")
                                data = json.loads(clean_data)
                                df = pd.DataFrame(data)

                        except Exception as parse_error:
                            print(f"Error parsing query result: {parse_error}")
                            # If all parsing fails, create a simple DataFrame with the raw result
                            df = pd.DataFrame({"result": [query_data]})

                    elif isinstance(query_data, list):
                        # Handle list of dictionaries or tuples
                        if query_data and isinstance(query_data[0], dict):
                            # Perfect case - list of dictionaries, directly convert to DataFrame
                            df = pd.DataFrame(query_data)
                            print(f"✅ Created DataFrame from list of dictionaries. Shape: {df.shape}")
                        elif query_data and isinstance(query_data[0], (tuple, list)):
                            # If it's a list of tuples/lists, we need column names
                            # This would require getting column names from the SQL execution
                            df = pd.DataFrame(query_data)
                        else:
                            df = pd.DataFrame(query_data)
                    else:
                        # Try to convert to DataFrame directly
                        df = pd.DataFrame(query_data)

                    # Get column names
                    columns = list(df.columns)

                    # Save to CSV
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Dataset saved to: {csv_path}")

                    # Calculate file metadata
                    row_count = len(df)
                    column_count = len(df.columns)
                    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)  # Convert bytes to MB

                except Exception as e:
                    print(f"❌ Error saving dataset to CSV: {e}")
                    csv_path = None
                    row_count = 0
                    column_count = 0
                    file_size_mb = 0.0
                    columns = []
            else:
                # No data was generated
                print("⚠️ Warning: No data was generated from the SQL query")
                if sql_query:
                    print(f"SQL Query was: {sql_query}")
                else:
                    print("No SQL query was extracted from agent response")

                row_count = 0
                column_count = 0
                file_size_mb = 0.0
                columns = []

            # If we successfully generated data, return success
            # If no data but we have a valid SQL query, still consider it partially successful
            success = bool(csv_path) or bool(sql_query)

            # Return comprehensive result
            return {
                "success": success,
                "response": final_response,
                "question": question,
                "sql_query": sql_query,
                "csv_path": csv_path,
                "row_count": row_count,
                "column_count": column_count,
                "file_size_mb": file_size_mb,
                "columns": columns,
                "timestamp": datetime.datetime.now().isoformat(),
                "warning": "No data generated" if not csv_path else None
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql_query": None,
                "csv_path": None,
                "row_count": 0,
                "column_count": 0,
                "file_size_mb": 0.0,
                "columns": []
            }


# Global instance for use in pipeline
sql_dataset_agent = SQLDatasetAgent()
