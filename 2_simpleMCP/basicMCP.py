#!/usr/bin/env python3
"""
MCP Server Example for VLLM Inference with Extended Tools
This creates an MCP server that provides tools for AI model inference using VLLM
plus additional tools for weather, code analysis, and mathematics
"""

import asyncio
import json
import ast
import re
import math
import statistics
from typing import Any, Dict, List, Optional
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from openai import OpenAI
import requests

class VLLMInferenceServer:
    """MCP Server that provides VLLM inference tools and utility tools"""

    def __init__(self):
        self.server = Server("vllm-inference-extended")
        self.vllm_client = None
        self.base_url = "http://localhost:8080/v1"
        self.available_models = []
        self.weather_api_key = None  # Set this to your weather API key

        # Register tools
        self.setup_tools()

    def setup_tools(self):
        """Setup MCP tools for VLLM inference and utility tools"""

        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            return [
                # VLLM Tools
                types.Tool(
                    name="vllm_chat_completion",
                    description="Perform chat completion using VLLM model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "List of chat messages",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                        "content": {"type": "string"}
                                    },
                                    "required": ["role", "content"]
                                }
                            },
                            "model": {"type": "string", "description": "Model name (optional)"},
                            "max_tokens": {"type": "integer", "description": "Maximum tokens", "default": 512},
                            "temperature": {"type": "number", "description": "Temperature", "default": 0.7}
                        },
                        "required": ["messages"]
                    }
                ),
                types.Tool(
                    name="vllm_text_completion",
                    description="Perform text completion using VLLM model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Input prompt"},
                            "model": {"type": "string", "description": "Model name (optional)"},
                            "max_tokens": {"type": "integer", "description": "Maximum tokens", "default": 512},
                            "temperature": {"type": "number", "description": "Temperature", "default": 0.7}
                        },
                        "required": ["prompt"]
                    }
                ),
                types.Tool(
                    name="vllm_get_models",
                    description="Get list of available models from VLLM server",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="vllm_server_status",
                    description="Check VLLM server status and health",
                    inputSchema={
                        "type": "object", 
                        "properties": {}
                    }
                ),
                # Weather Tools
                types.Tool(
                    name="get_weather",
                    description="Get current weather information for a city",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "country": {"type": "string", "description": "Country code (optional)", "default": ""},
                            "units": {"type": "string", "enum": ["metric", "imperial", "kelvin"], "default": "metric"}
                        },
                        "required": ["city"]
                    }
                ),
                types.Tool(
                    name="get_weather_forecast",
                    description="Get weather forecast for a city (5 days)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "country": {"type": "string", "description": "Country code (optional)", "default": ""},
                            "units": {"type": "string", "enum": ["metric", "imperial", "kelvin"], "default": "metric"}
                        },
                        "required": ["city"]
                    }
                ),
                # Code Analysis Tools
                types.Tool(
                    name="analyze_python_code",
                    description="Analyze Python code for syntax, complexity, and potential issues",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to analyze"},
                            "check_syntax": {"type": "boolean", "default": True},
                            "check_complexity": {"type": "boolean", "default": True},
                            "check_style": {"type": "boolean", "default": True}
                        },
                        "required": ["code"]
                    }
                ),
                types.Tool(
                    name="format_code",
                    description="Format and beautify code (supports Python, JavaScript, JSON)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to format"},
                            "language": {"type": "string", "enum": ["python", "javascript", "json"], "default": "python"}
                        },
                        "required": ["code"]
                    }
                ),
                types.Tool(
                    name="extract_functions",
                    description="Extract function definitions and their signatures from Python code",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to analyze"}
                        },
                        "required": ["code"]
                    }
                ),
                # Mathematics Tools
                types.Tool(
                    name="calculate_expression",
                    description="Safely evaluate mathematical expressions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
                            "precision": {"type": "integer", "description": "Decimal precision", "default": 10}
                        },
                        "required": ["expression"]
                    }
                ),
                types.Tool(
                    name="statistical_analysis",
                    description="Perform statistical analysis on a dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "items": {"type": "number"}, "description": "List of numbers"},
                            "operations": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["mean", "median", "mode", "std", "var", "min", "max", "range"]},
                                "default": ["mean", "median", "std"]
                            }
                        },
                        "required": ["data"]
                    }
                ),
                types.Tool(
                    name="solve_equation",
                    description="Solve mathematical equations (quadratic, linear)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "equation": {"type": "string", "description": "Equation to solve (e.g., 'x^2 + 2x + 1 = 0')"},
                            "equation_type": {"type": "string", "enum": ["quadratic", "linear"], "default": "quadratic"}
                        },
                        "required": ["equation"]
                    }
                ),
                types.Tool(
                    name="unit_conversion",
                    description="Convert between different units",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "value": {"type": "number", "description": "Value to convert"},
                            "from_unit": {"type": "string", "description": "Source unit"},
                            "to_unit": {"type": "string", "description": "Target unit"},
                            "category": {"type": "string", "enum": ["length", "weight", "temperature", "volume"], "description": "Unit category"}
                        },
                        "required": ["value", "from_unit", "to_unit", "category"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""

            # VLLM Tools
            if name == "vllm_chat_completion":
                return await self.handle_chat_completion(arguments)
            elif name == "vllm_text_completion":
                return await self.handle_text_completion(arguments)
            elif name == "vllm_get_models":
                return await self.handle_get_models()
            elif name == "vllm_server_status":
                return await self.handle_server_status()
            
            # Weather Tools
            elif name == "get_weather":
                return await self.handle_get_weather(arguments)
            elif name == "get_weather_forecast":
                return await self.handle_get_weather_forecast(arguments)
            
            # Code Analysis Tools
            elif name == "analyze_python_code":
                return await self.handle_analyze_python_code(arguments)
            elif name == "format_code":
                return await self.handle_format_code(arguments)
            elif name == "extract_functions":
                return await self.handle_extract_functions(arguments)
            
            # Mathematics Tools
            elif name == "calculate_expression":
                return await self.handle_calculate_expression(arguments)
            elif name == "statistical_analysis":
                return await self.handle_statistical_analysis(arguments)
            elif name == "solve_equation":
                return await self.handle_solve_equation(arguments)
            elif name == "unit_conversion":
                return await self.handle_unit_conversion(arguments)
            
            else:
                return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    # ... existing VLLM methods remain the same ...
    async def initialize_client(self):
        """Initialize VLLM client"""
        try:
            self.vllm_client = OpenAI(
                base_url=self.base_url,
                api_key="EMPTY"
            )
            # Test connection and get models
            models = self.vllm_client.models.list()
            self.available_models = [model.id for model in models.data]
            return True
        except Exception as e:
            print(f"Failed to initialize VLLM client: {e}")
            return False

    async def handle_chat_completion(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle chat completion requests"""
        if not self.vllm_client:
            if not await self.initialize_client():
                return [types.TextContent(type="text", text="Error: Could not connect to VLLM server")]

        try:
            messages = arguments["messages"]
            model = arguments.get("model", self.available_models[0] if self.available_models else "auto")
            max_tokens = arguments.get("max_tokens", 512)
            temperature = arguments.get("temperature", 0.7)

            response = self.vllm_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            result = {
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage.dict() if response.usage else None
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in chat completion: {str(e)}")]

    async def handle_text_completion(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle text completion requests"""
        if not self.vllm_client:
            if not await self.initialize_client():
                return [types.TextContent(type="text", text="Error: Could not connect to VLLM server")]

        try:
            prompt = arguments["prompt"]
            model = arguments.get("model", self.available_models[0] if self.available_models else "auto")
            max_tokens = arguments.get("max_tokens", 512)
            temperature = arguments.get("temperature", 0.7)

            response = self.vllm_client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            result = {
                "response": response.choices[0].text,
                "model": response.model,
                "usage": response.usage.dict() if response.usage else None
            }

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in text completion: {str(e)}")]

    async def handle_get_models(self) -> List[types.TextContent]:
        """Handle get models requests"""
        if not self.vllm_client:
            if not await self.initialize_client():
                return [types.TextContent(type="text", text="Error: Could not connect to VLLM server")]

        try:
            models = self.vllm_client.models.list()
            model_list = [{"id": model.id, "object": model.object} for model in models.data]
            return [types.TextContent(type="text", text=json.dumps(model_list, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting models: {str(e)}")]

    async def handle_server_status(self) -> List[types.TextContent]:
        """Handle server status requests"""
        try:
            # Check health endpoint
            health_url = self.base_url.replace("/v1", "/health")
            response = requests.get(health_url, timeout=5)
            health_status = response.status_code == 200
        except:
            health_status = False

        try:
            # Check models endpoint
            if not self.vllm_client:
                await self.initialize_client()
            models = self.vllm_client.models.list()
            models_available = len(models.data) > 0
        except:
            models_available = False

        status = {
            "server_url": self.base_url,
            "health_status": "healthy" if health_status else "unhealthy",
            "models_available": models_available,
            "available_models": self.available_models
        }

        return [types.TextContent(type="text", text=json.dumps(status, indent=2))]
    
    # Weather Tools Implementation
    async def handle_get_weather(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle weather requests using OpenWeatherMap API"""
        try:
            city = arguments["city"]
            country = arguments.get("country", "")
            units = arguments.get("units", "metric")
            
            # Mock weather data (replace with actual API call)
            if self.weather_api_key:
                location = f"{city},{country}" if country else city
                url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units={units}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    weather_info = {
                        "city": data["name"],
                        "country": data["sys"]["country"],
                        "temperature": data["main"]["temp"],
                        "feels_like": data["main"]["feels_like"],
                        "humidity": data["main"]["humidity"],
                        "pressure": data["main"]["pressure"],
                        "description": data["weather"][0]["description"],
                        "wind_speed": data["wind"]["speed"],
                        "units": units
                    }
                else:
                    weather_info = {"error": f"Weather data not found for {city}"}
            else:
                # Mock data when no API key is provided
                weather_info = {
                    "city": city,
                    "country": country or "Unknown",
                    "temperature": 22.5,
                    "feels_like": 24.0,
                    "humidity": 65,
                    "pressure": 1013,
                    "description": "partly cloudy",
                    "wind_speed": 3.2,
                    "units": units,
                    "note": "Mock data - set weather_api_key for real data"
                }
            
            return [types.TextContent(type="text", text=json.dumps(weather_info, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting weather: {str(e)}")]

    async def handle_get_weather_forecast(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle weather forecast requests"""
        try:
            city = arguments["city"]
            country = arguments.get("country", "")
            units = arguments.get("units", "metric")
            
            # Mock forecast data (replace with actual API call)
            if self.weather_api_key:
                location = f"{city},{country}" if country else city
                url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={self.weather_api_key}&units={units}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    forecast = {
                        "city": data["city"]["name"],
                        "country": data["city"]["country"],
                        "forecast": []
                    }
                    
                    for item in data["list"][:5]:  # 5-day forecast
                        forecast["forecast"].append({
                            "date": item["dt_txt"],
                            "temperature": item["main"]["temp"],
                            "description": item["weather"][0]["description"],
                            "humidity": item["main"]["humidity"]
                        })
                else:
                    forecast = {"error": f"Forecast data not found for {city}"}
            else:
                # Mock forecast data
                forecast = {
                    "city": city,
                    "country": country or "Unknown",
                    "forecast": [
                        {"date": "2024-01-01 12:00:00", "temperature": 20, "description": "sunny", "humidity": 60},
                        {"date": "2024-01-02 12:00:00", "temperature": 18, "description": "cloudy", "humidity": 70},
                        {"date": "2024-01-03 12:00:00", "temperature": 22, "description": "partly cloudy", "humidity": 65},
                        {"date": "2024-01-04 12:00:00", "temperature": 19, "description": "rainy", "humidity": 80},
                        {"date": "2024-01-05 12:00:00", "temperature": 21, "description": "sunny", "humidity": 55}
                    ],
                    "units": units,
                    "note": "Mock data - set weather_api_key for real data"
                }
            
            return [types.TextContent(type="text", text=json.dumps(forecast, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error getting forecast: {str(e)}")]

    # Code Analysis Tools Implementation
    async def handle_analyze_python_code(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Analyze Python code for syntax, complexity, and style"""
        try:
            code = arguments["code"]
            check_syntax = arguments.get("check_syntax", True)
            check_complexity = arguments.get("check_complexity", True)
            check_style = arguments.get("check_style", True)
            
            analysis = {"code_analysis": {}}
            
            # Syntax check
            if check_syntax:
                try:
                    ast.parse(code)
                    analysis["code_analysis"]["syntax"] = {"valid": True, "message": "Syntax is valid"}
                except SyntaxError as e:
                    analysis["code_analysis"]["syntax"] = {
                        "valid": False, 
                        "error": str(e),
                        "line": e.lineno
                    }
            
            # Complexity analysis
            if check_complexity:
                lines = code.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                # Count functions and classes
                functions = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
                classes = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
                
                # Cyclomatic complexity (simplified)
                complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
                complexity_score = sum(code.count(keyword) for keyword in complexity_keywords)
                
                analysis["code_analysis"]["complexity"] = {
                    "lines_of_code": len(non_empty_lines),
                    "functions": functions,
                    "classes": classes,
                    "cyclomatic_complexity": complexity_score,
                    "complexity_rating": "low" if complexity_score < 10 else "medium" if complexity_score < 20 else "high"
                }
            
            # Style check (basic)
            if check_style:
                style_issues = []
                
                # Check for long lines
                long_lines = [i+1 for i, line in enumerate(code.split('\n')) if len(line) > 79]
                if long_lines:
                    style_issues.append(f"Lines too long (>79 chars): {long_lines}")
                
                # Check for missing docstrings in functions
                if 'def ' in code and '"""' not in code and "'''" not in code:
                    style_issues.append("Functions missing docstrings")
                
                # Check for inconsistent indentation
                lines = code.split('\n')
                indentations = []
                for line in lines:
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        if indent > 0:
                            indentations.append(indent)
                
                if indentations and len(set(indentations)) > 2:
                    style_issues.append("Inconsistent indentation detected")
                
                analysis["code_analysis"]["style"] = {
                    "issues": style_issues,
                    "score": "good" if not style_issues else "needs_improvement"
                }
            
            return [types.TextContent(type="text", text=json.dumps(analysis, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error analyzing code: {str(e)}")]

    async def handle_format_code(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Format and beautify code"""
        try:
            code = arguments["code"]
            language = arguments.get("language", "python")
            
            if language == "python":
                # Basic Python formatting
                try:
                    # Parse and unparse to normalize formatting
                    tree = ast.parse(code)
                    formatted_code = ast.unparse(tree)
                except:
                    # Fallback to basic formatting
                    lines = code.split('\n')
                    formatted_lines = []
                    indent_level = 0
                    
                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            formatted_lines.append('')
                            continue
                        
                        # Adjust indent level
                        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')):
                            formatted_lines.append('    ' * indent_level + stripped)
                            indent_level += 1
                        elif stripped in ['else:', 'elif ', 'except:', 'finally:']:
                            formatted_lines.append('    ' * (indent_level - 1) + stripped)
                        elif stripped.startswith(('return', 'break', 'continue', 'pass')):
                            formatted_lines.append('    ' * indent_level + stripped)
                        else:
                            formatted_lines.append('    ' * indent_level + stripped)
                    
                    formatted_code = '\n'.join(formatted_lines)
            
            elif language == "json":
                try:
                    parsed = json.loads(code)
                    formatted_code = json.dumps(parsed, indent=2)
                except:
                    formatted_code = code
            
            else:
                formatted_code = code  # No formatting for unsupported languages
            
            result = {
                "original_code": code,
                "formatted_code": formatted_code,
                "language": language
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error formatting code: {str(e)}")]

    async def handle_extract_functions(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Extract function definitions from Python code"""
        try:
            code = arguments["code"]
            
            try:
                tree = ast.parse(code)
                functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Extract function info
                        func_info = {
                            "name": node.name,
                            "line_number": node.lineno,
                            "arguments": [],
                            "returns": None,
                            "docstring": ast.get_docstring(node)
                        }
                        
                        # Extract arguments
                        for arg in node.args.args:
                            func_info["arguments"].append(arg.arg)
                        
                        # Extract return annotation if present
                        if node.returns:
                            func_info["returns"] = ast.unparse(node.returns)
                        
                        functions.append(func_info)
                
                result = {
                    "functions_found": len(functions),
                    "functions": functions
                }
                
            except SyntaxError as e:
                result = {
                    "error": f"Syntax error in code: {str(e)}",
                    "line": e.lineno
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error extracting functions: {str(e)}")]

    # Mathematics Tools Implementation
    async def handle_calculate_expression(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Safely evaluate mathematical expressions"""
        try:
            expression = arguments["expression"]
            precision = arguments.get("precision", 10)
            
            # Safe evaluation - only allow mathematical operations
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
            
            try:
                # Parse the expression to check for safety
                tree = ast.parse(expression, mode='eval')
                
                # Check for unsafe operations
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id not in allowed_names:
                        if node.id not in ['x', 'y', 'z']:  # Allow variables
                            raise ValueError(f"Unsafe operation: {node.id}")
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id not in allowed_names:
                            raise ValueError(f"Unsafe function: {node.func.id}")
                
                # Evaluate the expression
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                
                # Format result with specified precision
                if isinstance(result, float):
                    result = round(result, precision)
                
                calculation_result = {
                    "expression": expression,
                    "result": result,
                    "type": type(result).__name__,
                    "precision": precision
                }
                
            except (ValueError, SyntaxError, NameError) as e:
                calculation_result = {
                    "expression": expression,
                    "error": str(e),
                    "result": None
                }
            
            return [types.TextContent(type="text", text=json.dumps(calculation_result, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error calculating expression: {str(e)}")]

    async def handle_statistical_analysis(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Perform statistical analysis on a dataset"""
        try:
            data = arguments["data"]
            operations = arguments.get("operations", ["mean", "median", "std"])
            
            if not data:
                return [types.TextContent(type="text", text="Error: Empty dataset provided")]
            
            # Convert to float if needed
            try:
                data = [float(x) for x in data]
            except (ValueError, TypeError):
                return [types.TextContent(type="text", text="Error: All data points must be numeric")]
            
            stats_result = {
                "dataset_size": len(data),
                "statistics": {}
            }
            
            # Calculate requested statistics
            for operation in operations:
                try:
                    if operation == "mean":
                        stats_result["statistics"]["mean"] = statistics.mean(data)
                    elif operation == "median":
                        stats_result["statistics"]["median"] = statistics.median(data)
                    elif operation == "mode":
                        try:
                            stats_result["statistics"]["mode"] = statistics.mode(data)
                        except statistics.StatisticsError:
                            stats_result["statistics"]["mode"] = "No unique mode"
                    elif operation == "std":
                        if len(data) > 1:
                            stats_result["statistics"]["standard_deviation"] = statistics.stdev(data)
                        else:
                            stats_result["statistics"]["standard_deviation"] = 0
                    elif operation == "var":
                        if len(data) > 1:
                            stats_result["statistics"]["variance"] = statistics.variance(data)
                        else:
                            stats_result["statistics"]["variance"] = 0
                    elif operation == "min":
                        stats_result["statistics"]["minimum"] = min(data)
                    elif operation == "max":
                        stats_result["statistics"]["maximum"] = max(data)
                    elif operation == "range":
                        stats_result["statistics"]["range"] = max(data) - min(data)
                except Exception as e:
                    stats_result["statistics"][operation] = f"Error: {str(e)}"
            
            return [types.TextContent(type="text", text=json.dumps(stats_result, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in statistical analysis: {str(e)}")]

    async def handle_solve_equation(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Solve mathematical equations"""
        try:
            equation = arguments["equation"]
            equation_type = arguments.get("equation_type", "quadratic")
            
            # Clean up the equation
            equation = equation.replace(" ", "").replace("=", "==")
            
            if equation_type == "quadratic":
                # Parse quadratic equation: ax^2 + bx + c = 0
                # Simple regex parsing for standard form
                import re
                
                # Convert to standard form if needed
                if "==" in equation:
                    left, right = equation.split("==")
                    equation = f"({left})-({right})"
                
                # Extract coefficients (simplified approach)
                # This is a basic implementation - could be enhanced
                try:
                    # Look for patterns like ax^2, bx, c
                    x2_match = re.search(r'([+-]?\d*\.?\d*)x\^?2', equation)
                    x_match = re.search(r'([+-]?\d*\.?\d*)x(?!\^)', equation)
                    const_match = re.search(r'([+-]?\d+\.?\d*)(?!.*x)', equation)
                    
                    a = float(x2_match.group(1)) if x2_match and x2_match.group(1) else 1.0
                    b = float(x_match.group(1)) if x_match and x_match.group(1) else 0.0
                    c = float(const_match.group(1)) if const_match and const_match.group(1) else 0.0
                    
                    # Solve quadratic equation: ax^2 + bx + c = 0
                    discriminant = b**2 - 4*a*c
                    
                    if discriminant > 0:
                        x1 = (-b + math.sqrt(discriminant)) / (2*a)
                        x2 = (-b - math.sqrt(discriminant)) / (2*a)
                        solutions = [x1, x2]
                        solution_type = "two_real_solutions"
                    elif discriminant == 0:
                        x = -b / (2*a)
                        solutions = [x]
                        solution_type = "one_real_solution"
                    else:
                        real_part = -b / (2*a)
                        imag_part = math.sqrt(-discriminant) / (2*a)
                        solutions = [f"{real_part} + {imag_part}i", f"{real_part} - {imag_part}i"]
                        solution_type = "complex_solutions"
                    
                    result = {
                        "equation": arguments["equation"],
                        "type": equation_type,
                        "coefficients": {"a": a, "b": b, "c": c},
                        "discriminant": discriminant,
                        "solution_type": solution_type,
                        "solutions": solutions
                    }
                    
                except Exception as e:
                    result = {
                        "equation": arguments["equation"],
                        "error": f"Could not parse quadratic equation: {str(e)}",
                        "note": "Expected format: ax^2 + bx + c = 0"
                    }
            
            elif equation_type == "linear":
                # Parse linear equation: ax + b = 0
                try:
                    # Simple linear equation solver
                    if "==" in equation:
                        left, right = equation.split("==")
                        equation = f"({left})-({right})"
                    
                    # Extract coefficient and constant
                    x_match = re.search(r'([+-]?\d*\.?\d*)x', equation)
                    const_match = re.search(r'([+-]?\d+\.?\d*)(?!.*x)', equation)
                    
                    a = float(x_match.group(1)) if x_match and x_match.group(1) else 1.0
                    b = float(const_match.group(1)) if const_match and const_match.group(1) else 0.0
                    
                    if a == 0:
                        if b == 0:
                            solution = "infinite_solutions"
                        else:
                            solution = "no_solution"
                    else:
                        solution = -b / a
                    
                    result = {
                        "equation": arguments["equation"],
                        "type": equation_type,
                        "coefficients": {"a": a, "b": b},
                        "solution": solution
                    }
                    
                except Exception as e:
                    result = {
                        "equation": arguments["equation"],
                        "error": f"Could not parse linear equation: {str(e)}",
                        "note": "Expected format: ax + b = 0"
                    }
            
            else:
                result = {
                    "error": f"Unsupported equation type: {equation_type}",
                    "supported_types": ["quadratic", "linear"]
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error solving equation: {str(e)}")]

    async def handle_unit_conversion(self, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Convert between different units"""
        try:
            value = arguments["value"]
            from_unit = arguments["from_unit"].lower()
            to_unit = arguments["to_unit"].lower()
            category = arguments["category"].lower()
            
            conversion_factors = {
                "length": {
                    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                    "inch": 0.0254, "ft": 0.3048, "yard": 0.9144, "mile": 1609.34
                },
                "weight": {
                    "mg": 0.000001, "g": 0.001, "kg": 1.0, "ton": 1000.0,
                    "oz": 0.0283495, "lb": 0.453592, "stone": 6.35029
                },
                "volume": {
                    "ml": 0.001, "l": 1.0, "m3": 1000.0,
                    "cup": 0.236588, "pint": 0.473176, "quart": 0.946353, "gallon": 3.78541
                },
                "temperature": {
                    # Special handling needed for temperature
                }
            }
            
            if category == "temperature":
                # Temperature conversions need special handling
                if from_unit == "celsius":
                    if to_unit == "fahrenheit":
                        result_value = (value * 9/5) + 32
                    elif to_unit == "kelvin":
                        result_value = value + 273.15
                    else:
                        result_value = value
                elif from_unit == "fahrenheit":
                    if to_unit == "celsius":
                        result_value = (value - 32) * 5/9
                    elif to_unit == "kelvin":
                        result_value = (value - 32) * 5/9 + 273.15
                    else:
                        result_value = value
                elif from_unit == "kelvin":
                    if to_unit == "celsius":
                        result_value = value - 273.15
                    elif to_unit == "fahrenheit":
                        result_value = (value - 273.15) * 9/5 + 32
                    else:
                        result_value = value
                else:
                    raise ValueError(f"Unsupported temperature unit: {from_unit}")
                
                result = {
                    "original_value": value,
                    "original_unit": from_unit,
                    "converted_value": round(result_value, 6),
                    "converted_unit": to_unit,
                    "category": category
                }
            
            else:
                # Standard unit conversions using factors
                if category not in conversion_factors:
                    raise ValueError(f"Unsupported category: {category}")
                
                factors = conversion_factors[category]
                
                if from_unit not in factors:
                    raise ValueError(f"Unsupported unit '{from_unit}' in category '{category}'")
                if to_unit not in factors:
                    raise ValueError(f"Unsupported unit '{to_unit}' in category '{category}'")
                
                # Convert to base unit, then to target unit
                base_value = value * factors[from_unit]
                result_value = base_value / factors[to_unit]
                
                result = {
                    "original_value": value,
                    "original_unit": from_unit,
                    "converted_value": round(result_value, 6),
                    "converted_unit": to_unit,
                    "category": category,
                    "conversion_factor": factors[to_unit] / factors[from_unit]
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error in unit conversion: {str(e)}")]

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    """Main entry point"""
    server = VLLMInferenceServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())