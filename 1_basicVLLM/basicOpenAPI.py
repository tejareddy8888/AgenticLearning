# COPIED FROM: https://github.com/vllm-project/vllm/blob/main/examples/online_serving/api_client.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized Python client for chat completion API endpoints.

Example usage:
    python basic.py --prompt "Hello, how are you?" --stream
    python basic.py --model "gpt-3.5-turbo" --temperature 0.7 --max-tokens 100
"""

import argparse
import json
import sys
from argparse import Namespace
from collections.abc import Iterator
from typing import Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ChatCompletionClient:
    """Optimized client for chat completion API calls."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session(timeout)
    
    def _create_session(self, timeout: int) -> requests.Session:
        """Create a session with connection pooling and retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.timeout = timeout
        
        return session
    
    def create_chat_completion(
        self,
        messages: list[Dict[str, str]],
        model: str,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> requests.Response:
        """Create a chat completion request."""
        headers = {
            "User-Agent": "ChatCompletion-Client/1.0",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        url = f"{self.base_url}/v1/chat/completions"
        
        try:
            response = self.session.post(url, headers=headers, json=payload, stream=stream)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}", file=sys.stderr)
            raise
    
    def get_streaming_response(self, response: requests.Response) -> Iterator[str]:
        """Process streaming response and yield content chunks."""
        try:
            for chunk in response.iter_lines(
                chunk_size=8192, decode_unicode=False, delimiter=b"\n"
            ):
                if not chunk:
                    continue
                
                chunk_str = chunk.decode("utf-8").strip()
                
                if chunk_str.startswith("data: "):
                    chunk_str = chunk_str[6:]
                
                if chunk_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(chunk_str)
                    if self._has_content(data):
                        content = data["choices"][0]["delta"]["content"]
                        yield content
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            print(f"Error processing streaming response: {e}", file=sys.stderr)
    
    def get_response(self, response: requests.Response) -> str:
        """Extract complete response content."""
        try:
            data = response.json()
            if self._has_message_content(data):
                return data["choices"][0]["message"]["content"]
            return ""
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing response: {e}", file=sys.stderr)
            return ""
    
    def _has_content(self, data: Dict[str, Any]) -> bool:
        """Check if response data has streaming content."""
        return (
            "choices" in data 
            and len(data["choices"]) > 0 
            and "delta" in data["choices"][0]
            and "content" in data["choices"][0]["delta"]
        )
    
    def _has_message_content(self, data: Dict[str, Any]) -> bool:
        """Check if response data has message content."""
        return (
            "choices" in data 
            and len(data["choices"]) > 0 
            and "message" in data["choices"][0]
            and "content" in data["choices"][0]["message"]
        )


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chat completion API client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--host", type=str, default="localhost", 
                       help="API server host")
    parser.add_argument("--port", type=int, default=8000, 
                       help="API server port")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B-Instruct",
                       help="Model name to use")
    parser.add_argument("--prompt", type=str, default="What is 2+2?",
                       help="User prompt")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.",
                       help="System message")
    parser.add_argument("--stream", action="store_true",
                       help="Enable streaming response")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (0.0 to 2.0)")
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="Maximum tokens to generate")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds")
    
    return parser.parse_args()


def main(args: Namespace) -> None:
    """Main execution function."""
    base_url = f"http://{args.host}:{args.port}"
    client = ChatCompletionClient(base_url, timeout=args.timeout)
    
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt}
    ]
    
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Streaming: {args.stream}")
    print("-" * 50)
    
    try:
        response = client.create_chat_completion(
            messages=messages,
            model=args.model,
            stream=args.stream,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        if args.stream:
            print("Response (streaming):")
            for chunk in client.get_streaming_response(response):
                print(chunk, end="", flush=True)
            print()  # New line after streaming
        else:
            output = client.get_response(response)
            print(f"Response: {output}")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)