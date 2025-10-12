from typing import Annotated
from fastapi import Depends, Request
import httpx

def get_http(request: Request) -> httpx.AsyncClient:
    return request.app.state.http

HttpDep = Annotated[httpx.AsyncClient, Depends(get_http)]